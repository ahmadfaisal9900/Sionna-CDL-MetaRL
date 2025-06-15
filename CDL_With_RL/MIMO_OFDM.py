# filepath: MIMO_OFDM.py
import os
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
try:
    import sionna.phy
except ImportError as e:
    import sys
    if 'google.colab' in sys.modules:
       # Install Sionna in Google Colab
       print("Installing Sionna and restarting the runtime. Please run the cell again.")
       os.system("pip install sionna")
       os.kill(os.getpid(), 5)
    else:
       raise e

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense

from sionna.phy import Block
from sionna.phy.channel import AWGN
from sionna.phy.utils import ebnodb2no, log10, expand_to_rank
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, Constellation, BinarySource
from sionna.phy.utils import sim_ber

sionna.phy.config.seed = 42 # Set seed for reproducible random number generation

import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

from sionna.phy import Block
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, \
                            OFDMModulator, OFDMDemodulator, RZFPrecoder, RemoveNulledSubcarriers
from sionna.phy.channel.tr38901 import AntennaArray, CDL
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel, \
                               time_lag_discrete_time_channel, ApplyOFDMChannel, ApplyTimeChannel, \
                               OFDMChannel, TimeChannel
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, BinarySource
from sionna.phy.utils import ebnodb2no, sim_ber, compute_ber
from tqdm import tqdm

###############################################
# SNR range for evaluation and training [dB]
###############################################
ebno_db_min = 4.0
ebno_db_max = 8.0
###############################################
# Modulation and coding configuration
###############################################
num_bits_per_symbol = 6 # Baseline is 64-QAM
modulation_order = 2**num_bits_per_symbol
coderate = 0.5 # Coderate for the outer code
n = 1500 # Codeword length [bit]. Must be a multiple of num_bits_per_symbol
num_symbols_per_codeword = n//num_bits_per_symbol # Number of modulated baseband symbols per codeword
k = int(n*coderate) # Number of information bits per codeword

###############################################
# Training configuration
###############################################
num_training_iterations_conventional = 200000 # Number of training iterations for conventional training
# Number of training iterations with RL-based training for the alternating training phase and fine-tuning of the receiver phase
num_training_iterations_rl_alt = 7000
num_training_iterations_rl_finetuning = 3000
training_batch_size = tf.constant(32, tf.int32) # Training batch size
rl_perturbation_var = 0.01 # Variance of the perturbation used for RL-based training of the transmitter
model_weights_path_conventional_training = "awgn_autoencoder_weights_conventional_training" # Filename to save the autoencoder weights once conventional training is done
model_weights_path_rl_training = "awgn_autoencoder_weights_rl_training" # Filename to save the autoencoder weights once RL-based training is done

###############################################
# Evaluation configuration
###############################################
results_filename = "awgn_autoencoder_results" # Location to save the results

class NeuralDemapper(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # Increase network capacity to handle complex demapping
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(512, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1440)  # Match codeword length
        
    def call(self, x_hat, no_eff):
        # Extract real and imaginary parts of the complex tensor
        x_hat_real = tf.math.real(x_hat)
        x_hat_imag = tf.math.imag(x_hat)
        
        # Concatenate real part, imaginary part, and noise variance
        inputs = tf.concat([x_hat_real, x_hat_imag, no_eff], axis=-1)
        
        # Process through neural network
        x = self.dense1(inputs)
        x = self.dense2(x)
        llr = self.output_layer(x)
        return llr  # Shape will now be [batch_size, 1, 4, 1440]

class NeuralModel(Block):
    """This block simulates OFDM MIMO transmissions over the CDL model.

    Parameters
    ----------
    domain : One of ["time", "freq"], str
        Determines if the channel is modeled in the time or frequency domain.

    direction : One of ["uplink", "downlink"], str
        For "uplink", the UT transmits. For "downlink" the BS transmits.

    cdl_model : One of ["A", "B", "C", "D", "E"], str
        The CDL model to use.

    delay_spread : float
        The nominal delay spread [s].

    perfect_csi : bool
        Indicates if perfect CSI at the receiver should be assumed.

    speed : float
        The UT speed [m/s].

    cyclic_prefix_length : int
        The length of the cyclic prefix in number of samples.

    pilot_ofdm_symbol_indices : list, int
        List of integers defining the OFDM symbol indices that are reserved
        for pilots.

    subcarrier_spacing : float
        The subcarrier spacing [Hz]. Defaults to 15e3.
    """

    def __init__(self,
                 domain,
                 direction,
                 cdl_model,
                 delay_spread,
                 perfect_csi,
                 speed,
                 cyclic_prefix_length,
                 pilot_ofdm_symbol_indices,
                 subcarrier_spacing = 15e3,
                 training = True
                ):
        super().__init__()

        # Provided parameters
        self._domain = domain
        self._direction = direction
        self._cdl_model = cdl_model
        self._delay_spread = delay_spread
        self._perfect_csi = perfect_csi
        self._speed = speed
        self._cyclic_prefix_length = cyclic_prefix_length
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices
        self._training = training

        # System parameters
        self._carrier_frequency = 2.6e9
        self._subcarrier_spacing = subcarrier_spacing
        self._fft_size = 72
        self._num_ofdm_symbols = 14
        self._num_ut_ant = 4 # Must be a multiple of two as dual-polarized antennas are used
        self._num_bs_ant = 8 # Must be a multiple of two as dual-polarized antennas are used
        self._num_streams_per_tx = self._num_ut_ant
        self._dc_null = True
        self._num_guard_carriers = [5, 6]
        self._pilot_pattern = "kronecker"
        self._pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices
        self._num_bits_per_symbol = 2
        self._coderate = 0.5

        # Required system components
        self._sm = StreamManagement(np.array([[1]]), self._num_streams_per_tx)

        self._rg = ResourceGrid(num_ofdm_symbols=self._num_ofdm_symbols,
                                fft_size=self._fft_size,
                                subcarrier_spacing = self._subcarrier_spacing,
                                num_tx=1,
                                num_streams_per_tx=self._num_streams_per_tx,
                                cyclic_prefix_length=self._cyclic_prefix_length,
                                num_guard_carriers=self._num_guard_carriers,
                                dc_null=self._dc_null,
                                pilot_pattern=self._pilot_pattern,
                                pilot_ofdm_symbol_indices=self._pilot_ofdm_symbol_indices)

        self._n = int(self._rg.num_data_symbols * self._num_bits_per_symbol)
        self._k = int(self._n * self._coderate)

        self._ut_array = AntennaArray(num_rows=1,
                                      num_cols=int(self._num_ut_ant/2),
                                      polarization="dual",
                                      polarization_type="cross",
                                      antenna_pattern="38.901",
                                      carrier_frequency=self._carrier_frequency)

        self._bs_array = AntennaArray(num_rows=1,
                                      num_cols=int(self._num_bs_ant/2),
                                      polarization="dual",
                                      polarization_type="cross",
                                      antenna_pattern="38.901",
                                      carrier_frequency=self._carrier_frequency)

        self._cdl = CDL(model=self._cdl_model,
                        delay_spread=self._delay_spread,
                        carrier_frequency=self._carrier_frequency,
                        ut_array=self._ut_array,
                        bs_array=self._bs_array,
                        direction=self._direction,
                        min_speed=self._speed)

        self._frequencies = subcarrier_frequencies(self._rg.fft_size, self._rg.subcarrier_spacing)

        if self._domain == "freq":
            self._channel_freq = ApplyOFDMChannel(add_awgn=True)

        elif self._domain == "time":
            self._l_min, self._l_max = time_lag_discrete_time_channel(self._rg.bandwidth)
            self._l_tot = self._l_max - self._l_min + 1
            self._channel_time = ApplyTimeChannel(self._rg.num_time_samples,
                                                  l_tot=self._l_tot,
                                                  add_awgn=True)
            self._modulator = OFDMModulator(self._cyclic_prefix_length)
            self._demodulator = OFDMDemodulator(self._fft_size, self._l_min, self._cyclic_prefix_length)

        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(self._k, self._n)
        self._mapper = Mapper("qam", self._num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(self._rg)

        if self._direction == "downlink":
            self._zf_precoder = RZFPrecoder(self._rg, self._sm, return_effective_channel=True)

        self._ls_est = LSChannelEstimator(self._rg, interpolation_type="nn")
        self._lmmse_equ = LMMSEEqualizer(self._rg, self._sm)
        self._demapper = NeuralDemapper()
        if not self._training:
            self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)
        
        #################
        # Loss function
        #################
        if self._training:
            self._bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)      
        self._remove_nulled_scs = RemoveNulledSubcarriers(self._rg)
        
    @tf.function # Run in graph mode
    def call(self, batch_size, ebno_db):

        no = ebnodb2no(ebno_db, self._num_bits_per_symbol, self._coderate, self._rg)
        if self._training:
            c = self._binary_source([batch_size, 1, self._num_streams_per_tx, self._n])
        else:
            b = self._binary_source([batch_size, 1, self._num_streams_per_tx, self._k])
            c = self._encoder(b)
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        if self._domain == "time":
            # Time-domain simulations
            a, tau = self._cdl(batch_size, self._rg.num_time_samples+self._l_tot-1, self._rg.bandwidth)
            h_time = cir_to_time_channel(self._rg.bandwidth, a, tau,
                                         l_min=self._l_min, l_max=self._l_max, normalize=True)

            a_freq = a[...,self._rg.cyclic_prefix_length:-1:(self._rg.fft_size+self._rg.cyclic_prefix_length)]
            a_freq = a_freq[...,:self._rg.num_ofdm_symbols]
            h_freq = cir_to_ofdm_channel(self._frequencies, a_freq, tau, normalize=True)

            if self._direction == "downlink":
                x_rg, g = self._zf_precoder(x_rg, h_freq)

            x_time = self._modulator(x_rg)
            y_time = self._channel_time(x_time, h_time, no)

            y = self._demodulator(y_time)

        elif self._domain == "freq":
            # Frequency-domain simulations
            cir = self._cdl(batch_size, self._rg.num_ofdm_symbols, 1/self._rg.ofdm_symbol_duration)
            h_freq = cir_to_ofdm_channel(self._frequencies, *cir, normalize=True)

            if self._direction == "downlink":
                x_rg, g = self._zf_precoder(x_rg, h_freq)

            y = self._channel_freq(x_rg, h_freq, no)

        if self._perfect_csi:
            if self._direction == "uplink":
                h_hat = self._remove_nulled_scs(h_freq)
            elif self._direction =="downlink":
                h_hat = g
            err_var = 0.0
        else:
            h_hat, err_var = self._ls_est(y, no)

        x_hat, no_eff = self._lmmse_equ(y, h_hat, err_var, no)
        llr = self._demapper(x_hat, no_eff)
        
        if self._training:
            llr_reshaped = tf.reshape(llr, [tf.shape(llr)[0], 1, 4, 1440])
            loss = self._bce(c, llr_reshaped)
            return loss
        else:
            llr_reshaped = tf.reshape(llr, [tf.shape(llr)[0], 1, 4, 1440])
            b_hat = self._decoder(llr_reshaped)
            return b, b_hat

# RL training utilities
def apply_perturbations(model, perturbation_var):
    """Apply random perturbations to model weights for exploration"""
    original_weights = model.get_weights()
    perturbed_weights = []
    
    for w in original_weights:
        # Add Gaussian noise with specified variance
        perturbation = tf.random.normal(shape=w.shape, mean=0.0, stddev=np.sqrt(perturbation_var))
        perturbed_weights.append(w + perturbation)
    
    return original_weights, perturbed_weights

def compute_reward(model, batch_size, ebno_db):
    """Compute reward as negative loss (higher reward for lower loss)"""
    loss = model(batch_size, ebno_db)
    return -loss

def rl_training_step(model, optimizer, batch_size, ebno_db, perturbation_var):
    """Perform one step of RL training using REINFORCE-like algorithm"""
    # Save original weights
    original_weights, perturbed_weights = apply_perturbations(model._demapper, perturbation_var)
    
    # Evaluate original performance
    original_loss = model(batch_size, ebno_db)
    original_reward = -original_loss
    
    # Try perturbed weights
    model._demapper.set_weights(perturbed_weights)
    perturbed_loss = model(batch_size, ebno_db)
    perturbed_reward = -perturbed_loss
    
    # Policy gradient update
    if perturbed_reward > original_reward:
        # Keep the better weights
        pass  # Perturbation was good, already applied
    else:
        # Revert to original weights
        model._demapper.set_weights(original_weights)
    
    # Perform additional gradient-based optimization
    with tf.GradientTape() as tape:
        loss = model(batch_size, ebno_db)
    
    # Apply gradients to fine-tune
    weights = tape.watched_variables()
    grads = tape.gradient(loss, weights)
    optimizer.apply_gradients(zip(grads, weights))
    
    return loss

UL_SIMS = {
    "ebno_db": list(np.arange(-5, 20, 4.0)),
    "cdl_model": ["A", "B", "C", "D", "E"],
    "delay_spread": 100e-9,
    "domain": "freq",
    "direction": "uplink",
    "perfect_csi": False,
    "speed": 0.0,
    "cyclic_prefix_length": 6,
    "pilot_ofdm_symbol_indices": [2, 11],
    "ber": [],
    "bler": [],
    "duration": None
}

if __name__ == "__main__":
    training = True  # Change to False to evaluate instead of train
    
    if training:
        # Initialize the model
        model = NeuralModel(domain=UL_SIMS["domain"],
                          direction=UL_SIMS["direction"],
                          cdl_model=UL_SIMS["cdl_model"][0],  # Use model A for training
                          delay_spread=UL_SIMS["delay_spread"],
                          perfect_csi=UL_SIMS["perfect_csi"],
                          speed=UL_SIMS["speed"],
                          cyclic_prefix_length=UL_SIMS["cyclic_prefix_length"],
                          pilot_ofdm_symbol_indices=UL_SIMS["pilot_ofdm_symbol_indices"],
                          training=True)

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        
        print("Starting conventional training...")
        # Phase 1: Conventional training
        for i in tqdm(range(num_training_iterations_conventional)):
            ebno_db = tf.random.uniform([], ebno_db_min, ebno_db_max)
            with tf.GradientTape() as tape:
                loss = model(training_batch_size, ebno_db)
            weights = tape.watched_variables()
            grads = tape.gradient(loss, weights)
            optimizer.apply_gradients(zip(grads, weights))
            
            if i % 1000 == 0:
                print(f"Iteration {i}/{num_training_iterations_conventional}, Loss: {loss.numpy():.4f}")
        
        # Save weights after conventional training
        demapper_weights = model._demapper.get_weights()
        with open(model_weights_path_conventional_training, 'wb') as f:
            pickle.dump(demapper_weights, f)
        print(f"Saved conventional training weights to {model_weights_path_conventional_training}")
        
        print("Starting RL-based alternating training...")
        # Phase 2: RL-based alternating training
        for i in tqdm(range(num_training_iterations_rl_alt)):
            ebno_db = tf.random.uniform([], ebno_db_min, ebno_db_max)
            loss = rl_training_step(model, optimizer, training_batch_size, ebno_db, rl_perturbation_var)
            
            if i % 500 == 0:
                print(f"RL Alt Iteration {i}/{num_training_iterations_rl_alt}, Loss: {loss.numpy():.4f}")
        
        print("Starting RL-based fine-tuning...")
        # Phase 3: RL-based fine-tuning
        for i in tqdm(range(num_training_iterations_rl_finetuning)):
            ebno_db = tf.random.uniform([], ebno_db_min, ebno_db_max)
            # Use smaller perturbations during fine-tuning
            loss = rl_training_step(model, optimizer, training_batch_size, ebno_db, rl_perturbation_var/10)
            
            if i % 500 == 0:
                print(f"RL Fine-tuning Iteration {i}/{num_training_iterations_rl_finetuning}, Loss: {loss.numpy():.4f}")
        
        # Save weights after RL training
        demapper_weights = model._demapper.get_weights()
        with open(model_weights_path_rl_training, 'wb') as f:
            pickle.dump(demapper_weights, f)
        print(f"Saved RL training weights to {model_weights_path_rl_training}")
        
    else:
        # Evaluation mode
        print("Starting evaluation across all CDL models...")
        
        # Loop through all CDL models
        for cdl in UL_SIMS["cdl_model"]:
            # Create model for evaluation (training=False)
            model = NeuralModel(domain=UL_SIMS["domain"],
                              direction=UL_SIMS["direction"],
                              cdl_model=cdl,
                              delay_spread=UL_SIMS["delay_spread"],
                              perfect_csi=UL_SIMS["perfect_csi"],
                              speed=UL_SIMS["speed"],
                              cyclic_prefix_length=UL_SIMS["cyclic_prefix_length"],
                              pilot_ofdm_symbol_indices=UL_SIMS["pilot_ofdm_symbol_indices"],
                              training=False)
            
            # Load trained weights
            try:
                with open(model_weights_path_rl_training, 'rb') as f:
                    trained_weights = pickle.load(f)
                model._demapper.set_weights(trained_weights)
                print(f"Loaded RL-trained weights for evaluation")
            except:
                print(f"Could not load RL weights, using conventional training weights")
                try:
                    with open(model_weights_path_conventional_training, 'rb') as f:
                        trained_weights = pickle.load(f)
                    model._demapper.set_weights(trained_weights)
                except:
                    print("Could not load any trained weights, using random initialization")
            
            # Test across different SNR points
            for ebno_db in UL_SIMS["ebno_db"]:
                print(f"Testing CDL model {cdl} at Eb/No = {ebno_db} dB")
                b, b_hat = model(training_batch_size, ebno_db)
                ber = compute_ber(b, b_hat)
                UL_SIMS["ber"].append(ber)
                print(f"CDL {cdl} | Eb/No {ebno_db:.1f} dB → BER {ber:.3e}")
        
        # Save results
        with open(results_filename + ".pkl", 'wb') as f:
            pickle.dump(UL_SIMS, f)
        print(f"Evaluation complete. Results saved to {results_filename}.pkl")