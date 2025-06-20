from sionna.phy import Block
from sionna.phy.channel import AWGN
from sionna.phy.utils import ebnodb2no, log10, expand_to_rank
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, Constellation, BinarySource
from sionna.phy.utils import sim_ber
import sionna
sionna.phy.config.seed = 42 # Set seed for reproducible random number generation

import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib.pyplot as plt
import numpy as np
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

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense
###########################

import numpy as np
import pickle
import tensorflow as tf

from MIMO_OFDM import (NeuralModel,
                       UL_SIMS,
                       training_batch_size,
                       model_weights_path_conventional_training,
                       results_filename)
from sionna.phy.utils import sim_ber
import tensorflow as tf
import numpy as np
import pickle
from tqdm import tqdm

def evaluate_adaptation(meta_model, cdl_model, ebno_db_range, adaptation_steps=5):
    """
    Evaluates the meta-model's adaptation capabilities on a specific CDL model.
    
    Parameters:
    -----------
    meta_model: NeuralModel
        Pre-trained meta-model
    cdl_model: str
        CDL model to test on ("A", "B", "C", "D", or "E")
    ebno_db_range: list
        Range of SNR values to evaluate
    adaptation_steps: int
        Number of adaptation steps to perform before evaluation
        
    Returns:
    --------
    ber_results: dict
        Dictionary mapping SNR values to BER before and after adaptation
    """
    results = {}
    
    # Create evaluation model with specified CDL model
    eval_model = NeuralModel(
        domain=UL_SIMS["domain"],
        direction=UL_SIMS["direction"],
        cdl_model=cdl_model,
        delay_spread=UL_SIMS["delay_spread"],
        perfect_csi=UL_SIMS["perfect_csi"],
        speed=UL_SIMS["speed"],
        cyclic_prefix_length=UL_SIMS["cyclic_prefix_length"],
        pilot_ofdm_symbol_indices=UL_SIMS["pilot_ofdm_symbol_indices"],
        training=True  # For adaptation
    )
    
    # Copy meta-learned weights
    eval_model._demapper.set_weights(meta_model._demapper.get_weights())
    
    # Evaluation batch size
    batch_size = 32
    
    for ebno_db in ebno_db_range:
        print(f"Evaluating CDL model {cdl_model} at {ebno_db} dB...")
        
        # Measure BER before adaptation
        eval_model_test = NeuralModel(
            domain=UL_SIMS["domain"],
            direction=UL_SIMS["direction"],
            cdl_model=cdl_model,
            delay_spread=UL_SIMS["delay_spread"],
            perfect_csi=UL_SIMS["perfect_csi"],
            speed=UL_SIMS["speed"],
            cyclic_prefix_length=UL_SIMS["cyclic_prefix_length"],
            pilot_ofdm_symbol_indices=UL_SIMS["pilot_ofdm_symbol_indices"],
            training=False  # For testing
        )
        eval_model_test._demapper.set_weights(eval_model._demapper.get_weights())
        b, b_hat = eval_model_test(batch_size, ebno_db)
        ber_before = compute_ber(b, b_hat).numpy()
        
        # Perform adaptation steps
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        for _ in range(adaptation_steps):
            with tf.GradientTape() as tape:
                loss = eval_model(batch_size, ebno_db)
            
            weights = eval_model._demapper.trainable_variables
            grads = tape.gradient(loss, weights)
            optimizer.apply_gradients(zip(grads, weights))
        
        # Measure BER after adaptation
        eval_model_test._demapper.set_weights(eval_model._demapper.get_weights())
        b, b_hat = eval_model_test(batch_size, ebno_db)
        ber_after = compute_ber(b, b_hat).numpy()
        
        results[ebno_db] = {
            'ber_before': ber_before,
            'ber_after': ber_after,
            'improvement': (ber_before - ber_after) / ber_before * 100  # % improvement
        }
        
        print(f"  BER before adaptation: {ber_before:.6f}")
        print(f"  BER after adaptation:  {ber_after:.6f}")
        print(f"  Improvement: {results[ebno_db]['improvement']:.2f}%")
    
    return results

# Example usage:
if __name__ == "__main__":
    # Load meta-trained model
    meta_model = NeuralModel(
        domain=UL_SIMS["domain"],
        direction=UL_SIMS["direction"],
        cdl_model="A",  # Default model
        delay_spread=UL_SIMS["delay_spread"],
        perfect_csi=UL_SIMS["perfect_csi"],
        speed=UL_SIMS["speed"],
        cyclic_prefix_length=UL_SIMS["cyclic_prefix_length"],
        pilot_ofdm_symbol_indices=UL_SIMS["pilot_ofdm_symbol_indices"],
        training=True
    )
    
    # Initialize model
    meta_model(1, tf.constant(10.0))
    
    # Load meta-trained weights
    with open(f"{model_weights_path_metarl_training}_final.pkl", 'rb') as f:
        demapper_weights = pickle.load(f)
        meta_model._demapper.set_weights(demapper_weights)
    
    # Evaluate adaptation on all CDL models
    all_results = {}
    for cdl_model in ["A", "B", "C", "D", "E"]:
        print(f"\nTesting adaptation on CDL model {cdl_model}")
        results = evaluate_adaptation(
            meta_model, 
            cdl_model, 
            ebno_db_range=[0.0, 4.0, 8.0, 12.0]
        )
        all_results[cdl_model] = results
    
    # Save results
    with open("cdl_meta_adaptation_results.pkl", 'wb') as f:
        pickle.dump(all_results, f)