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

# Load SNR grid and CDL models from your UL_SIMS dict
ebno_dbs = np.array(UL_SIMS["ebno_db"])
cdl_models = UL_SIMS["cdl_model"]

# Prepare result storage
BER = {}
BLER = {}

# Loop over CDL models
for cdl in cdl_models:
    print(f"Evaluating CDL model {cdl}...")
    # Instantiate in inference mode
    model = NeuralModel(domain=UL_SIMS["domain"],
                        direction=UL_SIMS["direction"],
                        cdl_model=cdl,
                        delay_spread=UL_SIMS["delay_spread"],
                        perfect_csi=UL_SIMS["perfect_csi"],
                        speed=UL_SIMS["speed"],
                        cyclic_prefix_length=UL_SIMS["cyclic_prefix_length"],
                        pilot_ofdm_symbol_indices=UL_SIMS["pilot_ofdm_symbol_indices"],
                        training=False)

    _ = model(training_batch_size,
            tf.constant(ebno_dbs[0], tf.float32))

    # 2) now you can safely assign the weights you trained
    with open(model_weights_path_conventional_training, 'rb') as f:
        demapper_w = pickle.load(f)
    model._demapper.set_weights(demapper_w)
    # Simulate BER and BLER over the SNR range
    ber, bler = sim_ber(model,
                        ebno_dbs,
                        batch_size=int(training_batch_size.numpy()),
                        num_target_block_errors=1000,
                        max_mc_iter=1000)

    BER[cdl]  = ber.numpy()
    BLER[cdl] = bler.numpy()

    print(f"  → done, BER: {BER[cdl]}\n")

# Save all results
out = (ebno_dbs, BER, BLER)
with open(results_filename + "_cdl.pkl", 'wb') as f:
    pickle.dump(out, f)

print("CDL evaluation complete. Results saved to", results_filename + "_cdl.pkl")
import pickle
import matplotlib.pyplot as plt

# 1) Load the results
with open("awgn_autoencoder_results_cdl.pkl", "rb") as f:
    ebno_dbs, BER, BLER = pickle.load(f)
    
# 2) Plot BER curves
plt.figure(figsize=(6,4))
for model, ber in BER.items():
    plt.semilogy(ebno_dbs, ber, marker='o', label=model)
plt.grid(True, which='both', ls='--', lw=0.5)
plt.xlabel("Eb/No [dB]")
plt.ylabel("BER")
plt.title("BER vs Eb/No (CDL Models)")
plt.legend(title="Model")
plt.tight_layout()

# 3) Plot BLER curves
plt.figure(figsize=(6,4))
for model, bler in BLER.items():
    plt.semilogy(ebno_dbs, bler, marker='s', label=model)
plt.grid(True, which='both', ls='--', lw=0.5)
plt.xlabel("Eb/No [dB]")
plt.ylabel("BLER")
plt.title("BLER vs Eb/No (CDL Models)")
plt.legend(title="Model")
plt.tight_layout()

plt.show()