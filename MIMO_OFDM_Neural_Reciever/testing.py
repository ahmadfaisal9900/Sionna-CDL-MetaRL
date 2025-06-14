import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sionna.phy.utils import sim_ber
from MIMO_OFDM_NRX_Testing import (
    UL_SIMS,
    training_batch_size,
    model_weights_path_conventional_training,
    results_filename,
)
# import your “untrained” block‐diagram model
from MIMO_OFDM import Model as BaselineModel
# import your learned neural receiver
from MIMO_OFDM import NeuralModel

# 1) SNR grid and CDL list
ebno_dbs  = np.array(UL_SIMS["ebno_db"], np.float32)
cdl_models = UL_SIMS["cdl_model"]

# 2) Run baseline
BER_base, BLER_base = {}, {}
print("Running untrained (baseline) E2E …")
for cdl in cdl_models:
    print("  baseline CDL", cdl)
    m0 = BaselineModel(domain=UL_SIMS["domain"],
                       direction=UL_SIMS["direction"],
                       cdl_model=cdl,
                       delay_spread=UL_SIMS["delay_spread"],
                       perfect_csi=UL_SIMS["perfect_csi"],
                       speed=UL_SIMS["speed"],
                       cyclic_prefix_length=UL_SIMS["cyclic_prefix_length"],
                       pilot_ofdm_symbol_indices=UL_SIMS["pilot_ofdm_symbol_indices"])
    # no need to build, sim_ber will call .call for you
    ber0, bler0 = sim_ber(m0,
                         ebno_dbs,
                         batch_size=int(training_batch_size.numpy()),
                         num_target_block_errors=1000,
                         max_mc_iter=1000)
    BER_base[cdl]  = ber0.numpy()
    BLER_base[cdl] = bler0.numpy()

# 3) Run trained neural receiver
BER_nn, BLER_nn = {}, {}
print("Running trained NeuralModel …")
for cdl in cdl_models:
    print("  trained CDL", cdl)
    m1 = NeuralModel(domain=UL_SIMS["domain"],
                     direction=UL_SIMS["direction"],
                     cdl_model=cdl,
                     delay_spread=UL_SIMS["delay_spread"],
                     perfect_csi=UL_SIMS["perfect_csi"],
                     speed=UL_SIMS["speed"],
                     cyclic_prefix_length=UL_SIMS["cyclic_prefix_length"],
                     pilot_ofdm_symbol_indices=UL_SIMS["pilot_ofdm_symbol_indices"],
                     training=False)
    # build so we can call set_weights
    _ = m1(training_batch_size,
           tf.constant(ebno_dbs[0], tf.float32))
    with open(model_weights_path_conventional_training, "rb") as f:
        w = pickle.load(f)
    m1._demapper.set_weights(w)
    ber1, bler1 = sim_ber(m1,
                         ebno_dbs,
                         batch_size=int(training_batch_size.numpy()),
                         num_target_block_errors=1000,
                         max_mc_iter=1000)
    BER_nn[cdl]  = ber1.numpy()
    BLER_nn[cdl] = bler1.numpy()

# 4) Plot together
plt.figure(figsize=(6,4))
for cdl in cdl_models:
    plt.semilogy(ebno_dbs, BER_base[cdl], "--", label=f"{cdl} baseline")
    plt.semilogy(ebno_dbs, BER_nn[cdl],  "-",   label=f"{cdl} trained")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.xlabel("Eb/No [dB]")
plt.ylabel("BER")
plt.title("Baseline vs. Trained Neural Demapper")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
for cdl in cdl_models:
    plt.semilogy(ebno_dbs, BLER_base[cdl], "--", label=f"{cdl} baseline")
    plt.semilogy(ebno_dbs, BLER_nn[cdl],  "-",   label=f"{cdl} trained")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.xlabel("Eb/No [dB]")
plt.ylabel("BLER")
plt.title("Baseline vs. Trained Neural Demapper")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()