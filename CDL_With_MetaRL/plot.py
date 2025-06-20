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