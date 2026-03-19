# File: code/plot_from_qualia_json.py
# TET–CVTL – Legge JSON metriche qualia e genera plot per paper
# Output: qualia_curvature_plots_from_json.pdf

import json
import matplotlib.pyplot as plt
import numpy as np

# Carica JSON
with open("qualia_curvature_metrics.json", "r") as f:
    data = json.load(f)

t = np.array(data["time_series"]["t"])
conc_base = np.array(data["time_series"]["concurrence_baseline"])
conc_ren = np.array(data["time_series"]["concurrence_renascent"])
Kq_ren = np.array(data["time_series"]["Kq_renascent"])

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Concurrence
axs[0].plot(t, conc_base, 'b-', lw=2, label=f'Baseline ({conc_base[-1]:.4f})')
axs[0].plot(t, conc_ren, color='gold', lw=2.5, label=f'RENASCENT-Q ({conc_ren[-1]:.4f})')
axs[0].set_ylabel('Concurrence')
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# Curvatura qualia
axs[1].plot(t, Kq_ren, color='darkorange', lw=2, label=f'K_q RENASCENT-Q (max {np.max(Kq_ren):.3f})')
axs[1].set_xlabel('Tempo (unità adimensionali)')
axs[1].set_ylabel('Curvatura qualia locale (bit/μm²)')
axs[1].legend()
axs[1].grid(True, alpha=0.3)

plt.suptitle('Metriche curvatura qualia da JSON – TET–CVTL v2.0')
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig("qualia_curvature_plots_from_json.pdf", dpi=300, bbox_inches='tight')
print("Plot salvato: qualia_curvature_plots_from_json.pdf")