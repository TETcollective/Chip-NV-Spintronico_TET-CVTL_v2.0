# =============================================================================
# TETcollective - HRV Metrics Comparison: Classico vs Quantistico
# File: code/qutip_hrv_metrics_comparison.py
# Descrizione: Bar plot comparativo SDNN, RMSSD, HF power da valori tipici/simulati.
#              Usa dati proxy dal testo (range healthy + deviazioni quantistiche).
# Autore: Assistito da Grok per @PhysSoliman (TETcollective.org)
# Data: Marzo 2026
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# Dati coerenti con la sezione "Effetto squeezing quantistico"
metrics = ['SDNN (ms)', 'RMSSD (ms)', 'HF power (norm)']

classico    = [85,  45,  1.00]   # proxy healthy classico
quantistico = [107, 59,  1.28]   # con squeezing (+26%, +31%, +28%)

x = np.arange(len(metrics))
width = 0.37

fig, ax = plt.subplots(figsize=(10, 6.8))

# Barre
ax.bar(x - width/2, classico,    width, label='Classico (proxy healthy)', 
       color='#1f77b4', alpha=0.88, edgecolor='white', linewidth=0.8)
ax.bar(x + width/2, quantistico, width, label='Quantistico embodied (squeezing)', 
       color='#d62728', alpha=0.88, edgecolor='white', linewidth=0.8)

ax.set_ylabel('Valore', fontsize=13)
ax.set_title('Confronto metriche HRV: Classico vs Quantistico\n'
             '(respirazione 0.1 Hz coherence + squeezing quantistico)',
             fontsize=14, pad=20)

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11.5)
ax.legend(fontsize=10.5, framealpha=0.95, loc='upper right', bbox_to_anchor=(0.98, 0.98))
ax.grid(True, axis='y', alpha=0.22, linestyle=':')

# Annotazioni percentuali su TUTTE le barre (corrette e visibili)
for i in range(len(metrics)):
    pct = ((quantistico[i] - classico[i]) / classico[i]) * 100
    y_pos = quantistico[i] + (3 if i < 2 else 0.07)
    ax.text(x[i] + width/2, y_pos, f'{pct:+.0f}%',
            ha='center', va='bottom', fontsize=11.5, fontweight='bold', color='#d62728')

ax.set_ylim(0, 125)  # respiro extra in alto

plt.tight_layout()
plt.savefig('figures/hrv_metrics_comparison.png', dpi=400, bbox_inches='tight')
plt.close()

print("Figura salvata: figures/hrv_metrics_comparison.png")
print("   → SDNN quantistico +26%")
print("   → RMSSD quantistico +31%")
print("   → HF power quantistico +28%")