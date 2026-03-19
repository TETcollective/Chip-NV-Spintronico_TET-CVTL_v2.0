#!/usr/bin/env python3
"""
plot_golden_spiral_palindromic_refined.py
Genera spirale aurea con overlay palindromico temporale, gradient entropico e overlay equazioni.
Salva PNG e PDF per paper.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

RESULTS_DIR = Path("figures")
RESULTS_DIR.mkdir(exist_ok=True)

PHI = (1 + np.sqrt(5)) / 2
BETA = 1 / PHI**2

def golden_spiral(theta_max=14*np.pi, n_points=12000):
    theta = np.linspace(0, theta_max, n_points)
    r = np.exp(theta / PHI) * 0.05  # scala per visualizzazione
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, theta

def palindromic_wave(t, period=8*np.pi, amp=1.0):
    return amp * np.sin(2*np.pi * t / period) * np.cos(2*np.pi * t / period)

fig, ax = plt.subplots(figsize=(11, 11), dpi=400)
ax.set_aspect('equal')
ax.axis('off')

x, y, theta = golden_spiral()
colors = plt.cm.viridis_r(np.linspace(0.1, 0.95, len(x)))  # gradient da alto a basso entropia

for i in range(len(x)-1):
    ax.plot(x[i:i+2], y[i:i+2], color=colors[i], lw=1.8, alpha=0.92, zorder=2)

# Overlay palindromico temporale (onda simmetrica)
t_pal = np.linspace(-7*np.pi, 7*np.pi, 2000)
palin = palindromic_wave(t_pal, period=8*np.pi, amp=0.9)
ax.plot(t_pal*0.08, palin*0.08, 'k--', lw=2.0, alpha=0.75, zorder=3, label='Modulazione palindromica temporale')

# Centro simmetria
ax.plot(0, 0, 'ko', ms=14, label='Punto centrale time-symmetric')

# Overlay equazioni chiave (semi-trasparenti)
ax.text(-0.4, 0.6, r"$\beta = \phi^{-2} \approx 0.382$", fontsize=11, bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))
ax.text(0.3, -0.7, r"$\theta_{N-k} = \theta_k + 2\pi m_k$", fontsize=11, bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))

ax.legend(loc='upper right', fontsize=10, frameon=True, bbox_to_anchor=(1.0, 1.0))

plt.title("Spirale aurea logaritmica con overlay palindromico temporale\nFusione aureo-palindromica nel paradigma TET–CVTL", fontsize=14, pad=20)
plt.savefig(RESULTS_DIR / "golden_spiral_palindromic_refined.png", dpi=400, bbox_inches='tight')
plt.savefig(RESULTS_DIR / "golden_spiral_palindromic_refined.pdf", bbox_inches='tight')
plt.close()

print("Plot raffinato salvato in figures/")