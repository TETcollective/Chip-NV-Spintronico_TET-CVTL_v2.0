#!/usr/bin/env python3
"""
plot_golden_beta_spiral_palindromic.py

Spirale aurea φ (crescente) + spirale inversa β (decrescente) con overlay palindromico temporale.
Gradient colore per riduzione entropica (ΔS < 0 verso il centro).
Versione raffinata per paper TET–CVTL: fusione aureo-palindromica.

Autore: Simon Soliman / TET Collective
Data: 16 Marzo 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("figures")
RESULTS_DIR.mkdir(exist_ok=True)

PHI = (1 + np.sqrt(5)) / 2          # ≈1.618034
BETA = 1 / PHI**2                   # ≈0.381966

def logarithmic_spiral(theta, growth_rate):
    """ Spirale logaritmica generica """
    r = np.exp(growth_rate * theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def palindromic_wave(t, period=8*np.pi, amp=0.9):
    """ Onda palindromica simmetrica """
    return amp * np.sin(2*np.pi * t / period) * np.cos(2*np.pi * t / period)

fig, ax = plt.subplots(figsize=(12, 12), dpi=400)
ax.set_aspect('equal')
ax.axis('off')

# ──────────────────────────────────────────────────────────────────────────────
# SPIRALE AUREA φ (crescente, verso l'esterno)
# ──────────────────────────────────────────────────────────────────────────────
theta_phi = np.linspace(0, 14*np.pi, 10000)
x_phi, y_phi = logarithmic_spiral(theta_phi, 1/PHI) * 0.05   # scala visiva
colors_phi = plt.cm.viridis(np.linspace(0.1, 0.9, len(theta_phi)))  # rosso → verde

for i in range(len(x_phi)-1):
    ax.plot(x_phi[i:i+2], y_phi[i:i+2], color=colors_phi[i], lw=1.6, alpha=0.92, zorder=2)

# ──────────────────────────────────────────────────────────────────────────────
# SPIRALE INVERSA β (decrescente, verso il centro)
# ──────────────────────────────────────────────────────────────────────────────
theta_beta = np.linspace(14*np.pi, 0, 8000)   # parte da fuori e converge al centro
x_beta, y_beta = logarithmic_spiral(theta_beta, -1/BETA) * 0.05
colors_beta = plt.cm.plasma_r(np.linspace(0.1, 0.95, len(theta_beta)))  # viola → blu

for i in range(len(x_beta)-1):
    ax.plot(x_beta[i:i+2], y_beta[i:i+2], color=colors_beta[i], lw=1.8, alpha=0.88, zorder=3)

# Overlay palindromico temporale (onda simmetrica centrale)
t_pal = np.linspace(-8*np.pi, 8*np.pi, 2000)
palin = palindromic_wave(t_pal)
ax.plot(t_pal*0.06, palin*0.06, 'k--', lw=2.1, alpha=0.75, label='Modulazione palindromica temporale')

# Centro simmetria time-symmetric
ax.plot(0, 0, 'ko', ms=16, mec='white', mew=2, label='Punto centrale time-symmetric')

# Overlay equazioni chiave
ax.text(-0.6, 0.7, r"$\phi \approx 1.618$" + "\n" + r"espansione entropica", fontsize=11,
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
ax.text(0.4, -0.8, r"$\beta = \phi^{-2} \approx 0.382$" + "\n" + r"contrazione negentropica", fontsize=11,
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
ax.text(0.0, 0.05, r"$\theta_{N-k} = \theta_k + 2\pi m_k$", fontsize=10, ha='center',
        bbox=dict(facecolor='ivory', alpha=0.8, edgecolor='gray'))

# Titolo e legenda
ax.set_title("Fusione aureo-palindromica: spirale φ (espansione) + spirale β (contrazione)\ncon overlay palindromico temporale (TET–CVTL)", fontsize=14, pad=25)
ax.legend(loc='upper right', fontsize=10, frameon=True, bbox_to_anchor=(1.0, 1.0))

plt.savefig(RESULTS_DIR / "golden_beta_spiral_palindromic.png", dpi=400, bbox_inches='tight')
plt.savefig(RESULTS_DIR / "golden_beta_spiral_palindromic.pdf", bbox_inches='tight')
plt.close()

print("Plot aureo-β con overlay palindromico salvato in figures/")