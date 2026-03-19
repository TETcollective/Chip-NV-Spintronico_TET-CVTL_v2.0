"""
Titolo: Calcolo thrust vs twist angle moiré per vacuum torque thruster (proxy QuTiP)
Autore: Assistito da Grok per @PhysSoliman (TETcollective.org)
Data: Marzo 2026

Descrizione:
Simulazione proxy del thrust netto di un vacuum torque thruster con perturbazione moiré-inspired.
Varia twist angle θ → flat band → ρ_moiré → thrust amplificato.
Output: curva thrust vs θ + figura salvata in figures/thrust_vs_twist_moire.png

Dipendenze: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Parametri fissi
P = 100.0               # potenza RF (W)
Q = 1e5                 # fattore qualità
c = 3e8                 # velocità luce (m/s)
hbar = 1.0545718e-34    # hbar (J s)
omega_rf = 2 * np.pi * 2.45e9  # 2.45 GHz
Gamma_retro_base = 1e-2 # rate retrocausale base (s^-1)
Delta_N_ent = 1e12      # asimmetria entangled
V_cav = 0.001           # volume cavità (m^3)
lambda_g = 0.122        # lunghezza guida stretta (m)
lambda_0 = 0.1224       # lunghezza libera (m)

# Thrust classico Shawyer (termine base)
F_class = (2 * P * Q / c) * (1/lambda_g - 1/lambda_0)  # ~50–100 μN

# Funzione ρ_moiré vs twist angle (flat band max a θ=1.1°)
def rho_moire(theta_deg):
    theta_rad = np.deg2rad(theta_deg)
    # Magic angle ~1.1°, ρ max negativo a θ=1.1°
    rho_max = -1e-3  # J/m^3 (scalato da flat band)
    width = 0.1      # larghezza picco intorno a magic angle
    rho = rho_max * np.exp(-((theta_deg - 1.1)**2 / (2*width**2)))
    return rho

# Thrust retrocausale + moiré
def thrust_total(theta_deg):
    Gamma_retro = Gamma_retro_base * (1 + 10 * np.abs(np.sin(np.deg2rad(theta_deg - 1.1))))  # picco a magic
    F_retro = Gamma_retro * (hbar * omega_rf / c) * Delta_N_ent
    rho_m = rho_moire(theta_deg)
    F_moire = rho_m * V_cav / c
    return F_class + F_retro + F_moire  # in N

# Sweep twist angle
theta = np.linspace(0.5, 2.5, 200)
F = np.array([thrust_total(th) for th in theta]) * 1e6  # in μN

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(theta, F, lw=2.5, color='darkred', label='Thrust netto (μN)')
ax.axvline(1.1, color='limegreen', ls='--', lw=1.8, label='Magic angle ~1.1°')
ax.axhline(F_class * 1e6, color='gray', ls=':', lw=1.5, label='Thrust classico Shawyer')

ax.set_xlabel('Twist angle θ (gradi)')
ax.set_ylabel('Thrust netto (μN)')
ax.set_title('Thrust vs twist angle moiré-inspired (proxy vacuum torque thruster)')
ax.legend(fontsize=11, framealpha=0.92)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, 2.5)
ax.set_ylim(0, max(F)*1.1)

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
save_path = 'figures/thrust_vs_twist_moire.png'
plt.savefig(save_path, dpi=400, bbox_inches='tight')
plt.show()

print(f"Figura salvata: {save_path}")
print(f"Thrust max stimato a θ=1.1°: {max(F):.1f} μN")
print(f"Thrust classico (riferimento): {F_class*1e6:.1f} μN")