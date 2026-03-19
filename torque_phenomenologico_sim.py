#!/usr/bin/env python3
"""
torque_phenomenologico_sim.py

Simulazione toy del torque fenomenologico gerarchico:
- Flusso negentropico layer-by-layer
- Calcolo τ_φ(t) con contributi entropici + weak value + anyon phase
- Curvatura informativa locale
- Plot dinamico + dati CSV

Autore: Simon Soliman / TET Collective
Data: 15 Marzo 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results/torque")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def simulate_phenomenological_torque(t_max: float = 200,
                                     dt: float = 0.1,
                                     kappa: float = 1.2,
                                     lambda_wv: float = 0.8,
                                     mu_cross: float = 0.45,
                                     beta: float = 0.382) -> pd.DataFrame:
    """
    Simulazione toy τ_φ gerarchico.
    """
    t = np.arange(0, t_max, dt)
    N = len(t)

    # Entropia layer (diminuisce con RENASCENT-Q)
    S_micro = 2.5 * np.exp(-0.04 * t) * (1 + 0.15 * np.sin(2*np.pi*0.08*t))
    S_meso  = 1.8 * np.exp(-0.032 * t) * (1 + 0.12 * np.sin(2*np.pi*0.12*t))
    S_macro = 1.2 * np.exp(-0.025 * t)

    # Cross entropy (aumenta poi satura)
    S_cross = 0.9 * (1 - np.exp(-0.06 * t)) * (1 + 0.08 * np.sin(2*np.pi*0.05*t))

    # Weak value oscillante (amplificazione)
    A_w = 2.0 + 1.2 * np.sin(2*np.pi*0.03*t + np.pi/4)

    # Fase anyonica global (braiding)
    theta_braid = 0.5 * t + 0.8 * np.sin(0.4 * t)

    # Torque contributi
    dS_micro = np.gradient(S_micro, dt)
    dS_meso  = np.gradient(S_meso, dt)
    dS_macro = np.gradient(S_macro, dt)
    dS_cross = np.gradient(S_cross, dt)
    dtheta   = np.gradient(theta_braid, dt)

    tau = (kappa * (0.55 * dS_micro + 0.30 * dS_meso + 0.15 * dS_macro) +
           lambda_wv * A_w * dtheta +
           mu_cross * dS_cross)

    # Curvatura informativa
    R_info = tau / 1.0545718  # ħ in eV·fs, scalare arbitrario

    df = pd.DataFrame({
        't': t,
        'tau_phi': tau,
        'R_info': R_info,
        'S_micro': S_micro,
        'S_meso': S_meso,
        'S_macro': S_macro,
        'S_cross': S_cross,
        'A_w': A_w,
        'theta_braid': theta_braid
    })

    df.to_csv(RESULTS_DIR / "torque_phenomenologico_dynamics.csv", index=False)

    return df


if __name__ == "__main__":
    df = simulate_phenomenological_torque()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 12), sharex=True)

    ax1.plot(df['t'], df['tau_phi'], lw=2.2, color='darkorange', label=r'$\tau_\phi$')
    ax1.set_ylabel(r'Torque fenomenologico')
    ax1.legend()
    ax1.grid(True, alpha=0.35)

    ax2.plot(df['t'], df['S_micro'], label='micro (Trp/MT)', lw=1.8)
    ax2.plot(df['t'], df['S_meso'],  label='meso (vago/HRV)', lw=1.8)
    ax2.plot(df['t'], df['S_macro'], label='macro', lw=1.8)
    ax2.plot(df['t'], df['S_cross'], '--', label='cross', lw=1.6)
    ax2.set_ylabel(r'$S_{ent}$ layer')
    ax2.legend()
    ax2.grid(True, alpha=0.35)

    ax3.plot(df['t'], df['R_info'], lw=2.4, color='purple', label=r'$\mathcal{R}_{info}$')
    ax3.set_xlabel('Tempo')
    ax3.set_ylabel(r'Curvatura informativa')
    ax3.legend()
    ax3.grid(True, alpha=0.35)

    plt.suptitle(r'Simulazione toy torque fenomenologico gerarchico TET–CVTL', fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(RESULTS_DIR / "torque_phenomenologico_sim.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Simulazione torque completata.")
    print(f"Dati e plot in: {RESULTS_DIR}")