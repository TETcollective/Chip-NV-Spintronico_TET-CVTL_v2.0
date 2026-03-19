#!/usr/bin/env python3
"""
quTip_renascent_full_sim.py

Simulazione completa QuTiP per RENASCENT-Q:
- Sistema due qubit entangled
- Lindblad standard vs con termine retrocausale custom
- Calcolo concurrence, purity, von Neumann entropy
- Plot multipli + salvataggio dati

Autore: Simon Soliman / TET Collective
Data: 15 Marzo 2026
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("results/renascent")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# PARAMETRI FISICI TET–CVTL
# ──────────────────────────────────────────────────────────────────────────────

gamma_relax = 0.012                 # decoerenza relaxation dominante
beta        = (np.sqrt(5)-1)/2**2   # φ^{-2} ≈ 0.381966
epsilon     = 0.08                  # strength retrocausale
A_w         = 2.2                   # weak value amplificato (placeholder)
t_max       = 150
tlist       = np.linspace(0, t_max, 800)

# Stato iniziale: Bell state entangled
bell = (qt.basis(2,0) + qt.basis(2,1)).unit()
rho0 = bell * bell.dag().tensor(qt.qeye(2))

# Hamiltoniana base + drive sinusoidale (SAW-like)
omega = 2 * np.pi * 0.08
H0 = omega * (qt.sigmax().tensor(qt.qeye(2)) + qt.qeye(2).tensor(qt.sigmax()))
H = qt.QobjEvo([H0, lambda t,args: 0.4 * np.sin(omega * t)])

# Dissipatori standard
sm1 = qt.sigmam().tensor(qt.qeye(2))
sm2 = qt.qeye(2).tensor(qt.sigmam())
c_ops_standard = [np.sqrt(gamma_relax) * sm1, np.sqrt(gamma_relax/2) * sm2]

# ──────────────────────────────────────────────────────────────────────────────
# TERMINE RETROCAUSALE RENASCENT-Q (custom superoperatore)
# ──────────────────────────────────────────────────────────────────────────────

def renascent_superop(t: float, rho: qt.Qobj, args: dict) -> qt.Qobj:
    """
    Superoperatore retrocausale semplificato:
    - Proiettore su stato futuro (post-selezione weak)
    - Commutatore weak-value amplificato
    - Scalato da beta * epsilon
    """
    # Stato futuro post-selezionato (esempio: |0><0| su qubit 1)
    proj_f = qt.basis(2,0) * qt.basis(2,0).dag().tensor(qt.qeye(2))

    # Termine projector Lindblad-like
    L_proj = proj_f * rho * proj_f - 0.5 * (proj_f * rho + rho * proj_f)

    # Termine commutatore weak-value (simula feedback retrocausale)
    sz1 = qt.sigmaz().tensor(qt.qeye(2))
    L_wv = 0.02 * A_w * qt.commutator(sz1, rho)

    return beta * epsilon * (L_proj + L_wv)


# Evoluzione standard
result_std = qt.mesolve(H, rho0, tlist, c_ops=c_ops_standard)

# Evoluzione RENASCENT-Q (con c_ops time-dependent custom)
def renascent_c_op(t, rho, args):
    return renascent_superop(t, rho, args)

c_ops_ren = c_ops_standard + [renascent_c_op]
result_ren = qt.mesolve(H, rho0, tlist, c_ops=c_ops_ren)

# ──────────────────────────────────────────────────────────────────────────────
# METRICHE QUANTISTICHE
# ──────────────────────────────────────────────────────────────────────────────

def concurrence_two_qubit(rho: qt.Qobj) -> float:
    """Concurrence per stato due qubit (Wootters formula semplificata)"""
    rho_pt = rho.ptrace(0).ptrace(0)  # partial transpose
    evals = rho_pt.eigenenergies()
    evals = np.sort(evals)[::-1]
    return max(0, np.sqrt(evals[0]) - np.sqrt(evals[1]) - np.sqrt(evals[2]) - np.sqrt(evals[3]))


def purity(rho: qt.Qobj) -> float:
    return (rho * rho).tr().real


def vn_entropy(rho: qt.Qobj) -> float:
    return qt.entropy_vn(rho)


# Calcolo metriche
data = []
for i, t in enumerate(tlist):
    rho_std = result_std.states[i]
    rho_ren = result_ren.states[i]

    row = {
        'time': t,
        'C_std': concurrence_two_qubit(rho_std),
        'C_ren': concurrence_two_qubit(rho_ren),
        'purity_std': purity(rho_std),
        'purity_ren': purity(rho_ren),
        'Svn_std': vn_entropy(rho_std),
        'Svn_ren': vn_entropy(rho_ren)
    }
    data.append(row)

df = pd.DataFrame(data)
df.to_csv(OUTPUT_DIR / "renascent_metrics.csv", index=False)

# Plot
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

axs[0].plot(df['time'], df['C_std'], label='Standard Lindblad', lw=2)
axs[0].plot(df['time'], df['C_ren'], label='RENASCENT-Q', lw=2.5)
axs[0].set_ylabel('Concurrence')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(df['time'], df['purity_std'], label='Standard')
axs[1].plot(df['time'], df['purity_ren'], label='RENASCENT-Q')
axs[1].set_ylabel('Purity')
axs[1].legend()
axs[1].grid(True)

axs[2].plot(df['time'], df['Svn_std'], label='Standard')
axs[2].plot(df['time'], df['Svn_ren'], label='RENASCENT-Q')
axs[2].set_ylabel('Von Neumann Entropy')
axs[2].set_xlabel('Tempo')
axs[2].legend()
axs[2].grid(True)

plt.suptitle('Recovery entanglement e negentropia via RENASCENT-Q')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "renascent_full_plot.png", dpi=200, bbox_inches='tight')
plt.close()

print(f"Concurrence finale RENASCENT-Q: {df['C_ren'].iloc[-1]:.4f}")
print(f"Recovery %: {100 * (df['C_ren'].iloc[-1] - df['C_std'].iloc[-1]) / df['C_std'].iloc[-1]:.1f}%")
print(f"Dati salvati in: {OUTPUT_DIR}")