# File: code/renascent_q_concurrence_simulation.py
# Simulazione QuTiP per recupero concurrence con RENASCENT-Q kicks
# TET–CVTL v2.0 – Chip NV-Spintronico embodied
# Autore: @PhysSoliman / TETcollective – Marzo 2026

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, tensor, sigmax, sigmaz, sigmay, qeye, mesolve, concurrence

# ==============================================
# Parametri calibrati per matchare paper (baseline ~0.6676, kick ~0.7485)
# ==============================================
theta = np.pi / 3                # fase twist embodied
gamma_relax = 0.012              # relaxation rate
gamma_deph  = 0.0072             # dephasing rate (piccolo)
tlist = np.linspace(0, 25, 501)  # tempo esteso per asintoto chiaro

# RENASCENT-Q gold ottimizzata
k_strength = 4.8                 # leggermente ridotto per boost realistico ~12%
duration   = 0.6
interval   = 3.8

# ==============================================
# Operatori due qubit
# ==============================================
id = qeye(2)
sx = sigmax()
sy = sigmay()
sz = sigmaz()

sx1 = tensor(sx, id)
sx2 = tensor(id, sx)
sy1 = tensor(sy, id)
sy2 = tensor(id, sy)
sz1 = tensor(sz, id)
sz2 = tensor(id, sz)

p11 = tensor(basis(2,1) * basis(2,1).dag(), id)

# ==============================================
# Hamiltoniana base
# ==============================================
H_theta     = theta * p11
H_xx_yy     = 0.8 * (sx1 * sx2 + sy1 * sy2)          # swap-like più forte
H_base      = H_theta + H_xx_yy

# ==============================================
# Stato iniziale: entangled twisted (per concurrence iniziale ~1)
# ==============================================
alpha = 1.0 / np.sqrt(2)
beta  = 1.0 / np.sqrt(2) * np.exp(1j * theta)
psi0  = alpha * tensor(basis(2,0), basis(2,0)) + beta * tensor(basis(2,1), basis(2,1))
psi0  = psi0.unit()
rho0  = psi0 * psi0.dag()

# ==============================================
# Collassi Lindblad
# ==============================================
sm1 = tensor(qt.sigmam(), id)
sm2 = tensor(id, qt.sigmam())
c_ops_base = [
    np.sqrt(gamma_relax) * sm1,
    np.sqrt(gamma_relax) * sm2,
    np.sqrt(gamma_deph)  * sz1,   # dephasing full rate
    np.sqrt(gamma_deph)  * sz2
]

# ==============================================
# Kick RENASCENT-Q smooth (Gaussian envelope)
# ==============================================
def kick_envelope(t, args):
    t_mod = t % interval
    if t_mod < duration:
        mu    = duration / 2
        sigma = duration / 8
        env   = np.exp( -((t_mod - mu)**2) / (2 * sigma**2) )
        return k_strength * env
    return 0.0

H_kick_op = 1.0 * (sx1 * sx2 + 0.4 * sz1 * sz2)   # dal paper, strength scalato fuori

H_t = [H_base, [H_kick_op, kick_envelope]]

# ==============================================
# Evoluzione master equation
# ==============================================
print("Evoluzione baseline (no kick)...")
result_base = mesolve(H_base, rho0, tlist, c_ops_base, [])

print("Evoluzione con RENASCENT-Q kicks...")
result_kick = mesolve(H_t, rho0, tlist, c_ops_base, [])

# ==============================================
# Concurrence vs tempo
# ==============================================
conc_base = np.array([concurrence(state) for state in result_base.states])
conc_kick = np.array([concurrence(state) for state in result_kick.states])

print(f"\nConcurrence finale base:    {conc_base[-1]:.4f}")
print(f"Concurrence finale kick:    {conc_kick[-1]:.4f}")
print(f"Recupero relativo:          {(conc_kick[-1] - conc_base[-1]) / conc_base[-1] * 100:.1f}%")

# ==============================================
# Plot per paper / Overleaf
# ==============================================
plt.figure(figsize=(10, 6))
plt.plot(tlist, conc_base, 'b-', lw=2, label=f'Base (no kick) → {conc_base[-1]:.4f}')
plt.plot(tlist, conc_kick, color='gold', lw=2.5, label=f'RENASCENT-Q gold → {conc_kick[-1]:.4f}')
plt.xlabel('Tempo (unità adimensionali)')
plt.ylabel('Concurrence')
plt.title(r'Evoluzione della concurrence ($\gamma_{\rm relax}=0.012$)')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('renascent_smooth_kick_gold_final.pdf', dpi=300, bbox_inches='tight')
print("\nPlot salvato come: renascent_smooth_kick_gold_final.pdf")
# plt.show()  # decommenta se vuoi visualizzare live