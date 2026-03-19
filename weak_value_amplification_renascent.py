# File: code/weak_value_amplification_renascent.py
# Simulazione QuTiP per amplificazione weak value con RENASCENT-Q
# TET–CVTL v2.0 – Chip NV-Spintronico embodied retrocausale
# Autore: @PhysSoliman / TETcollective – Marzo 2026

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, tensor, sigmaz, sigmax, sigmay, qeye, mesolve, expect

# ==============================================
# Parametri fisici (calibrati per paper)
# ==============================================
theta = np.pi / 3                # fase twist embodied
gamma_relax = 0.012              # relaxation dominante
gamma_deph  = 0.0072             # dephasing
tlist = np.linspace(0, 12, 301)  # tempo di evoluzione prima della weak measurement

# RENASCENT-Q gold (per ridurre decoerenza e preservare weak value)
k_strength = 4.8
duration   = 0.6
interval   = 3.8

# Range di overlap post-selezione δ (log scale)
delta_list = np.logspace(-3, -0.1, 25)  # da 0.001 a ~0.8

# ==============================================
# Operatori due qubit
# ==============================================
id  = qeye(2)
sz1 = tensor(sigmaz(), id)
sz2 = tensor(id, sigmaz())
sx1 = tensor(sigmax(), id)
sx2 = tensor(id, sigmax())
sy1 = tensor(sigmay(), id)
sy2 = tensor(id, sigmay())

p11 = tensor(basis(2,1) * basis(2,1).dag(), id)

# Osservabile weak: σ_z sul qubit 1
A_op = sz1

# ==============================================
# Hamiltoniana base
# ==============================================
H_theta   = theta * p11
H_xx_yy   = 0.8 * (sx1 * sx2 + sy1 * sy2)
H_base    = H_theta + H_xx_yy

# ==============================================
# Stato iniziale pre-selezionato (entangled twisted)
# ==============================================
alpha = 1.0 / np.sqrt(2)
beta  = 1.0 / np.sqrt(2) * np.exp(1j * theta)
psi_pre = alpha * tensor(basis(2,0), basis(2,0)) + beta * tensor(basis(2,1), basis(2,1))
psi_pre = psi_pre.unit()
rho_pre = psi_pre * psi_pre.dag()

# ==============================================
# Dissipatori Lindblad
# ==============================================
sm1 = tensor(qt.sigmam(), id)
sm2 = tensor(id, qt.sigmam())
c_ops_base = [
    np.sqrt(gamma_relax) * sm1,
    np.sqrt(gamma_relax) * sm2,
    np.sqrt(gamma_deph)  * sz1,
    np.sqrt(gamma_deph)  * sz2
]

# ==============================================
# Kick RENASCENT-Q smooth Gaussian
# ==============================================
def kick_envelope(t, args):
    t_mod = t % interval
    if t_mod < duration:
        mu    = duration / 2
        sigma = duration / 8
        env   = np.exp( -((t_mod - mu)**2) / (2 * sigma**2) )
        return k_strength * env
    return 0.0

H_kick_op = sx1 * sx2 + 0.4 * sz1 * sz2
H_t = [H_base, [H_kick_op, kick_envelope]]

# ==============================================
# Funzione per calcolare weak value dopo evoluzione
# ==============================================
def compute_weak_value(rho_evolved, post_state):
    post_ket = post_state
    post_bra = post_ket.dag()
    denom = post_bra * rho_evolved * post_ket
    if abs(denom) < 1e-12:
        return np.nan  # post-selezione quasi nulla
    num = post_bra * A_op * rho_evolved * post_ket
    A_w = num / denom
    return A_w[0,0]  # estrai scalare complesso

# ==============================================
# Sweep su δ (overlap piccolo → post-selezione su stato quasi ortogonale)
# ==============================================
Aw_base  = []
Aw_kick  = []

print("Calcolo baseline weak values...")
for delta in delta_list:
    # Post-stato: |+θ⟩ sul qubit 2, con piccolo mixing per δ
    phi_post = (basis(2,0) + np.sqrt(delta) * basis(2,1)).unit()
    post_op  = tensor(id, phi_post * phi_post.dag())
    
    # Evoluzione base
    result_base = mesolve(H_base, rho_pre, tlist, c_ops_base, [])
    rho_final_base = result_base.states[-1]
    Aw_base.append(compute_weak_value(rho_final_base, post_op * rho_pre * post_op))  # approx weak su evolved

print("Calcolo con RENASCENT-Q...")
for delta in delta_list:
    phi_post = (basis(2,0) + np.sqrt(delta) * basis(2,1)).unit()
    post_op  = tensor(id, phi_post * phi_post.dag())
    
    result_kick = mesolve(H_t, rho_pre, tlist, c_ops_base, [])
    rho_final_kick = result_kick.states[-1]
    Aw_kick.append(compute_weak_value(rho_final_kick, post_op * rho_pre * post_op))

# Converti in array e prendi modulo
Aw_base_mod  = np.abs(np.array(Aw_base))
Aw_kick_mod  = np.abs(np.array(Aw_kick))

# Filtra NaN
valid = ~np.isnan(Aw_base_mod) & ~np.isnan(Aw_kick_mod)
delta_valid = delta_list[valid]
Aw_base_valid = Aw_base_mod[valid]
Aw_kick_valid = Aw_kick_mod[valid]

# ==============================================
# Plot |A_w| vs δ (log-log, simile al paper)
# ==============================================
plt.figure(figsize=(10, 6))
plt.loglog(delta_valid, Aw_base_valid, 'b-o', lw=2, label=f'Base (no kick)')
plt.loglog(delta_valid, Aw_kick_valid, color='gold', marker='s', lw=2.5, label=f'RENASCENT-Q gold kick')
plt.xlabel(r'Overlap post-selezione $\delta$')
plt.ylabel(r'$|A_w|$ (modulo weak value)')
plt.title(r'Amplificazione weak value vs overlap ($\gamma_{\rm relax}=0.012$)')
plt.grid(True, which='both', alpha=0.3)
plt.legend(loc='upper right')
plt.tight_layout()

plt.savefig('weak_value_vs_overlap_renascent.pdf', dpi=300, bbox_inches='tight')
print("\nPlot salvato come: weak_value_vs_overlap_renascent.pdf")

# Esempio gain medio per piccoli δ
if len(Aw_kick_valid) > 0 and len(Aw_base_valid) > 0:
    gain = np.nanmean(Aw_kick_valid / Aw_base_valid)
    print(f"Gain medio RENASCENT-Q / base: ~{gain:.1f}x")