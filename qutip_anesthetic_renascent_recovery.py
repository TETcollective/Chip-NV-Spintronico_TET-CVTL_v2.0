# File: code/qutip_anesthetic_renascent_recovery.py
# Simulazione QuTiP: anestesia + recupero RENASCENT-Q (Protocollo 2)
# TET–CVTL v2.0 – NV sensing e Orch OR
# Autore: @PhysSoliman / TETcollective – Marzo 2026

import numpy as np
import qutip as qt
from qutip import concurrence, basis, tensor, sigmaz, sigmam, qeye, mesolve
import matplotlib.pyplot as plt
import json

gamma_relax_base = 0.012
gamma_deph_base  = 0.0072
tlist = np.linspace(0, 20, 401)

# Define constant collapse operators
sm1 = tensor(sigmam(), qeye(2))
sm2 = tensor(qeye(2), sigmam())
sz1 = tensor(sigmaz(), qeye(2))
sz2 = tensor(qeye(2), sigmaz())

# Functions to calculate time-dependent coefficients for collapse operators
def get_relax_coeff(t, args):
    if 5 < t < 10:
        return np.sqrt(0.05)
    return np.sqrt(gamma_relax_base)

def get_deph_coeff(t, args):
    if 5 < t < 10:
        return np.sqrt(0.03)
    return np.sqrt(gamma_deph_base)

# Combine operators and coefficient functions into c_ops list
c_ops_time_dependent = [
    [sm1, get_relax_coeff],
    [sm2, get_relax_coeff],
    [sz1, get_deph_coeff],
    [sz2, get_deph_coeff]
]

H = 0.05 * (tensor(sigmaz(), qeye(2)) + tensor(qeye(2), sigmaz()))

psi0 = (tensor(basis(2,0), basis(2,1)) + tensor(basis(2,1), basis(2,0))).unit()
rho0 = psi0 * psi0.dag()

# Baseline (anestesia senza RENASCENT-Q)
result_anest = mesolve(H, rho0, tlist, c_ops_time_dependent, [])

# Con RENASCENT-Q kicks (definito come prima)
def kick_envelope(t, args):
    t_mod = t % 4.0
    if t_mod < 0.6:
        return np.exp( -((t_mod - 0.3)**2) / (2 * 0.075**2) )
    return 0.0

H_kick = 5.0 * (tensor(qt.sigmax(), qt.sigmax()) + 0.4 * tensor(qt.sigmaz(), qt.sigmaz()))
H_t = [H, [H_kick, kick_envelope]]

result_ren = mesolve(H_t, rho0, tlist, c_ops_time_dependent, [])

conc_anest = np.array([concurrence(rho) for rho in result_anest.states])
conc_ren   = np.array([concurrence(rho) for rho in result_ren.states])

# Plot
plt.figure(figsize=(10, 6))
plt.plot(tlist, conc_anest, 'b-', label='Anestesia (no RENASCENT-Q)')
plt.plot(tlist, conc_ren, 'gold', label='Anestesia + RENASCENT-Q')
plt.axvspan(5, 10, alpha=0.1, color='red', label='Fase anestetica')
plt.xlabel('Tempo (unità)')
plt.ylabel('Concurrence proxy')
plt.title('Soppressione anestetica e recupero RENASCENT-Q (Protocollo 2)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('anesthetic_renascent_recovery.pdf', dpi=300, bbox_inches='tight')

# JSON
metrics = {
    "min_conc_anest": float(np.min(conc_anest[100:200])),
    "final_conc_ren": float(conc_ren[-1]),
    "recovery_percent": float((conc_ren[-1] - np.min(conc_anest[100:200])) / np.min(conc_anest[100:200]) * 100)
}
with open('anesthetic_renascent_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("Plot salvato: anesthetic_renascent_recovery.pdf")