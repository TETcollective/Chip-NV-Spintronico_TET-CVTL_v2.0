# File: code/qutip_moire_chirality_asymmetry.py
# Simulazione QuTiP per asimmetria chirale moiré (Protocollo 3)
# TET–CVTL v2.0 – NV sensing moiré torque
# Autore: @PhysSoliman / TETcollective – Marzo 2026

import numpy as np
import qutip as qt
from qutip import concurrence, basis, tensor, sigmax, sigmay, mesolve
import matplotlib.pyplot as plt
import json

tlist = np.linspace(0, 15, 301)

H_base = 0.05 * (tensor(qt.sigmax(), qt.sigmax()) + tensor(qt.sigmay(), qt.sigmay()))

# Due polarità moiré (CISS-like)
H_moire_plus  = 0.03 * tensor(qt.sigmax(), qt.sigmay())
H_moire_minus = -0.03 * tensor(qt.sigmax(), qt.sigmay())

c_ops = [np.sqrt(0.012) * tensor(qt.sigmam(), qt.qeye(2)),
         np.sqrt(0.012) * tensor(qt.qeye(2), qt.sigmam())]

psi0 = (tensor(basis(2,0), basis(2,1)) + tensor(basis(2,1), basis(2,0))).unit()
rho0 = psi0 * psi0.dag()

# Simulazioni
result_plus  = mesolve(H_base + H_moire_plus, rho0, tlist, c_ops, [])
result_minus = mesolve(H_base + H_moire_minus, rho0, tlist, c_ops, [])

conc_plus  = np.array([concurrence(rho) for rho in result_plus.states])
conc_minus = np.array([concurrence(rho) for rho in result_minus.states])

asymmetry = conc_plus - conc_minus

# Plot
plt.figure(figsize=(10, 6))
plt.plot(tlist, conc_plus, 'b-', label='Polarità + (CISS-like)')
plt.plot(tlist, conc_minus, 'r--', label='Polarità -')
plt.plot(tlist, asymmetry, 'g-', lw=2.5, label='Asimmetria chirale')
plt.xlabel('Tempo (unità)')
plt.ylabel('Concurrence / Asimmetria')
plt.title('Asimmetria chirale da perturbazione moiré (Protocollo 3)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('moire_chirality_asymmetry.pdf', dpi=300, bbox_inches='tight')

# JSON
metrics = {
    "max_asymmetry": float(np.max(asymmetry)),
    "mean_asymmetry": float(np.mean(asymmetry)),
    "final_conc_plus": float(conc_plus[-1]),
    "final_conc_minus": float(conc_minus[-1])
}
with open('moire_chirality_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("Plot salvato: moire_chirality_asymmetry.pdf")