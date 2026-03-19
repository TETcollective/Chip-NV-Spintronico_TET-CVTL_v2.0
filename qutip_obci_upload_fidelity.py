# File: code/qutip_obci_upload_fidelity.py
# Simulazione QuTiP per OBCI upload fidelity (OBCI espansione)
# TET–CVTL v2.0 – Organoid Brain-Computer Interface
# Autore: @PhysSoliman / TETcollective – Marzo 2026

import numpy as np
import qutip as qt
from qutip import concurrence, fidelity, basis, tensor, mesolve
import matplotlib.pyplot as plt
import json

tlist = np.linspace(0, 20, 401)

# Organoide (2 qubit) → Substrato digitale (qubit topologico proxy)
psi_org = (tensor(basis(2,0), basis(2,1)) + tensor(basis(2,1), basis(2,0))).unit()
rho_org = psi_org * psi_org.dag()

# Substrato target (stato target)
psi_target = qt.bell_state('00')
rho_target = psi_target * psi_target.dag()

# Decoerenza durante upload
c_ops_upload = [np.sqrt(0.015) * tensor(qt.sigmam(), qt.qeye(2)),
                np.sqrt(0.015) * tensor(qt.qeye(2), qt.sigmam())]

# Hamiltoniana upload (trasferimento graduale)
H_upload = 0.1 * tensor(qt.sigmax(), qt.sigmax())  # coupling

result_upload = mesolve(H_upload, rho_org, tlist, c_ops_upload, [])

fidel = [fidelity(rho, rho_target) for rho in result_upload.states]
conc_upload = [concurrence(rho) for rho in result_upload.states]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(tlist, fidel, 'b-', label='Fidelity upload')
plt.plot(tlist, conc_upload, 'gold', label='Concurrence preserved')
plt.xlabel('Tempo upload (unità)')
plt.ylabel('Fidelity / Concurrence')
plt.title('OBCI: Upload fidelity e concurrence preservata')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('obci_upload_fidelity.pdf', dpi=300, bbox_inches='tight')

# JSON
metrics = {
    "final_fidelity": float(fidel[-1]),
    "final_concurrence": float(conc_upload[-1]),
    "max_fidelity": float(np.max(fidel))
}
with open('obci_upload_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("Plot salvato: obci_upload_fidelity.pdf")