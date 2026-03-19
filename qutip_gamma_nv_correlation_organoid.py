# File: code/qutip_gamma_nv_correlation_organoid.py
# Simulazione QuTiP per correlazione NV-gamma in organoidi (Protocollo 1)
# TET–CVTL v2.0 – NV sensing di gamma oscillations
# Autore: @PhysSoliman / TETcollective – Marzo 2026

import numpy as np
!pip install qutip
import qutip as qt
from qutip import concurrence, basis, tensor, sigmaz, mesolve
import matplotlib.pyplot as plt
import json

# Parametri
gamma_relax = 0.012
gamma_deph  = 0.0072
tlist = np.linspace(0, 20, 401)

# Due qubit proxy microtubuli
sz1 = tensor(sigmaz(), qt.qeye(2))
sz2 = tensor(qt.qeye(2), sigmaz())

# Hamiltoniana base (interazione debole per vibrazione)
H = 0.05 * (sz1 + sz2)  # splitting Zeeman-like

# Collapse operators
c_ops = [
    np.sqrt(gamma_relax) * tensor(qt.sigmam(), qt.qeye(2)),
    np.sqrt(gamma_relax) * tensor(qt.qeye(2), qt.sigmam()),
    np.sqrt(gamma_deph)  * sz1,
    np.sqrt(gamma_deph)  * sz2
]

# Stato iniziale entangled
psi0 = (tensor(basis(2,0), basis(2,1)) + tensor(basis(2,1), basis(2,0))).unit()
rho0 = psi0 * psi0.dag()

# Simulazione
result = mesolve(H, rho0, tlist, c_ops, [])

# Concurrence e proxy gamma power (oscillazione sinusoidale + rumore)
conc = np.array([concurrence(rho) for rho in result.states])
gamma_power = 0.5 * (1 + 0.8 * np.sin(2 * np.pi * tlist / 0.033)) + 0.1 * np.random.randn(len(tlist))  # gamma ~30 Hz (periodo 33 ms)

# Correlazione rolling (latenza max 5 ms = 5 punti a dt=0.05)
corr = np.correlate(conc, gamma_power, mode='same')
corr = corr / (np.std(conc) * np.std(gamma_power) * len(tlist))

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax1.plot(tlist, conc, 'b-', label='Concurrence proxy MT')
ax1.plot(tlist, gamma_power, 'r--', label='Gamma power proxy')
ax1.set_ylabel('Valori normalizzati')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(tlist, corr, 'g-', label='Correlazione rolling')
ax2.axvline(5e-3, color='k', ls='--', label='Latenza max 5 ms')
ax2.set_xlabel('Tempo (unità adimensionali)')
ax2.set_ylabel('Correlazione')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('Correlazione NV-gamma in organoidi (Protocollo 1)')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('nv_gamma_correlation_organoid.pdf', dpi=300, bbox_inches='tight')

# JSON
metrics = {
    "max_correlation": float(np.max(corr)),
    "latency_peak_ms": float(tlist[np.argmax(corr)] * 1000),
    "final_concurrence": float(conc[-1])
}
with open('nv_gamma_correlation_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("Plot salvato: nv_gamma_correlation_organoid.pdf")
print("Metriche salvate: nv_gamma_correlation_metrics.json")