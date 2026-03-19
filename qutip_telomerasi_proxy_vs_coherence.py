# File: code/qutip_telomerasi_proxy_vs_coherence.py
# Simulazione QuTiP: proxy telomerasi quantistica vs coerenza entanglement
# TET–CVTL v2.0 – Chip NV-Spintronico embodied & longevità radicale
# Autore: @PhysSoliman / TETcollective – Marzo 2026

import numpy as np
import qutip as qt
from qutip import concurrence, basis, tensor, sigmax, sigmay, sigmaz, qeye, mesolve
import matplotlib.pyplot as plt
import json
from datetime import datetime

# ==============================================
# Parametri fisici e simulazione
# ==============================================
gamma_relax = 0.012     # relaxation rate (MHz, proxy decoerenza termica)
gamma_deph  = 0.0072    # pure dephasing rate (MHz)
theta = np.pi / 3       # fase twist embodied (da Hamiltoniana base)

k_strength = 5.0        # strength RENASCENT-Q kicks
duration   = 0.6        # durata Gaussian kick
interval   = 4.0        # intervallo tra kicks

t_total = 25.0          # tempo totale simulato (unità adimensionali)
n_points = 601          # punti tempo
tlist = np.linspace(0, t_total, n_points)

# Proxy telomerasi: recupero graduale della "capacità massima" di coherence
# Simula estensione telomeri: riduce rate decoerenza cumulativa nel tempo
telomerasi_rate = 0.0008  # per unità tempo, riduzione lenta decoerenza
telomerasi_max = 0.35     # massimo recupero relativo possibile (35%)

# Stato iniziale entangled twisted
alpha = 1.0 / np.sqrt(2)
beta  = 1.0 / np.sqrt(2) * np.exp(1j * theta)
psi0  = alpha * tensor(basis(2,0), basis(2,0)) + beta * tensor(basis(2,1), basis(2,1))
psi0  = psi0.unit()
rho0  = psi0 * psi0.dag()

# Operatori due qubit
id = qeye(2)
sx = sigmax(); sy = sigmay(); sz = sigmaz()

sx1 = tensor(sx, id); sx2 = tensor(id, sx)
sy1 = tensor(sy, id); sy2 = tensor(id, sy)
sz1 = tensor(sz, id); sz2 = tensor(id, sz)

# Hamiltoniana base
H_theta = theta * tensor(basis(2,1)*basis(2,1).dag(), id)
H_swap  = 0.8 * (sx1*sx2 + sy1*sy2)
H_base  = H_theta + H_swap

# Collapse operators base (decoerenza dinamica con telomerasi)
# Define functions that return the time-dependent *coefficients* for each operator.
def get_eff_relax_coefficient(t, args):
    # Calculate the time-dependent effective relaxation rate
    reduction = telomerasi_rate * t
    eff_rate = max(gamma_relax - reduction, gamma_relax * (1 - telomerasi_max))
    return np.sqrt(eff_rate)

def get_eff_deph_coefficient(t, args):
    # Calculate the time-dependent effective dephasing rate
    reduction = telomerasi_rate * t
    eff_rate = max(gamma_deph - reduction, gamma_deph * (1 - telomerasi_max))
    return np.sqrt(eff_rate)

# Now, define c_ops as a list of [operator, coefficient_function] pairs
c_ops_time_dependent = [
    [tensor(qt.sigmam(), id), get_eff_relax_coefficient],
    [tensor(id, qt.sigmam()), get_eff_relax_coefficient],
    [sz1, get_eff_deph_coefficient],
    [sz2, get_eff_deph_coefficient]
]

# ==============================================
# RENASCENT-Q kicks smooth (Gaussian envelope)
# ==============================================
def kick_envelope(t, args):
    t_mod = t % interval
    if t_mod < duration:
        mu    = duration / 2
        sigma = duration / 8
        return np.exp( -((t_mod - mu)**2) / (2 * sigma**2) )
    return 0.0

H_kick_op = k_strength * (sx1*sx2 + 0.4 * sz1*sz2)  # coupling XX + ZZ (da paper)

H_t = [H_base, [H_kick_op, kick_envelope]]

# ==============================================
# Evoluzione (con telomerasi dinamica)
# ==============================================
print("Simulazione baseline (no kicks) con telomerasi proxy...")
result_base = mesolve(H_base, rho0, tlist, c_ops_time_dependent, [], args={})

print("Simulazione RENASCENT-Q kicks con telomerasi proxy...")
result_kick = mesolve(H_t, rho0, tlist, c_ops_time_dependent, [], args={})

# Calcolo concurrence
conc_base = np.array([concurrence(state) for state in result_base.states])
conc_kick = np.array([concurrence(state) for state in result_kick.states])

# Proxy telomerasi: riduzione cumulativa decoerenza (media su tempo)
tel_base = gamma_relax - telomerasi_rate * tlist
tel_base = np.maximum(tel_base, gamma_relax * (1 - telomerasi_max))
tel_kick = tel_base.copy()  # stesso per entrambi (telomerasi è effetto cumulativo)

# ==============================================
# Salva metriche in JSON
# ==============================================
metrics = {
    "timestamp": datetime.now().isoformat(),
    "parameters": {
        "gamma_relax": gamma_relax,
        "gamma_deph": gamma_deph,
        "k_strength": k_strength,
        "duration": duration,
        "interval": interval,
        "telomerasi_rate": telomerasi_rate,
        "telomerasi_max": telomerasi_max
    },
    "final_conc_base": float(conc_base[-1]),
    "final_conc_kick": float(conc_kick[-1]),
    "recovery_percent": float((conc_kick[-1] - conc_base[-1]) / conc_base[-1] * 100),
    "telomerasi_final": float(tel_base[-1]),
    "tlist": tlist.tolist(),
    "conc_base": conc_base.tolist(),
    "conc_kick": conc_kick.tolist(),
    "telomerasi_proxy": tel_base.tolist()
}

with open('telomerasi_proxy_vs_coherence_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print(f"Metriche salvate in: telomerasi_proxy_vs_coherence_metrics.json")
print(f"Recupero finale concurrence: {metrics['recovery_percent']:.1f}%")

# ==============================================
# Plot per paper / Overleaf
# ==============================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Panel 1: Concurrence vs tempo
ax1.plot(tlist, conc_base, 'b-', lw=2.2, label=f'Baseline (no kicks) → {conc_base[-1]:.4f}')
ax1.plot(tlist, conc_kick, color='gold', lw=2.5, label=f'RENASCENT-Q + telomerasi → {conc_kick[-1]:.4f}')
ax1.set_ylabel('Concurrence C')
ax1.set_ylim(0.0, 0.85)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=10)
ax1.set_title('Proxy telomerasi quantistica vs coerenza entanglement\n(TET–CVTL v2.0 – Marzo 2026)')

# Panel 2: Proxy telomerasi (riduzione decoerenza cumulativa)
ax2.plot(tlist, tel_base, 'r--', lw=2, label='Proxy telomerasi (riduzione decoerenza)')
ax2.axhline(gamma_relax, color='gray', ls=':', label='Decoerenza iniziale')
ax2.axhline(gamma_relax * (1 - telomerasi_max), color='darkred', ls='--', label='Limite massimo telomerasi')
ax2.set_xlabel('Tempo (unità adimensionali)')
ax2.set_ylabel('Rate decoerenza effettivo (MHz)')
ax2.set_ylim(0, gamma_relax * 1.1)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('telomerasi_proxy_vs_coherence.pdf', dpi=300, bbox_inches='tight')
print("Plot salvato come: telomerasi_proxy_vs_coherence.pdf")



# plt.show()  # decommenta se vuoi visualizzare live