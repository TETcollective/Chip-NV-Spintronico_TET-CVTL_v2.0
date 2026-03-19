# File: code/qutip_quant_fhn_full.py
# TET–CVTL v2.0 – FitzHugh-Nagumo quantistico completo per nodo seno-atriale
# Autore: @PhysSoliman / TETcollective – Marzo 2026
# Descrizione: Evoluzione master equation Lindblad con potenziale cubico quantizzato,
#              operatori v e w quantizzati (truncated Fock-like basis), squeezing opzionale,
#              calcolo correlatore C(τ) e violazione Cauchy-Schwarz come witness entanglement.
# Requisiti: QuTiP, numpy, matplotlib, json
# Output: plot v(t), correlatore C(τ), violazione CS, file JSON risultati

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

# ==============================================
# PARAMETRI REALISTICI (calibrati per SAN ~1 Hz / 60 bpm)
# ==============================================
N = 40                  # cutoff dimensione spazio Hilbert (per v e w quantizzati)
eps = 0.009             # separazione scale tempi (ε << 1)
a = 0.20                # eccitabilità
b = 1.0                 # pendenza recupero
tau = 12.5              # scala tempo recupero
I_ext = 0.0             # corrente esterna (endogena)

gamma_v = 1.0           # decoerenza su v (Γ_v ~1 s⁻¹)
gamma_w = 0.1           # decoerenza su w (lenta)
gamma_phi = 20.0        # dephasing rapido (τ_c ≈ 50 ms)

tlist = np.linspace(0, 100, 2001)  # tempo simulazione (s)
tau_max = 20.0          # max ritardo correlatore
n_tau = 401             # punti τ

# ==============================================
# OPERATORI QUANTIZZATI (truncated basis per v e w)
# ==============================================
v_op = qt.destroy(N) + qt.create(N)  # quadratura posizione-like per v
w_op = qt.destroy(N) + qt.create(N)  # per w (stessa base per semplicità)
p_v = 1j * (qt.create(N) - qt.destroy(N))  # momento coniugato v

v2 = v_op * v_op
v4 = v2 * v2

# ==============================================
# Hamiltoniano quantistico FHN (potenziale cubico doppia buca)
# ==============================================
V_v = v4 / 12 - v2 / 2               # potenziale V(v) = v⁴/12 - v²/2
H_v = V_v + (tau / 2) * (w_op * w_op) - a * v_op * w_op
H_fhn = H_v  # con ħ=1

# ==============================================
# Dissipatori Lindblad
# ==============================================
c_ops = [
    np.sqrt(gamma_v) * v_op,                # relaxation v
    np.sqrt(gamma_w) * w_op,                # relaxation w
    np.sqrt(gamma_phi) * (v_op - qt.expect(v_op, qt.basis(N, 0)))  # dephasing v
]

# ==============================================
# Stato iniziale: coherent-like per oscillazione pacemaker
# ==============================================
alpha_v = 2.5 + 0.5j    # ampiezza iniziale v (per ciclo limite)
alpha_w = 1.0 + 0.2j    # w
psi0 = qt.tensor(qt.coherent(N, alpha_v), qt.coherent(N, alpha_w))
rho0 = psi0 * psi0.dag()

# ==============================================
# Evoluzione master equation
# ==============================================
print("Evoluzione master equation FHN quantistico...")
result = qt.mesolve(H_fhn, rho0, tlist, c_ops=c_ops)

# ==============================================
# Calcolo potenziale medio v(t) e correlatore C(τ)
# ==============================================
v_expect = qt.expect(v_op, result.states)

def compute_correlator(states, tau_indices):
    C = []
    v_exp = qt.expect(v_op, states)
    v2_exp = qt.expect(v_op * v_op, states)
    for i, tau in enumerate(tau_indices):
        corr = []
        for j in range(len(states) - tau):
            exp_prod = qt.expect(v_op * v_op, states[j].dag() * states[j+tau])
            corr.append(exp_prod - v_exp[j] * v_exp[j+tau])
        C.append(np.mean(corr))
    return np.array(C)

tau_list = np.linspace(0, tau_max, n_tau, dtype=int)
C_tau = compute_correlator(result.states, tau_list)

# Bound CS
v2_t = qt.expect(v_op * v_op, result.states)
cs_bound_diag = np.sqrt(v2_t[:-len(tau_list)] * v2_t[len(tau_list):])
violation = np.abs(C_tau)**2 - cs_bound_diag
violation_percent = (violation / cs_bound_diag) * 100

# ==============================================
# Plot principali
# ==============================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

ax1.plot(tlist, v_expect, 'b-', lw=2, label='⟨v(t)⟩')
ax1.set_ylabel('Potenziale medio ⟨v⟩')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(tau_list, C_tau, 'g-', lw=2, label='C(τ)')
ax2.set_ylabel('Correlatore C(τ)')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3.plot(tau_list, violation_percent, 'r-', lw=2, label='Violazione CS (%)')
ax3.axhline(0, color='k', linestyle='--', lw=1)
ax3.set_xlabel('τ (unità tempo)')
ax3.set_ylabel('Violazione CS (%)')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.suptitle('FHN Quantistico – Oscillazioni, Correlatore e Violazione CS')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('qutip_quant_fhn_full_results.pdf', dpi=300, bbox_inches='tight')
print("Plot salvato: qutip_quant_fhn_full_results.pdf")

# ==============================================
# Salvataggio JSON risultati
# ==============================================
data = {
    "simulation_info": {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "framework": "TET–CVTL v2.0 – FHN Quantistico SAN",
        "parameters": {
            "eps": eps, "a": a, "b": b, "tau": tau,
            "gamma_v": gamma_v, "gamma_w": gamma_w, "gamma_phi": gamma_phi
        }
    },
    "time": tlist.tolist(),
    "v_expect": v_expect.tolist(),
    "tau": tau_list.tolist(),
    "C_tau": C_tau.tolist(),
    "violation_percent": violation_percent.tolist(),
    "max_violation": float(np.max(violation_percent)),
    "mean_violation": float(np.mean(violation_percent[violation_percent > 0]))
}

with open("fhn_quant_full_results.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

print("Risultati salvati in fhn_quant_full_results.json")
print(f"Massima violazione CS: {data['max_violation']:.2f}%")