# =============================================================================
# TETcollective - QuTiP Simulation: Quantum HRV from Damped-Driven Oscillator
# File: code/qutip_hrv_quantistica.py
# Descrizione: Simulazione oscillatore bosonico damped-driven con squeezing
#              per modellare fluttuazioni quantistiche in HRV (SAN pacemaker)
# Autore: Grok-assisted per Simon Soliman @physsoliman (TETcollective.org)
# Data: Marzo 2026
# =============================================================================

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from qutip import basis, destroy, qeye, tensor, steadystate, correlation_2op_1t

# Parametri realistici
N = 30                  # cutoff Fock space
omega0 = 2 * np.pi * 1.1  # freq pacemaker ~1.1 Hz
chi = 0.05 * omega0     # weak Kerr nonlinearity
gamma = 0.5             # energy loss rate (s^-1)
kappa = 0.3             # thermal pumping
n_th = 5000             # thermal occupation ~310K
Gamma_phi = 10.0        # dephasing rate (s^-1) -> tau_c ~100 ms
g0 = 0.2 * omega0       # driving amplitude respiratorio
f_resp = 0.1            # 0.1 Hz coherence breathing
T_sim = 120.0           # simulation time (s)
dt = 0.01               # time step

# Operatori
a = destroy(N)
ad = a.dag()
num = a.dag() * a
x = (a + ad) / np.sqrt(2)   # quadratura per delta RR

# Hamiltoniano coerente + driving respiratorio
H0 = omega0 * num + chi * num**2
def H_drive(t, args):
    return g0 * np.sin(2 * np.pi * f_resp * t) * (a + ad)
H = [H0, [a + ad, H_drive]]

# Dissipatori Lindblad
c_ops = [
    np.sqrt(gamma) * a,                         # perdita
    np.sqrt(kappa * (n_th + 1)) * a.dag(),      # gain termico
    np.sqrt(kappa * n_th) * a,                  # bath absorption
    np.sqrt(Gamma_phi) * num                    # dephasing
]

# Opzionale: squeezing term (per entanglement witness)
# Es. Hamiltonian squeezing semplice (parametric drive)
# H_squeeze = 0.1 * omega0 * (a**2 + ad**2)  # uncomment per squeezing ~few dB

# Stato iniziale: coherent displaced
alpha0 = 2.0
rho0 = (alpha0 * a.dag() - alpha0.conjugate() * a).expm() * qt.fock_dm(N, 0) * \
       (-alpha0 * a.dag() + alpha0.conjugate() * a).expm()

# Evoluzione mesolve (o steadystate per long time)
times = np.arange(0, T_sim, dt)
result = qt.mesolve(H, rho0, times, c_ops=c_ops, e_ops=[x, num])

# Estrazione <x(t)> e varianza per delta RR proxy
expect_x = result.expect[0]
var_x = [qt.expect(x**2, result.states[i]) - expect_x[i]**2 for i in range(len(times))]

# Simula RR series (semplicistico)
T0 = 1 / 1.1
delta_t = expect_x / 0.1   # v_rep ~0.1 V/s proxy
rr_times = np.cumsum(T0 + delta_t * dt + 0.001 * np.random.randn(len(times)))  # + noise
rr_series = np.diff(rr_times) * 1000  # in ms

# Metriche HRV
sdnn = np.std(rr_series)
rmssd = np.sqrt(np.mean(np.diff(rr_series)**2))

print(f"SDNN quantistico: {sdnn:.2f} ms")
print(f"RMSSD quantistico: {rmssd:.2f} ms")

# Plot
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

axs[0].plot(times, expect_x, label='<x(t)>')
axs[0].set_ylabel('Quadratura <x>')
axs[0].legend(); axs[0].grid(True)

axs[1].plot(times[:-1], rr_series, 'r-', lw=1.2, label='RR series (ms)')
axs[1].set_ylabel('RR (ms)'); axs[1].legend(); axs[1].grid(True)

axs[2].plot(times, var_x, 'k-', label='Var(x(t))')
axs[2].axhline(0.5, color='gray', ls='--', label='shot-noise limit')
axs[2].set_xlabel('Tempo (s)'); axs[2].set_ylabel('Varianza quadratura')
axs[2].legend(); axs[2].grid(True)

plt.tight_layout()
plt.savefig('figures/rr_series_quant_vs_class.png', dpi=300)
plt.show()

# Per figure aggiuntive: phase space o PSD -> usa qutip.bloch o welch PSD su rr_series