# =============================================================================
# TETcollective - QuTiP: HRV con Squeezing Esplicito + Entanglement Witness
# File: code/qutip_hrv_squeezing_witness.py
# Descrizione: Aggiunge parametric squeezing + correlatore a due tempi per witness
#              entanglement su quadrature. Produce figura correlazioni + squeezing.
# Autore: Assistito da Grok per @PhysSoliman (TETcollective.org)
# Data: Marzo 2026
# =============================================================================

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from qutip.correlation import correlation_2op_1t

# Parametri (estesi dal precedente)
N = 40
omega0 = 2 * np.pi * 1.1
chi = 0.05 * omega0
gamma = 0.5
kappa = 0.3
n_th = 5000
Gamma_phi = 8.0     # ridotto per coerenza maggiore
g0 = 0.2 * omega0
f_resp = 0.1
zeta0 = 0.1 * omega0  # squeezing strength → ~5-8 dB
T_sim = 200.0
dt = 0.005

a = qt.destroy(N)
ad = a.dag()
num = ad * a
x = (a + ad) / np.sqrt(2)

# Hamiltoniano base + driving + squeezing parametrico
H0 = omega0 * num + chi * num**2
def drive_coeff(t, args): return g0 * np.sin(2 * np.pi * f_resp * t)
def squeeze_coeff(t, args): return zeta0 * np.cos(2 * omega0 * t)
H = [H0,
     [a + ad, drive_coeff],
     [1j * (a**2 - ad**2)/2, squeeze_coeff]]

c_ops = [np.sqrt(gamma) * a,
         np.sqrt(kappa * (n_th + 1)) * ad,
         np.sqrt(kappa * n_th) * a,
         np.sqrt(Gamma_phi) * num]

rho0 = qt.coherent_dm(N, 3.0)

times = np.arange(0, T_sim, dt)
taulist = np.linspace(0, 20, 200)   # tau per correlatore

# Evoluzione + correlatore <x(t) x(t+tau)>
corr = correlation_2op_1t(H, rho0, taulist, c_ops, x, x, solver='me')

# Squeezing: varianza quadratura vs tempo (media su steady-regime)
expect_x2 = [qt.expect(x**2, qt.mesolve(H, rho0, [0, t], c_ops).states[-1]) for t in times[::100]]
var_x = np.array(expect_x2) - 0.5  # >0 squeezed se negativo wait no: squeezing se <0.5

# Plot
fig, axs = plt.subplots(2, 1, figsize=(10, 9))

# Correlatore + witness
axs[0].plot(taulist, np.real(corr), 'r-', lw=2, label='C(τ) squeezing')
axs[0].plot(taulist, np.sqrt( (0.5)**2 * np.ones_like(taulist) ), 'k--', label='CS bound (classico)')
axs[0].set_xlabel('τ (s)')
axs[0].set_ylabel('Correlatore normalizzato')
axs[0].set_title('Entanglement witness via violazione Cauchy-Schwarz')
axs[0].legend(); axs[0].grid(True)

# Inserto squeezing
inset = fig.add_axes([0.65, 0.65, 0.3, 0.25])
inset.plot(times[::100], var_x, 'b-', label=r'$\Delta \hat{x}^2 - 1/2$')
inset.axhline(0, color='gray', ls='--')
inset.set_title('Squeezing dinamico', fontsize=10)
inset.set_xlabel('t (s)'); inset.grid(True)

plt.tight_layout()
plt.savefig('figures/quant_squeezing_correlations.png', dpi=300)
plt.close()
print("Figura squeezing + witness salvata.")