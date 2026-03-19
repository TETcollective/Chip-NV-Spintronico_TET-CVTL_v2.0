# =============================================================================
# TETcollective - FHN Quantistico Semiclassico: Phase Space & Time Series
# File: code/qutip_fhn_semiclassical_phase.py
# Descrizione: Integrazione numerica semiclassica del FHN (media <v>, <w>)
#              con fluttuazioni quantistiche gaussiane aggiunte per proxy.
#              Produce phase portrait + time series v(t) con noise.
# Autore: Assistito da Grok per @PhysSoliman (TETcollective.org)
# Data: Marzo 2026
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

# Parametri
epsilon = 0.008
a = 0.15
b = 1.0
tau = 12.5
I_ext = 0.0

t = np.linspace(0, 200, 20000)

def fhn_semiclass(y, t):
    v, w = y
    dvdt = v - (v**3)/3 - w + I_ext
    dwdt = (v + a - b * w) / tau
    return [dvdt, dwdt]

y0 = [0.5, 0.5]
sol = odeint(fhn_semiclass, y0, t)
v = sol[:, 0]
w = sol[:, 1]

delta_v_quant = 0.05 * np.random.randn(len(t))
v_quant = v + delta_v_quant

# Nullcline
v_null = np.linspace(-2.5, 2.5, 300)
w_cubic  = v_null - (v_null**3)/3 + I_ext
w_linear = (v_null + a) / b

# Figura
fig = plt.figure(figsize=(9.8, 8.0))

ax = fig.add_subplot(111)

ax.plot(v, w, 'r-', lw=1.8, label='orbita media (semiclassica)')
ax.plot(v_quant, w, color='red', alpha=0.40, lw=0.9, 
        label='+ fluttuazioni quantistiche')

ax.plot(v_null, w_cubic,  color='limegreen', ls='--', lw=1.7, 
        label=r'$\dot{v}=0$')
ax.plot(v_null, w_linear, color='deepskyblue', ls='--', lw=1.7, 
        label=r'$\dot{w}=0$')

ax.set_xlabel(r'$v$  (potenziale normalizzato)', fontsize=12.5)
ax.set_ylabel(r'$w$  (variabile di recupero)',   fontsize=12.5)

ax.set_title(
    "FHN semiclassico con fluttuazioni quantistiche\n"
    r"$\varepsilon = 0.008,\ a = 0.15,\ b = 1.0,\ \tau = 12.5$",
    fontsize=13.5,
    pad=22
)

ax.legend(
    loc='upper right',
    bbox_to_anchor=(0.98, 0.98),
    fontsize=9.8,
    framealpha=0.93,
    edgecolor='0.8'
)

ax.grid(True, alpha=0.18, ls=':')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-1.8, 2.0)

# Inset spostato in BASSO a SINISTRA
inset = fig.add_axes([0.14, 0.14, 0.26, 0.22])   # ← qui il cambiamento principale

inset.plot(t[:5000], v_quant[:5000], 'r-', lw=1.3, alpha=0.95)
inset.set_title(r'$v(t)$   $\delta v \approx 0.05$', fontsize=10, pad=5)
inset.set_xlabel('tempo (s)', fontsize=9.5)
inset.set_ylabel('$v$', fontsize=9.5)
inset.tick_params(axis='both', labelsize=8.5)
inset.grid(True, alpha=0.25)

plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.93])

os.makedirs('figures', exist_ok=True)
plt.savefig('figures/fhn_phase_space.png', dpi=380, bbox_inches='tight')
plt.close()

print("Salvata: figures/fhn_phase_space.png")