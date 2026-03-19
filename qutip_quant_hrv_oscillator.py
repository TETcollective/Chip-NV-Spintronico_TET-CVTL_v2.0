"""
Titolo: Calcolo metriche HRV realistiche da simulazione quantistica Floquet-Lindblad
Autore: Assistito da Grok per @PhysSoliman (TETcollective.org)
Data: Marzo 2026

Descrizione:
Simula oscillatore bosonico damped-driven embodied (pacemaker ~1.1 Hz + driving respiratorio 0.1 Hz)
Calcola serie RR quantistica e classica proxy
Metriche HRV complete: SDNN, RMSSD, pNN50, LF/HF ratio, potenza HF (0.15–0.4 Hz)
Figura: serie RR + PSD + metriche stampate

Output: figures/hrv_quant_vs_class_realistic.png
"""

# !pip install -q qutip

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from scipy.signal import welch
import os

print("QuTiP version:", qt.__version__)

# ================================================
# Parametri realistici per HRV umana (sani)
# ================================================
N = 40                  # cutoff Fock
omega0 = 2 * np.pi * 1.1  # ~1.1 Hz (pacemaker intrinseco ~55 bpm)
chi = 0.02 * omega0     # Kerr debole
gamma = 0.4             # damping
kappa = 0.25            # thermal bath
n_th = 5000             # ~310 K
Gamma_phi = 8.0         # dephasing
g0 = 0.12 * omega0      # driving respiratorio debole
f_resp = 0.1            # 0.1 Hz coerente
T_sim = 300.0           # 5 min (300 s) → ~330 battiti
dt = 0.002              # step fine
v_rep = 0.015           # scala quadratura → delta RR (calibrato per SDNN ~50–80 ms quant)

# Operatori
a = qt.destroy(N)
ad = a.dag()
num = ad * a
x = (a + ad) / np.sqrt(2)

H0 = omega0 * num + chi * num**2
def drive_coeff(t, args):
    return g0 * np.sin(2 * np.pi * f_resp * t)
H = [H0, [a + ad, drive_coeff]]

c_ops = [
    np.sqrt(gamma) * a,
    np.sqrt(kappa * (n_th + 1)) * ad,
    np.sqrt(kappa * n_th) * a,
    np.sqrt(Gamma_phi) * num
]

rho0 = qt.coherent_dm(N, 2.5)  # alpha0 ridotto per oscillazioni più fisiologiche

times = np.arange(0, T_sim, dt)

options = {
    "method": "bdf",
    "nsteps": 500000,
    "atol": 1e-8,
    "rtol": 1e-6,
    "progress_bar": "enhanced"
}

print("Simulazione in corso (~1–3 min)...")
result = qt.mesolve(H, rho0, times, c_ops=c_ops, e_ops=[x], options=options)

expect_x = result.expect[0]

# ================================================
# Serie RR quantistica
# ================================================
T0 = 1.0 / 1.1  # ~909 ms
delta_t = expect_x / v_rep
rr_cum = np.cumsum(T0 + delta_t * dt)

step_battito = int(1 / (dt * 1.1))  # ~909 punti per battito
rr_times = rr_cum[::step_battito]
rr_series_quant = np.diff(rr_times) * 1000  # ms

# Proxy classico: media + rumore gaussiano realistico (SDNN ~100–140 ms)
rr_series_class = np.ones_like(rr_series_quant) * 909 + 120 * np.random.randn(len(rr_series_quant))

# ================================================
# Metriche HRV complete
# ================================================
def calc_hrv(rr):
    if len(rr) < 2:
        return 0, 0, 0, 0, 0
    sdnn = np.std(rr)
    rmssd = np.sqrt(np.mean(np.diff(rr)**2))
    pnn50 = 100 * np.sum(np.abs(np.diff(rr)) > 50) / (len(rr) - 1)
    return sdnn, rmssd, pnn50

sdnn_q, rmssd_q, pnn50_q = calc_hrv(rr_series_quant)
sdnn_c, rmssd_c, pnn50_c = calc_hrv(rr_series_class)

# PSD per HF power e LF/HF
fs_rr = 1 / np.mean(np.diff(rr_times)) if len(rr_times) > 1 else 1
f_q, Pxx_q = welch(rr_series_quant, fs=fs_rr, nperseg=min(256, len(rr_series_quant)//2), detrend='linear')
f_c, Pxx_c = welch(rr_series_class, fs=fs_rr, nperseg=min(256, len(rr_series_class)//2), detrend='linear')

lf_mask_q = (f_q >= 0.04) & (f_q < 0.15)
hf_mask_q = (f_q >= 0.15) & (f_q <= 0.4)
lf_power_q = np.trapz(Pxx_q[lf_mask_q], f_q[lf_mask_q]) if np.any(lf_mask_q) else 0
hf_power_q = np.trapz(Pxx_q[hf_mask_q], f_q[hf_mask_q]) if np.any(hf_mask_q) else 0
lfhf_q = lf_power_q / hf_power_q if hf_power_q > 0 else np.inf

lf_mask_c = (f_c >= 0.04) & (f_c < 0.15)
hf_mask_c = (f_c >= 0.15) & (f_c <= 0.4)
lf_power_c = np.trapz(Pxx_c[lf_mask_c], f_c[lf_mask_c]) if np.any(lf_mask_c) else 0
hf_power_c = np.trapz(Pxx_c[hf_mask_c], f_c[hf_mask_c]) if np.any(hf_mask_c) else 0
lfhf_c = lf_power_c / hf_power_c if hf_power_c > 0 else np.inf

print("\nMetriche HRV:")
print(f"Quantistico: SDNN {sdnn_q:.1f} ms | RMSSD {rmssd_q:.1f} ms | pNN50 {pnn50_q:.1f}% | LF/HF {lfhf_q:.2f} | HF power {hf_power_q:.2e}")
print(f"Classico:    SDNN {sdnn_c:.1f} ms | RMSSD {rmssd_c:.1f} ms | pNN50 {pnn50_c:.1f}% | LF/HF {lfhf_c:.2f} | HF power {hf_power_c:.2e}")

# ================================================
# Plot
# ================================================
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

# <x(t)> zoom 20 s
zoom_idx = times <= 20
axs[0].plot(times[zoom_idx], expect_x[zoom_idx], 'b-', lw=1.5, label=r'$\langle \hat{x}(t) \rangle$ (zoom 20 s)')
axs[0].set_ylabel('Quadratura')
axs[0].legend(); axs[0].grid(True, alpha=0.4)

# Serie RR
axs[1].plot(rr_times[:-1], rr_series_quant, 'r-', lw=1.2, label=f'Quantistico (SDNN {sdnn_q:.1f} ms)')
axs[1].plot(rr_times[:-1], rr_series_class, 'b--', lw=1.0, alpha=0.7, label=f'Classico (SDNN {sdnn_c:.1f} ms)')
axs[1].set_ylabel('RR interval (ms)')
axs[1].legend(); axs[1].grid(True, alpha=0.4)

# PSD
axs[2].semilogy(f_q, Pxx_q, 'r-', label='Quantistico')
axs[2].semilogy(f_c, Pxx_c, 'b--', label='Classico')
axs[2].set_xlabel('Frequenza (Hz)')
axs[2].set_ylabel('PSD (a.u.)')
axs[2].axvspan(0.1, 0.4, alpha=0.15, color='gray', label='HF band')
axs[2].legend(); axs[2].grid(True, alpha=0.4)

plt.suptitle('Analisi HRV: Simulazione TET–CVTL Quantistica vs Classica')
plt.tight_layout(rect=[0, 0, 1, 0.96])
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/rr_series_quant_vs_class_realistic.png', dpi=300, bbox_inches='tight')
plt.show()

print("Figura salvata: figures/rr_series_quant_vs_class_realistic.png")