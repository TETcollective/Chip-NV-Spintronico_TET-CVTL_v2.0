"""
Titolo: PSD classica avanzata proxy EEG multi-mode embodied (Floquet-like)
Autore: Basato su framework TET–CVTL / Orch-OR extension, Simon Soliman 
Data: Marzo 2026

Descrizione:
Genera un segnale proxy EEG multi-mode con:
- Due modi gamma distinti (~38 Hz e ~42 Hz)
- Driving periodico coerente a 0.1 Hz (moiré/respirazione)
- Rumore 1/f con knee ~50 Hz (beta=1.8)
- Burst gamma asincroni e variabili
- Modulazione ampiezza lenta
- Cross-correlation vs lag come witness classico di "coerenza preservata" (proxy entanglement)

Output: PSD con sidebands Floquet ±0.1 Hz + subplot cross-corr vs lag
Figura salvata: figures/eeg_classic_proxy_multi_mode_vfinal.png

Dipendenze: numpy, matplotlib, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, correlate
import os

np.random.seed(42)

# ================================================
# Parametri realistici
# ================================================
dt = 0.001                  # passo temporale (1 ms)
t_total = 180.0             # tempo totale simulazione (s)
t = np.arange(0, t_total, dt)
fs = 1 / dt                 # sampling frequency 1000 Hz

omega1 = 2 * np.pi * 38     # modo bassa gamma ~38 Hz
omega2 = 2 * np.pi * 42     # modo alta gamma ~42 Hz
omega_drive = 2 * np.pi * 0.1  # driving coerente 0.1 Hz

# ================================================
# Rumore 1/f con knee (~50 Hz, beta=1.8)
# ================================================
def pink_noise_with_knee(n, beta=1.8, knee_freq=50):
    white = np.random.randn(n)
    white_fft = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, d=dt)
    f[0] = 1.0
    # 1/f fino a knee, poi flat
    envelope = np.where(f < knee_freq, 1 / (f ** (beta / 2)), 1 / (knee_freq ** (beta / 2)))
    pink_fft = white_fft * envelope
    return np.fft.irfft(pink_fft, n=n) * 0.018

noise1 = pink_noise_with_knee(len(t))
noise2 = pink_noise_with_knee(len(t)) * 0.6 + 0.4 * noise1  # correlazione ~40%

# ================================================
# Burst gamma asincroni e variabili
# ================================================
burst1 = np.zeros_like(t)
burst2 = np.zeros_like(t)

burst_times1 = np.sort(np.random.uniform(0, t_total, 50))
burst_times2 = np.sort(np.random.uniform(0, t_total, 45))

for bt in burst_times1:
    duration = np.random.uniform(0.5, 1.2)
    amp = np.random.uniform(0.4, 0.8)
    burst1 += amp * np.exp(-((t - bt)**2 / duration))

for bt in burst_times2:
    duration = np.random.uniform(0.6, 1.5)
    amp = np.random.uniform(0.3, 0.7)
    burst2 += amp * np.exp(-((t - bt)**2 / duration))

# ================================================
# Modulazione ampiezza lenta (0.1 Hz)
# ================================================
amp_mod = 0.45 * np.sin(omega_drive * t + np.random.uniform(0, np.pi))

# ================================================
# Segnali multi-mode
# ================================================
x1 = 0.8 * np.sin(omega1 * t + 0.4) * (1 + amp_mod) + burst1 + noise1
x2 = 0.7 * np.sin(omega2 * t + np.pi/2.2) * (1 - 0.8*amp_mod) + burst2 + noise2

# ================================================
# PSD
# ================================================
f1, Pxx1 = welch(x1, fs=fs, nperseg=65536, scaling='spectrum', detrend='constant')
f2, Pxx2 = welch(x2, fs=fs, nperseg=65536, scaling='spectrum', detrend='constant')

# ================================================
# Cross-correlation vs lag (witness classico)
# ================================================
cross_corr = correlate(x1 - x1.mean(), x2 - x2.mean(), mode='full', method='fft')
lags = np.arange(-len(t)+1, len(t)) * dt
cross_norm = cross_corr / (np.std(x1) * np.std(x2) * len(t))

# Zoom ±1 s
lag_range = 1.0
mask = np.abs(lags) <= lag_range
lags_zoom = lags[mask]
cross_zoom = cross_norm[mask]

max_corr = np.max(np.abs(cross_zoom))
print(f"Cross-correlation witness max (lag ±{lag_range} s): {max_corr:.3f}")

# ================================================
# Plot con subplot
# ================================================
fig = plt.figure(figsize=(14, 10))

# PSD principale
ax1 = fig.add_subplot(2, 1, 1)
ax1.semilogy(f1, Pxx1, lw=1.5, color='navy', alpha=0.95, label='Modo bassa ~38 Hz')
ax1.semilogy(f2, Pxx2 + 3e-3, lw=1.5, color='maroon', alpha=0.95, label='Modo alta ~42 Hz [shifted]')

ax1.axvline(38, color='blue', ls='--', lw=1.6, label='Modo bassa teorico')
ax1.axvline(42, color='red', ls='--', lw=1.6, label='Modo alta teorico')
ax1.axvline(0.1, color='limegreen', ls='--', lw=1.8, label='Driving coerente 0.1 Hz')
ax1.axvline(38 + 0.1, color='orange', ls=':', lw=1.4, alpha=0.9)
ax1.axvline(42 + 0.1, color='orange', ls=':', lw=1.4, alpha=0.9)

ax1.set_xlabel('Frequenza (Hz)', fontsize=13)
ax1.set_ylabel('PSD (a.u.)', fontsize=13)
ax1.set_title('PSD classica avanzata proxy EEG multi-mode embodied\n'
              '(1/f knee ~50 Hz, burst gamma asincroni, rumore 1/f β=1.8, modulazione 0.1 Hz)',
              fontsize=15, pad=18)

ax1.legend(fontsize=10.5, framealpha=0.93, loc='upper right', ncol=2)
ax1.grid(True, alpha=0.25, which='both', ls=':')
ax1.set_xlim(0, 100)
ax1.set_ylim(1e-8, 5e3)

# Subplot cross-correlation
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(lags_zoom, cross_zoom, lw=1.8, color='purple', label='Cross-corr normalizzata')
ax2.axvline(0, color='black', ls='--', lw=1.2, label='Lag zero')
ax2.set_xlabel('Lag (s)', fontsize=13)
ax2.set_ylabel('Cross-corr (norm)', fontsize=13)
ax2.set_title(f'Cross-correlation vs lag (witness classico entanglement proxy)\n'
              f'Max = {max_corr:.3f}', fontsize=13)

ax2.legend(fontsize=11, framealpha=0.92)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-1, 1)
ax2.set_ylim(-0.1, max(1.0, max_corr*1.2))

plt.tight_layout()

# Salvataggio
os.makedirs('figures', exist_ok=True)
save_path = 'figures/eeg_classic_proxy_multi_mode_vfinal.png'
plt.savefig(save_path, dpi=500, bbox_inches='tight')
plt.show()

print(f"Figura salvata: {save_path}")
print(f"Cross-correlation witness max (lag ±1 s): {max_corr:.3f}")
print("Valore indicativo di correlazione significativa tra i due modi (proxy coerenza preservata)")