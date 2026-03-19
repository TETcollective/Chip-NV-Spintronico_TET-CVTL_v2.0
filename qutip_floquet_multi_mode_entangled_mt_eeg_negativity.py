"""
Titolo: Simulazione quantistica Floquet-Lindblad multi-mode entangled MT proxy EEG con negativity entanglement
Autore: Basato su framework TET–CVTL / Orch-OR extension
Data: Marzo 2026

Descrizione:
- Due modi vibrazionali accoppiati (gamma ~38/42 Hz)
- Stato iniziale Bell-like entangled
- Driving periodico 0.1 Hz (moiré/respirazione)
- Dissipazione Lindblad realistica
- Output: PSD con sidebands Floquet + cross-corr vs lag + negativity vs tempo (misura entanglement vera)
- Negativity calcolata su stato ridotto parziale (traccia sul secondo modo)

Nota: su Colab free ~2–5 min con N=4. Riduci N a 3 o n_points a 4000 se troppo lento.
"""

# !pip install -q qutip  # Esegui una volta

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from scipy.signal import welch, correlate
import os

print("QuTiP version:", qt.__version__)

# ================================================
# Parametri quantistici (bilanciati per velocità)
# ================================================
N = 4                       # 16×16 = 256 dim
omega_low  = 2 * np.pi * 38
omega_high = 2 * np.pi * 42
omega_drive = 2 * np.pi * 0.1
g_coupling = 2 * np.pi * 0.7   # coupling forte per entanglement visibile
Omega_drive = 2 * np.pi * 0.2  # driving visibile
gamma_decay = 1.5e6            # decoerenza bilanciata
t_total = 20.0
n_points = 6000                # ridotto per velocità
fs = n_points / t_total

print(f"Dim Hilbert: {N*N} | t_total: {t_total}s | punti: {n_points}")

# Operatori
a1 = qt.tensor(qt.destroy(N), qt.qeye(N))
a2 = qt.tensor(qt.qeye(N), qt.destroy(N))

H0 = omega_low * a1.dag() * a1 + omega_high * a2.dag() * a2 + \
     g_coupling * (a1.dag() * a2 + a1 * a2.dag())

def driving_coeff(t, args):
    return np.cos(omega_drive * t)

V_drive = Omega_drive * (a1 + a1.dag() + a2 + a2.dag())
H = [H0, [V_drive, driving_coeff]]

c_ops = [np.sqrt(gamma_decay) * a1, np.sqrt(gamma_decay) * a2]

# Stato iniziale entangled Bell-like
state0 = qt.basis(N, 0)
state1 = qt.basis(N, 1)
psi_ent = (qt.tensor(state0, state1) + qt.tensor(state1, state0)).unit()
psi0 = psi_ent

tlist = np.linspace(0, t_total, n_points)

options = {
    "method": "bdf",
    "nsteps": 1000000,
    "atol": 1e-7,
    "rtol": 1e-5,
    "progress_bar": "enhanced"
}

# ================================================
# Simulazione
# ================================================
print("Simulazione quantistica in corso (2–5 min)...")
result = qt.mesolve(H, psi0, tlist, c_ops=c_ops,
                    e_ops=[a1 + a1.dag(), a2 + a2.dag()],
                    options=options)

expect_x1 = result.expect[0]
expect_x2 = result.expect[1]

print("Simulazione completata.")

# ================================================
# Negativity vs tempo (misura entanglement vera)
# ================================================
print("Calcolo negativity (entanglement witness quantistico)...")
negativity = []
for state in result.states[::100]:  # ogni 100 passi per velocità
    rho_ab = state.ptrace([0,1])    # stato ridotto ai due modi
    neg = qt.entanglement.negativity(rho_ab, [0])  # negativity su sottosistema 1
    negativity.append(neg)

neg_time = tlist[::100]  # tempi campionati

# ================================================
# PSD
# ================================================
f1, Pxx1 = welch(expect_x1, fs=fs, nperseg=4096, noverlap=2048, scaling='spectrum')
f2, Pxx2 = welch(expect_x2, fs=fs, nperseg=4096, noverlap=2048, scaling='spectrum')

# ================================================
# Cross-correlation vs lag
# ================================================
cross_corr = correlate(expect_x1 - expect_x1.mean(), expect_x2 - expect_x2.mean(), mode='full', method='fft')
lags = np.arange(-len(tlist)+1, len(tlist)) * (t_total / (n_points-1))
cross_norm = cross_corr / (np.std(expect_x1) * np.std(expect_x2) * n_points)

lag_range = 1.0
mask = np.abs(lags) <= lag_range
lags_zoom = lags[mask]
cross_zoom = cross_norm[mask]

max_corr = np.max(np.abs(cross_zoom))
print(f"Cross-correlation witness max (lag ±1 s): {max_corr:.3f}")

# ================================================
# Plot: PSD + Negativity + Cross-corr
# ================================================
fig = plt.figure(figsize=(14, 12))

# PSD
ax1 = fig.add_subplot(3, 1, 1)
ax1.semilogy(f1, Pxx1, lw=1.5, color='navy', label='Modo bassa ~38 Hz')
ax1.semilogy(f2, Pxx2 + 1e-3, lw=1.5, color='maroon', label='Modo alta ~42 Hz [shifted]')
ax1.axvline(38, color='blue', ls='--')
ax1.axvline(42, color='red', ls='--')
ax1.axvline(0.1, color='limegreen', ls='--')
ax1.axvline(38 + 0.1, color='orange', ls=':')
ax1.axvline(42 + 0.1, color='orange', ls=':')
ax1.set_xlabel('Frequenza (Hz)')
ax1.set_ylabel('PSD (a.u.)')
ax1.set_title('PSD quantistica Floquet-Lindblad multi-mode entangled MT proxy EEG')
ax1.legend()
ax1.grid(True, alpha=0.25)
ax1.set_xlim(0, 100)

# Negativity vs tempo
ax2 = fig.add_subplot(3, 1, 2)
ax2.plot(neg_time, negativity, lw=1.8, color='darkgreen', label='Negativity (entanglement)')
ax2.axhline(0, color='black', ls='--', alpha=0.5)
ax2.set_xlabel('Tempo (s)')
ax2.set_ylabel('Negativity')
ax2.set_title('Negativity vs tempo (misura entanglement quantistica)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.05, 0.5)

# Cross-corr vs lag
ax3 = fig.add_subplot(3, 1, 3)
ax3.plot(lags_zoom, cross_zoom, lw=1.8, color='purple', label='Cross-corr normalizzata')
ax3.axvline(0, color='black', ls='--')
ax3.set_xlabel('Lag (s)')
ax3.set_ylabel('Cross-corr (norm)')
ax3.set_title(f'Cross-correlation vs lag (witness classico)\nMax = {max_corr:.3f}')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-1, 1)

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
save_path = 'figures/eeg_quantum_floquet_multi_entangled_negativity.png'
plt.savefig(save_path, dpi=400, bbox_inches='tight')
plt.show()

print(f"Figura salvata: {save_path}")
print(f"Negativity iniziale: {negativity[0]:.3f}")
print(f"Negativity finale: {negativity[-1]:.3f}")
print(f"Cross-correlation witness max: {max_corr:.3f}")