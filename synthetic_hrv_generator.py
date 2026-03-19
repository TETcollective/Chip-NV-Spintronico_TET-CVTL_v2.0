#!/usr/bin/env python3
"""
synthetic_hrv_generator.py

Generatore di serie HRV realistiche per test:
- Oscillazioni respiratorie (HF)
- Oscillazioni Mayer (LF)
- Rumore 1/f (pink noise)
- Artefatti opzionali

Utile per validare codice HRV quando mancano dati reali.

Autore: Simon Soliman / TET Collective
Data: 15 Marzo 2026
"""

import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results/hrv_synthetic")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_realistic_hrv(N: int = 1800,
                           mean_rr: float = 820,
                           hf_amp: float = 45,
                           lf_amp: float = 35,
                           pink_noise_std: float = 22,
                           seed: int = 42) -> np.ndarray:
    """
    Serie RR sintetica realistica.
    """
    np.random.seed(seed)
    t = np.arange(N)

    # Oscillazione respiratoria (HF ~0.25 Hz)
    hf = hf_amp * np.sin(2 * np.pi * 0.25 * t / 60)

    # Oscillazione Mayer / baroriflesso (LF ~0.1 Hz)
    lf = lf_amp * np.sin(2 * np.pi * 0.1 * t / 60)

    # Rumore 1/f (pink noise approssimato)
    freq = np.fft.rfftfreq(N, d=1)
    pink = 1 / (freq + 1e-6)  # 1/f
    pink[0] = 0
    pink_phase = np.random.uniform(0, 2*np.pi, len(freq))
    pink_fft = pink * np.exp(1j * pink_phase) * pink_noise_std
    pink_time = np.fft.irfft(pink_fft)

    rr = mean_rr + hf + lf + pink_time

    # Aggiungi qualche ectopic beat raro
    ectopic_idx = np.random.choice(N, size=3, replace=False)
    rr[ectopic_idx] *= np.random.uniform(0.6, 0.8, 3)

    return rr


if __name__ == "__main__":
    rr = generate_realistic_hrv(N=2400, mean_rr=810, hf_amp=50, lf_amp=40, pink_noise_std=28)

    df = pd.DataFrame({'RR_ms': rr})
    df.to_csv(RESULTS_DIR / "synthetic_hrv_realistic.csv", index=False)

    plt.figure(figsize=(14,6))
    plt.plot(rr, lw=1.1, color='navy')
    plt.title("Serie HRV sintetica realistica (TET–CVTL test)")
    plt.xlabel("Beat number")
    plt.ylabel("RR interval (ms)")
    plt.grid(True, alpha=0.3)
    plt.savefig(RESULTS_DIR / "synthetic_hrv_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Serie HRV sintetica generata: {len(rr)} campioni")
    print(f"Media RR: {np.mean(rr):.1f} ms")
    print(f"SD RR:    {np.std(rr):.1f} ms")
    print(f"File salvati in: {RESULTS_DIR}")