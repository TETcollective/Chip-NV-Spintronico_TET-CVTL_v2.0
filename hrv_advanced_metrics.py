#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hrv_advanced_metrics.py

Calcolo completo HRV per TET–CVTL paper:
- Sample Entropy (SampEn) vettoriale veloce
- Detrended Fluctuation Analysis (DFA α) con fit robusto
- PSD Welch per bande VLF/LF/HF + normalized HF power
- Proxy entanglement witness (SampEn + DFA + HF + coherence placeholder)

Citabile nel paper per:
- Item 4: protocolli quantum sensing embodied
- Falsificabilità HRV/NV-enhanced 2026–2030
- Proxy per entanglement persistente embodied

Autore: Simon Soliman / TET Collective
Data: 15 Marzo 2026
Versione: 2.1
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional

warnings.filterwarnings("ignore", category=RuntimeWarning)

RESULTS_DIR = Path("results/hrv")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def sample_entropy(rr: np.ndarray,
                   m: int = 2,
                   r_factor: float = 0.2,
                   normalize: bool = True) -> Dict[str, float]:
    """
    Sample Entropy ottimizzato con broadcasting NumPy.
    Basso valore = alta regolarità → proxy coerenza quantistica.
    """
    N = len(rr)
    if N < 200:
        return {'sampen': np.nan, 'error': 'serie troppo corta'}

    if normalize:
        rr = (rr - np.mean(rr)) / (np.std(rr, ddof=1) + 1e-12)

    r = r_factor * np.std(rr)

    def count_matches(dim: int) -> int:
        templates = np.lib.stride_tricks.sliding_window_view(rr, dim)
        # Distanza Chebyshev (max-norm)
        dist_matrix = np.max(np.abs(templates[:, None] - templates[None, :]), axis=2)
        np.fill_diagonal(dist_matrix, np.inf)
        return np.sum(dist_matrix < r)

    B = count_matches(m)
    A = count_matches(m + 1)

    if B == 0 or A == 0:
        return {'sampen': np.nan, 'A': A, 'B': B, 'r_used': r}

    return {'sampen': -np.log(A / B), 'A': A, 'B': B, 'r_used': r}


def detrended_fluctuation_analysis(rr: np.ndarray,
                                   min_box: int = 4,
                                   max_box_ratio: float = 0.25,
                                   poly_order: int = 2,
                                   return_scales: bool = False) -> Dict:
    """
    DFA con validazione statistica del fit.
    """
    N = len(rr)
    if N < 300:
        return {'alpha': np.nan, 'error': 'serie troppo corta'}

    y = (rr - np.mean(rr)) / (np.std(rr, ddof=1) + 1e-12)
    Y = np.cumsum(y)

    max_box = max(min_box, int(max_box_ratio * N))
    box_sizes = np.unique(np.logspace(np.log10(min_box), np.log10(max_box), num=40, dtype=int))

    F_n = np.zeros(len(box_sizes))

    for i, n in enumerate(box_sizes):
        n_seg = N // n
        rms = []
        for v in range(n_seg):
            seg = Y[v*n : (v+1)*n]
            t = np.arange(len(seg))
            if len(t) < poly_order + 1:
                continue
            p = np.polyfit(t, seg, poly_order)
            trend = np.polyval(p, t)
            detr = seg - trend
            rms.append(np.sqrt(np.mean(detr**2) + 1e-20))

        if len(rms) == 0:
            F_n[i] = np.nan
        else:
            F_n[i] = np.sqrt(np.mean(np.array(rms)**2))

    valid = (F_n > 0) & np.isfinite(np.log(F_n)) & np.isfinite(np.log(box_sizes))
    if np.sum(valid) < 10:
        return {'alpha': np.nan, 'error': 'pochi punti validi per fit'}

    log_n = np.log(box_sizes[valid])
    log_F = np.log(F_n[valid])

    res = stats.linregress(log_n, log_F)
    result = {
        'alpha': res.slope,
        'r_squared': res.rvalue**2,
        'p_value': res.pvalue,
        'std_err': res.stderr,
        'n_points': np.sum(valid)
    }

    if return_scales:
        result['box_sizes'] = box_sizes
        result['F_n'] = F_n

    return result


def compute_psd_hrv_bands(rr: np.ndarray,
                          fs_resample: float = 4.0) -> Dict:
    """
    PSD Welch con bande HRV standard.
    """
    # Cumulativa → tempo in secondi
    t_cum = np.cumsum(rr / 1000.0)
    t_new = np.arange(0, t_cum[-1], 1/fs_resample)
    if len(t_new) < 128:
        return {'error': 'serie troppo corta dopo resampling'}

    rr_interp = np.interp(t_new, t_cum[:-1], rr[:-1])

    f, Pxx = signal.welch(rr_interp, fs=fs_resample,
                          nperseg=min(256, len(rr_interp)//2),
                          noverlap=None)

    def band_power(low: float, high: float) -> float:
        mask = (f >= low) & (f <= high)
        return np.trapz(Pxx[mask], f[mask]) if np.any(mask) else 0.0

    total = band_power(0, 0.5) + 1e-12
    vlf = band_power(0.003, 0.04) / total
    lf  = band_power(0.04, 0.15) / total
    hf  = band_power(0.15, 0.40) / total

    return {
        'hf_norm': hf,
        'lf_hf_ratio': lf / (hf + 1e-10),
        'vlf_norm': vlf,
        'total_power': total,
        'freq': f,
        'psd': Pxx
    }


def entanglement_proxy_hrv(rr: np.ndarray) -> Dict[str, float]:
    """
    Proxy entanglement witness completo.
    """
    try:
        se = sample_entropy(rr)['sampen']
        dfa = detrended_fluctuation_analysis(rr)['alpha']
        psd = compute_psd_hrv_bands(rr)
        hf = psd['hf_norm']

        # Coherence proxy semplificato (più basso SampEn + α vicino 1 + HF alto → meglio)
        base = np.exp(-5.2 * se) * (1 - np.abs(dfa - 1)) * hf
        bonus = 0.45 * np.exp(-2 * (se + np.abs(dfa - 1)))

        proxy = base + bonus

        return {
            'proxy_value': proxy,
            'sampen': se,
            'dfa_alpha': dfa,
            'hf_norm': hf,
            'lf_hf_ratio': psd['lf_hf_ratio'],
            'base_score': base,
            'bonus': bonus
        }
    except Exception as e:
        return {'proxy_value': np.nan, 'error': str(e)}


# ──────────────────────────────────────────────────────────────────────────────
# ESEMPIO ESECUZIONE + SALVATAGGIO
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Serie sintetica realistica
    np.random.seed(42)
    N = 1800
    t = np.arange(N)
    trend = 820 + 45 * np.sin(2*np.pi*0.08*t/60) + 18 * np.sin(2*np.pi*0.18*t/60)
    noise = 22 * np.random.randn(N)
    rr = trend + noise

    pd.DataFrame({'RR_ms': rr}).to_csv(RESULTS_DIR / "synthetic_rr.csv", index=False)

    results = {
        'sampen': sample_entropy(rr),
        'dfa': detrended_fluctuation_analysis(rr),
        'psd_hrv': compute_psd_hrv_bands(rr),
        'proxy': entanglement_proxy_hrv(rr)
    }

    # Stampa report
    print("\n" + "═"*80)
    print("RISULTATI HRV AVANZATI TET–CVTL")
    print("═"*80)
    print(f"SampEn:          {results['sampen']['sampen']:.4f}")
    print(f"DFA α:           {results['dfa']['alpha']:.4f} (R² = {results['dfa']['r_squared']:.4f})")
    print(f"HF power norm:   {results['psd_hrv']['hf_norm']:.4f}")
    print(f"LF/HF ratio:     {results['psd_hrv']['lf_hf_ratio']:.4f}")
    print(f"Proxy witness:   {results['proxy']['proxy_value']:.4f}")

    # Plot PSD
    plt.figure(figsize=(11,5))
    plt.semilogy(results['psd_hrv']['freq'], results['psd_hrv']['psd'], lw=1.2)
    plt.axvspan(0.15, 0.40, alpha=0.18, color='green', label='HF (0.15–0.4 Hz)')
    plt.axvspan(0.04, 0.15, alpha=0.12, color='orange', label='LF')
    plt.xlabel('Frequenza (Hz)')
    plt.ylabel('PSD (ms²/Hz)')
    plt.title('PSD HRV – Bande classiche')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.savefig(RESULTS_DIR / "hrv_psd_advanced.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nRisultati e plot salvati in: {RESULTS_DIR}")