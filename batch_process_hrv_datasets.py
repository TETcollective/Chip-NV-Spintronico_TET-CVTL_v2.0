#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_process_hrv_datasets.py

Script batch per processare multipli dataset HRV (intervalli RR in ms):
- Input: tutti i file .csv / .tsv / .txt nella cartella data/hrv_datasets/
- Output per ogni file: results/hrv_batch/{filename}_metrics.csv + _report.txt
- Report globale: results/hrv_batch/summary_hrv_batch.csv + comparative plots
- Metriche calcolate: SampEn, DFA α, PSD bande (HF/LF/VLF), proxy entanglement witness

Utilizzo nel paper TET–CVTL:
- Validazione falsificabilità HRV sotto stimoli retrocausali/meditativi
- Confronto baseline vs post-NV-enhanced / meditazione

Autore: Simon Soliman / TET Collective
Data: 16 Marzo 2026
Versione: 1.0 – batch processing per paper submission
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Dict
import warnings

from hrv_advanced_metrics import (  # importa dal file precedente
    sample_entropy,
    detrended_fluctuation_analysis,
    compute_psd_hrv_bands,
    entanglement_proxy_hrv
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAZIONE
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR      = Path("data/hrv_datasets")
RESULTS_BASE  = Path("results/hrv_batch")
RESULTS_BASE.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXT = {'.csv', '.tsv', '.txt'}
RR_COLUMN     = 'RR_ms'           # nome colonna intervalli RR (modifica se diverso)

# Parametri metriche
SAMPEN_M         = 2
SAMPEN_R_FACTOR  = 0.2
DFA_POLY_ORDER   = 2
FS_RESAMPLE      = 4.0

# ──────────────────────────────────────────────────────────────────────────────
# FUNZIONI DI SUPPORTO
# ──────────────────────────────────────────────────────────────────────────────

def read_hrv_file(filepath: Path) -> np.ndarray:
    """ Legge serie RR da file CSV/TSV/TXT """
    ext = filepath.suffix.lower()
    try:
        if ext == '.csv':
            df = pd.read_csv(filepath)
        elif ext == '.tsv':
            df = pd.read_csv(filepath, sep='\t')
        else:  # .txt
            df = pd.read_csv(filepath, sep='\s+', header=None, names=[RR_COLUMN])

        if RR_COLUMN not in df.columns:
            raise ValueError(f"Colonna '{RR_COLUMN}' non trovata")

        rr = df[RR_COLUMN].dropna().values.astype(float)
        if len(rr) < 300:
            raise ValueError(f"Serie troppo corta ({len(rr)} campioni)")

        return rr

    except Exception as e:
        print(f"Errore lettura {filepath.name}: {e}")
        return None


def process_single_dataset(filepath: Path) -> Dict:
    """ Processa un singolo file HRV e restituisce tutte le metriche """
    rr = read_hrv_file(filepath)
    if rr is None:
        return {'filename': filepath.name, 'error': 'lettura fallita'}

    results = {
        'filename': filepath.name,
        'length': len(rr),
        'mean_rr': np.mean(rr),
        'std_rr': np.std(rr),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        # SampEn
        se_dict = sample_entropy(rr, m=SAMPEN_M, r_factor=SAMPEN_R_FACTOR)
        results['sampen'] = se_dict.get('sampen', np.nan)

        # DFA
        dfa_dict = detrended_fluctuation_analysis(rr, poly_order=DFA_POLY_ORDER)
        results['dfa_alpha'] = dfa_dict.get('alpha', np.nan)
        results['dfa_r2']    = dfa_dict.get('r_squared', np.nan)

        # PSD HRV bands
        psd_dict = compute_psd_hrv_bands(rr, fs_resample=FS_RESAMPLE)
        results['hf_norm']      = psd_dict.get('hf_norm', np.nan)
        results['lf_hf_ratio']  = psd_dict.get('lf_hf_ratio', np.nan)
        results['vlf_norm']     = psd_dict.get('vlf_norm', np.nan)

        # Proxy entanglement witness
        proxy_dict = entanglement_proxy_hrv(rr)
        results['proxy_value'] = proxy_dict.get('proxy_value', np.nan)

    except Exception as e:
        results['processing_error'] = str(e)

    # Salva report individuale
    individual_csv = RESULTS_BASE / f"{filepath.stem}_metrics.csv"
    pd.DataFrame([results]).to_csv(individual_csv, index=False)

    # Report testuale
    with open(RESULTS_BASE / f"{filepath.stem}_report.txt", "w") as f:
        f.write(f"Report HRV TET–CVTL – {filepath.name}\n")
        f.write("="*60 + "\n")
        for k, v in results.items():
            f.write(f"{k:18}: {v}\n")
        f.write("\n")

    return results


def generate_summary_report(all_results: List[Dict]):
    """ Crea report globale e plot comparativi """
    df_summary = pd.DataFrame(all_results)
    df_summary = df_summary.sort_values('filename')

    # Salva summary CSV
    df_summary.to_csv(RESULTS_BASE / "summary_hrv_batch.csv", index=False)

    # Plot comparativi
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False)

    # SampEn
    axes[0,0].bar(df_summary['filename'], df_summary['sampen'], color='teal')
    axes[0,0].set_title('Sample Entropy per dataset')
    axes[0,0].set_ylabel('SampEn')
    axes[0,0].tick_params(axis='x', rotation=45)

    # DFA α
    axes[0,1].bar(df_summary['filename'], df_summary['dfa_alpha'], color='coral')
    axes[0,1].set_title('DFA α per dataset')
    axes[0,1].axhline(1.0, color='k', ls='--', alpha=0.6, label='α=1 (1/f)')
    axes[0,1].legend()
    axes[0,1].tick_params(axis='x', rotation=45)

    # HF norm
    axes[1,0].bar(df_summary['filename'], df_summary['hf_norm'], color='purple')
    axes[1,0].set_title('Normalized HF power')
    axes[1,0].set_ylabel('HF norm')
    axes[1,0].tick_params(axis='x', rotation=45)

    # Proxy witness
    axes[1,1].bar(df_summary['filename'], df_summary['proxy_value'], color='darkgreen')
    axes[1,1].set_title('Proxy Entanglement Witness')
    axes[1,1].axhline(0.6, color='r', ls='--', alpha=0.5, label='threshold alta coerenza')
    axes[1,1].legend()
    axes[1,1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "hrv_batch_comparative_plots.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("\nReport globale generato:")
    print(f"  - {RESULTS_BASE / 'summary_hrv_batch.csv'}")
    print(f"  - {RESULTS_BASE / 'hrv_batch_comparative_plots.png'}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN BATCH
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("Batch processing HRV datasets TET–CVTL")
    print(f"Cartella input:  {DATA_DIR}")
    print(f"Cartella output: {RESULTS_BASE}\n")

    all_files = []
    for ext in SUPPORTED_EXT:
        all_files.extend(DATA_DIR.rglob(f"*{ext}"))

    if not all_files:
        print("Nessun file trovato in data/hrv_datasets/")
        return

    print(f"Trovati {len(all_files)} file da processare\n")

    all_results = []
    for idx, filepath in enumerate(all_files, 1):
        print(f"[{idx}/{len(all_files)}] Processando: {filepath.name}")
        res = process_single_dataset(filepath)
        all_results.append(res)

    print("\nGenerazione report globale...")
    generate_summary_report(all_results)

    print("\nBatch completato con successo.")


if __name__ == "__main__":
    main()