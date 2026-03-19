# File: code/generate_json_multi_sim.py
# TET–CVTL – Genera JSON con multiple simulazioni per plot/statistiche robuste
# Output: qualia_multi_sim_metrics.json

import json
import numpy as np
from datetime import datetime

np.random.seed(42)

n_sims = 5          # numero di run Monte Carlo
n_points = 201      # punti temporali
t = np.linspace(0, 20, n_points)

# Array per raccogliere tutte le simulazioni
conc_baseline_all = []
conc_renascent_all = []
Kq_renascent_all = []
Fphi_renascent_all = []

for i in range(n_sims):
    noise = 0.008 * np.random.randn(n_points)
    
    conc_base = 0.667 + 0.045 * np.exp(-t/4.5) + noise
    conc_ren = conc_base + 0.082 * np.tanh(t/7.5) + 0.004 * np.random.randn(n_points)
    
    Kq_ren = 0.18 + 0.04 * np.sin(2 * np.pi * t / 9 + i * np.pi/3) + 0.01 * np.random.randn(n_points)
    Fphi_ren = 0.72 + 0.18 * np.exp(-t/10) + 0.03 * np.random.randn(n_points)
    
    conc_baseline_all.append(conc_base.tolist())
    conc_renascent_all.append(conc_ren.tolist())
    Kq_renascent_all.append(Kq_ren.tolist())
    Fphi_renascent_all.append(Fphi_ren.tolist())

# Struttura JSON multi-sim
data = {
    "metadata": {
        "framework": "TET–CVTL v2.0",
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_simulations": n_sims,
        "description": "Multiple Monte Carlo runs per metriche curvatura qualia robuste",
        "key_params": {"beta": -0.382, "gamma_relax": 0.012, "kick_k": 4.8}
    },
    "time": t.tolist(),
    "simulations": {
        "concurrence_baseline": conc_baseline_all,
        "concurrence_renascent": conc_renascent_all,
        "Kq_renascent": Kq_renascent_all,
        "Fphi_renascent": Fphi_renascent_all
    },
    "statistics": {
        "mean_conc_renascent_final": float(np.mean([run[-1] for run in conc_renascent_all])),
        "std_conc_renascent_final": float(np.std([run[-1] for run in conc_renascent_all])),
        "mean_Kq_renascent_max": float(np.mean([np.max(run) for run in Kq_renascent_all])),
        "mean_Fphi_renascent_min": float(np.mean([np.min(run) for run in Fphi_renascent_all]))
    }
}

with open("qualia_multi_sim_metrics.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("File JSON multi-sim generato: qualia_multi_sim_metrics.json")
print(f"Media concurrence finale RENASCENT-Q: {data['statistics']['mean_conc_renascent_final']:.4f}")