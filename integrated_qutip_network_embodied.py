#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
integrated_qutip_network_embodied.py

Integrazione avanzata QuTiP + NetworkX per TET–CVTL paper:
- Simulazione RENASCENT-Q su sistema multi-qubit (proxy nodi embodied)
- Estrazione metriche quantistiche (concurrence, S_vn, negativity)
- Mappatura dinamica su pesi del network embodied gerarchico
- Aggiornamento entropia embodied in funzione dell'evoluzione quantistica
- Plot temporali + salvataggio dati per figure/tables

Citabile per:
- Modello entanglement embodied + torque fenomenologico
- Collegamento RENASCENT-Q → network embodied → torque gerarchico

Autore: Simon Soliman / TET Collective
Data: 16 Marzo 2026
Versione: 1.0 – integrazione full per paper v2.0
"""

import numpy as np
import qutip as qt
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List

RESULTS_DIR = Path("results/integrated_qutip_network")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# PARAMETRI GLOBALI
# ──────────────────────────────────────────────────────────────────────────────

N_QUBITS      = 3
GAMMA_RELAX   = 0.012
BETA          = ((1 + np.sqrt(5))/2)**(-2)
EPSILON_RETRO = 0.08
A_W           = 2.2
T_MAX         = 120
T_STEPS       = 600
tlist         = np.linspace(0, T_MAX, T_STEPS)
LEVEL_WEIGHTS = {1: 0.55, 2: 0.30, 3: 0.15}

# ──────────────────────────────────────────────────────────────────────────────
# 1. SIMULAZIONE RENASCENT-Q
# ──────────────────────────────────────────────────────────────────────────────

def create_multi_qubit_initial_state(n_qubits=N_QUBITS):
    ghz = (qt.tensor([qt.basis(2,0)]*n_qubits) + qt.tensor([qt.basis(2,1)]*n_qubits)).unit()
    return ghz * ghz.dag()

def run_renascent_simulation():
    rho0 = create_multi_qubit_initial_state()
    
    H_terms = []
    for i in range(N_QUBITS-1):
        ops = [qt.qeye(2)] * N_QUBITS
        ops[i] = qt.sigmax(); ops[i+1] = qt.sigmax()
        H_terms.append(qt.tensor(ops))
    H_int = 0.6 * np.pi * sum(H_terms)
    H = qt.QobjEvo([H_int, [H_int, '0.4 * sin(0.377 * t)']])

    c_ops_std = []
    for k in range(N_QUBITS):
        ops = [qt.qeye(2)] * N_QUBITS
        ops[k] = qt.sigmam()
        c_ops_std.append(np.sqrt(GAMMA_RELAX) * qt.tensor(ops))

    # Simulazione RENASCENT-Q
    proj_f = qt.tensor([qt.basis(2,0)*qt.basis(2,0).dag() for _ in range(N_QUBITS)])
    c_ops_ren = c_ops_std + [np.sqrt(BETA * EPSILON_RETRO) * proj_f]
    res_ren = qt.mesolve(H, rho0, tlist, c_ops=c_ops_ren)

    metrics = {
        't': tlist,
        'C_ren': np.array([qt.concurrence(r.ptrace([0,1])) for r in res_ren.states]),
        'Svn_ren': np.array([qt.entropy_vn(r) for r in res_ren.states]),
        'neg_ren': np.array([qt.negativity(r, subsys=[0,1]) for r in res_ren.states])
    }
    return metrics

def create_dynamic_embodied_network():
    G = nx.MultiDiGraph()
    nodes = [("Trp/MT", dict(level=1, base_weight=0.45)), ("Vagus", dict(level=2, base_weight=0.30)),
             ("Heart/HRV", dict(level=2, base_weight=0.25)), ("Receptors", dict(level=3, base_weight=0.20))]
    G.add_nodes_from(nodes)
    G.add_edges_from([("Trp/MT", "Vagus", {"base_weight": 0.70}), ("Trp/MT", "Heart/HRV", {"base_weight": 0.65}),
                      ("Vagus", "Heart/HRV", {"base_weight": 0.85}), ("Vagus", "Receptors", {"base_weight": 0.68}),
                      ("Heart/HRV", "Receptors", {"base_weight": 0.62})])
    return G

def update_network_weights(G, concurrence_ren, svn_ren, negativity_ren, t_idx):
    boost = np.clip(concurrence_ren * 2.5, 0.5, 1.8) * np.exp(-svn_ren) * np.clip(negativity_ren * 3.0, 0.6, 1.5)
    for u, v, d in G.edges(data=True):
        G[u][v][0]['dynamic_weight'] = d['base_weight'] * boost * (1 + 0.1 * np.sin(0.5 * t_idx))
    return G

def compute_dynamic_entropy(G):
    node_S = {}; level_S = {1: 0.0, 2: 0.0, 3: 0.0}; cross_S = 0.0
    for n in G.nodes:
        lvl = G.nodes[n]['level']
        avg_w = np.mean([d['dynamic_weight'] for _,_,d in G.edges(n, data=True)] or [G.nodes[n]['base_weight']])
        S_node = avg_w * np.log(G.degree(n) + 1.618) * LEVEL_WEIGHTS.get(lvl, 1.0)
        node_S[n] = S_node; level_S[lvl] += S_node
    for u, v, d in G.edges(data=True):
        if G.nodes[u]['level'] != G.nodes[v]['level']:
            cross_S += d['dynamic_weight'] * 0.5 * (node_S[u] + node_S[v])
    return {'total': sum(level_S.values()) + cross_S}

def main_integration():
    q_metrics = run_renascent_simulation()
    G_base = create_dynamic_embodied_network()
    data = []
    for i, t in enumerate(tlist):
        G_dyn = update_network_weights(G_base.copy(), q_metrics['C_ren'][i], q_metrics['Svn_ren'][i], q_metrics['neg_ren'][i], i)
        ent = compute_dynamic_entropy(G_dyn)
        data.append({'t': t, 'C_ren': q_metrics['C_ren'][i], 'total_S': ent['total']})

    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    plt.plot(df['t'], df['C_ren'], label='Quantum Concurrence')
    plt.plot(df['t'], df['total_S']/max(df['total_S']), label='Normalized Embodied Entropy')
    plt.title("Integrazione QuTiP-Network: Feedback RENASCENT-Q")
    plt.legend(); plt.show()

if __name__ == "__main__":
    main_integration()