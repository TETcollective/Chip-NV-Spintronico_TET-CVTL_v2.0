#!/usr/bin/env python3
"""
embodied_network_advanced.py

Generazione e analisi dettagliata network embodied gerarchico:
- Livelli + sub-nodi (Trp cluster, HRV metrics)
- Pesi entropici layer & cross
- Calcolo S_ent gerarchica
- Visualizzazione con colormap entropia

Autore: Simon Soliman / TET Collective
Data: 15 Marzo 2026
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results/network")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_embodied_graph() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()

    # Micro level + sub-nodi
    micro_nodes = [
        ("Trp_mega",   dict(level=1, weight=0.42, subtype="superradiance")),
        ("MT_lattice", dict(level=1, weight=0.38, subtype="vibration")),
        ("Exciton_flow", dict(level=1, weight=0.20, subtype="transport")),
    ]

    # Meso level + HRV sub-nodi
    meso_nodes = [
        ("Vagus",      dict(level=2, weight=0.28, subtype="autonomic")),
        ("Heart_EM",   dict(level=2, weight=0.24, subtype="cardiac_field")),
        ("SampEn",     dict(level=2, weight=0.13, subtype="complexity")),
        ("DFA_alpha",  dict(level=2, weight=0.13, subtype="fractal")),
        ("HF_power",   dict(level=2, weight=0.12, subtype="vagal")),
    ]

    # Macro level
    macro_nodes = [
        ("Peripheral_receptors", dict(level=3, weight=0.18, subtype="sensory")),
        ("Environment_loop",     dict(level=3, weight=0.10, subtype="feedback")),
    ]

    G.add_nodes_from(micro_nodes + meso_nodes + macro_nodes)

    # Edges con pesi e tipi
    G.add_edges_from([
        ("Trp_mega", "MT_lattice", {"weight": 0.95, "type": "superradiance"}),
        ("MT_lattice", "Exciton_flow", {"weight": 0.82, "type": "exciton"}),
        ("MT_lattice", "Vagus", {"weight": 0.70, "type": "entanglement"}),
        ("MT_lattice", "Heart_EM", {"weight": 0.65, "type": "piezo"}),
        ("Trp_mega", "Heart_EM", {"weight": 0.58, "type": "EM_coupling"}),
        ("Vagus", "Heart_EM", {"weight": 0.88, "type": "vagal"}),
        ("Heart_EM", "SampEn", {"weight": 0.92, "type": "complexity"}),
        ("Heart_EM", "DFA_alpha", {"weight": 0.85, "type": "scaling"}),
        ("Heart_EM", "HF_power", {"weight": 0.90, "type": "HF"}),
        ("Vagus", "Peripheral_receptors", {"weight": 0.68, "type": "feedback"}),
        ("Heart_EM", "Peripheral_receptors", {"weight": 0.62, "type": "coherence"}),
        ("SampEn", "Environment_loop", {"weight": 0.48, "type": "presentiment"}),
    ])

    return G


def compute_hierarchical_entropy(G: nx.MultiDiGraph) -> Dict:
    level_S = {1: 0.0, 2: 0.0, 3: 0.0}
    node_S = {}
    cross_S = 0.0

    for n in G.nodes:
        lvl = G.nodes[n]['level']
        w = G.nodes[n]['weight']
        deg = G.degree(n)
        S_node = w * np.log(deg + np.e)  # log naturale + e per stabilità
        node_S[n] = S_node
        level_S[lvl] += S_node

    for u, v, d in G.edges(data=True):
        if G.nodes[u]['level'] != G.nodes[v]['level']:
            cross_S += d['weight'] * 0.5 * (node_S[u] + node_S[v])

    total = sum(level_S.values()) + cross_S

    return {
        'node_entropy': node_S,
        'level_entropy': level_S,
        'cross_entropy': cross_S,
        'total_embodied_entropy': total
    }


if __name__ == "__main__":
    G = build_embodied_graph()
    ent = compute_hierarchical_entropy(G)

    # Salva dati
    pd.DataFrame(ent['node_entropy'].items(), columns=['Node', 'S_node']).to_csv(
        RESULTS_DIR / "node_entropy.csv", index=False)
    pd.DataFrame([ent['level_entropy']]).to_csv(
        RESULTS_DIR / "level_entropy.csv", index=False)

    print("Entropia embodied totale (approx):", ent['total_embodied_entropy'])

    # Plot
    pos = nx.multipartite_layout(G, subset_key="level")
    colors = [ent['node_entropy'][n] for n in G.nodes]
    plt.figure(figsize=(18, 11))
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.plasma_r,
                           node_size=2400, edgecolors='k', linewidths=1.8)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color='slategray', alpha=0.65, arrowsize=18)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma_r, norm=plt.Normalize(min(colors), max(colors)))
    plt.colorbar(sm, label='Contributo entropico locale')
    plt.title("Network Embodied Gerarchico TET–CVTL\n(overlay entanglement entropy per nodo)")
    plt.axis('off')
    plt.savefig(RESULTS_DIR / "embodied_network_detailed.png", dpi=400, bbox_inches='tight')
    plt.close()

    print(f"File salvati in: {RESULTS_DIR}")