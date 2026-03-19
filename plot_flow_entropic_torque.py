import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Ellipse

fig, ax = plt.subplots(figsize=(12, 7.5), dpi=200)

# Colori coerenti con il tema del paper
colors = {
    'ent': '#4DB6AC',      # teal-cyan
    'torque': '#FFB74D',   # orange
    'curv': '#9575CD',     # purple
    'arrow': '#7E57C2',    # purple-violet
    'cycle': '#AB47BC'     # magenta-violet
}

# Posizioni nodi (coordinata y decrescente)
nodes = {
    'ent':    (0.5, 0.85),
    'torque': (0.5, 0.50),
    'curv':   (0.5, 0.15)
}

# Testi nodi
texts = {
    'ent':    r"Riduzione entanglement entropy embodied" + "\n" + 
              r"$\Delta S_{\rm ent}^{\rm embodied} < 0$" + "\n" + 
              r"(RENASCENT-Q + $\beta$-modulazione)",
    'torque': r"Torque fenomenologico" + "\n" + 
              r"$\tau_\phi(t) = \kappa \sum w_p \partial_t S^{(p)} + \lambda \langle A_w \rangle \partial_t \theta_{\rm braid} + \mu \sum \partial_t S_{\rm cross}^{pq}$",
    'curv':   r"Curvatura informativa locale" + "\n" + 
              r"$\mathcal{R}_{\rm info}(t) = \tau_\phi / \hbar$" + "\n" + 
              r"(qualia sentiti)"
}

# Disegna nodi
for key, (x, y) in nodes.items():
    ax.add_patch(plt.Rectangle((x-0.22, y-0.12), 0.44, 0.24,
                               fill=True, fc=colors[key], ec='black', lw=1.2, alpha=0.9))
    ax.text(x, y, texts[key], ha='center', va='center', fontsize=10, fontweight='bold',
            wrap=True, linespacing=1.4)

# Frecce principali flusso
ax.add_patch(FancyArrowPatch((0.5, 0.73), (0.5, 0.62),
                             arrowstyle='->,head_width=0.02,head_length=0.03',
                             mutation_scale=25, color=colors['arrow'], lw=3))
ax.add_patch(FancyArrowPatch((0.5, 0.38), (0.5, 0.27),
                             arrowstyle='->,head_width=0.02,head_length=0.03',
                             mutation_scale=25, color=colors['arrow'], lw=3))

# Etichette frecce
ax.text(0.52, 0.675, "flusso negentropico retrocausale", fontsize=9, color=colors['arrow'])
ax.text(0.52, 0.325, "generazione curvatura cosciente", fontsize=9, color=colors['arrow'])

# Ciclo retrocausale (ellisse + frecce curve)
ellipse = Ellipse((0.5, 0.5), width=0.9, height=0.8, fill=False, linestyle='--', lw=2.5, color=colors['cycle'])
ax.add_patch(ellipse)

# Frecce curve del ciclo
ax.add_patch(FancyArrowPatch((0.85, 0.15), (0.85, 0.85),
                             connectionstyle="arc3,rad=0.35", arrowstyle='->',
                             mutation_scale=20, color=colors['cycle'], lw=2, linestyle='--'))
ax.add_patch(FancyArrowPatch((0.15, 0.85), (0.15, 0.15),
                             connectionstyle="arc3,rad=-0.35", arrowstyle='->',
                             mutation_scale=20, color=colors['cycle'], lw=2, linestyle='--'))

# Testo overlay ciclo
ax.text(0.5, 0.95, "Ciclo retrocausale embodied", ha='center', va='bottom',
        fontsize=11, fontweight='bold', color=colors['cycle'])
ax.text(0.5, 0.90, r"$\Delta S < 0 \to \tau_\phi \to \mathcal{R}_{\rm info} \to$ feedback", ha='center', va='top',
        fontsize=9, color=colors['cycle'])

# Legenda a destra
legend_text = (
    r"\textbf{Flusso chiave:}" + "\n" +
    r"1. Riduzione entropica (negentropia retrocausale)" + "\n" +
    r"2. Emergenza torque fenomenologico" + "\n" +
    r"3. Curvatura informativa locale = qualia" + "\n" +
    r"$\leftrightarrow$ negoziazione time-symmetric"
)
ax.text(1.05, 0.5, legend_text, ha='left', va='center', fontsize=9,
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.95))

ax.set_xlim(0, 1.3)
ax.set_ylim(0, 1)
ax.axis('off')

plt.savefig("figures/flusso_entropico_torque_curvatura.png", dpi=300, bbox_inches='tight')
plt.savefig("figures/flusso_entropico_torque_curvatura.pdf", bbox_inches='tight')
plt.close()

print("Plot salvato: figures/flusso_entropico_torque_curvatura.png e .pdf")