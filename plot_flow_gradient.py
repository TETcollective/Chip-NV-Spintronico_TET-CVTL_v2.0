import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Ellipse, ConnectionPatch
import numpy as np

fig = plt.figure(figsize=(13, 8), dpi=200)
ax = fig.add_subplot(111)

# Gradient sfondo (simulazione flusso entropico)
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
Z = np.exp(-4*Y)  # riduzione entropia verso il basso
im = ax.imshow(Z, extent=[0,1,0,1], origin='lower', cmap='Purples_r', alpha=0.25)
fig.colorbar(im, ax=ax, label=r'$\Delta S_{\rm ent}^{\rm embodied}$ (riduzione negentropica)', shrink=0.6)

# Nodi
nodes = {
    'ent': (0.5, 0.88),
    'torque': (0.5, 0.55),
    'curv': (0.5, 0.22)
}

texts = {
    'ent': r"$\Delta S_{\rm ent}^{\rm embodied} < 0$" + "\n" + r"(RENASCENT-Q + $\beta$)",
    'torque': r"$\tau_\phi(t)$" + "\n" + r"torque fenomenologico",
    'curv': r"$\mathcal{R}_{\rm info}(t) = \tau_\phi / \hbar$" + "\n" + r"curvatura informativa (qualia)"
}

for key, (x,y) in nodes.items():
    ax.add_patch(plt.Circle((x,y), 0.12, color='white', ec='black', lw=1.5, zorder=3))
    ax.text(x, y, texts[key], ha='center', va='center', fontsize=10, fontweight='bold', zorder=4)

# Frecce multiple con gradient
for dy in [-0.02, 0, 0.02]:
    ax.add_patch(FancyArrowPatch((0.5, 0.76+dy), (0.5, 0.67+dy),
                                 arrowstyle='->', mutation_scale=25,
                                 color='purple', lw=2.2, alpha=0.9, zorder=2))
    ax.add_patch(FancyArrowPatch((0.5, 0.43+dy), (0.5, 0.34+dy),
                                 arrowstyle='->', mutation_scale=25,
                                 color='purple', lw=2.2, alpha=0.9, zorder=2))

# Ciclo retrocausale
ellipse = Ellipse((0.5, 0.55), width=0.85, height=0.75, fill=False, linestyle='--', lw=3, color='magenta', alpha=0.7)
ax.add_patch(ellipse)

# Frecce curve ciclo
ax.add_patch(FancyArrowPatch((0.9, 0.22), (0.9, 0.88), connectionstyle="arc3,rad=0.4",
                             arrowstyle='->', color='magenta', lw=2.5, linestyle='--'))
ax.add_patch(FancyArrowPatch((0.1, 0.88), (0.1, 0.22), connectionstyle="arc3,rad=-0.4",
                             arrowstyle='->', color='magenta', lw=2.5, linestyle='--'))

ax.text(0.5, 1.02, "Ciclo retrocausale embodied (negoziazione time-symmetric)", ha='center', fontsize=12, fontweight='bold', color='magenta')
ax.text(0.5, -0.05, r"$\Delta S < 0 \to \tau_\phi \to \mathcal{R}_{\rm info} \to$ feedback attivo", ha='center', fontsize=10, color='purple')

ax.set_xlim(0,1)
ax.set_ylim(-0.1,1.1)
ax.axis('off')

plt.savefig("figures/flusso_entropico_torque_curvatura_gradient.png", dpi=400, bbox_inches='tight')
plt.savefig("figures/flusso_entropico_torque_curvatura_gradient.pdf", bbox_inches='tight')
plt.close()

print("Plot gradient salvato.")