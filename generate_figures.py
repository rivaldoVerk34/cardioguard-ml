"""
generate_figures.py — Génère les figures propres pour le Chapitre 3, à partir
de comparaison_finale.json (déjà généré par train_comparatif.py).

Usage : python3 generate_figures.py
Produit : figures/confusion_rf.png, figures/confusion_svm.png,
          figures/confusion_cnn.png, figures/comparaison_f1.png
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

with open("comparaison_finale.json") as f:
    data = json.load(f)

def plot_confusion(result, filename):
    cm = np.array(result["confusion_matrix"])
    labels = result["target_names"]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    ax.set_title(f"{result['model']} — Matrice de confusion normalisée\n(F1 macro = {result['f1_macro']:.3f})")

    for i in range(len(labels)):
        for j in range(len(labels)):
            text_color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)",
                    ha="center", va="center", color=text_color, fontsize=8)

    fig.colorbar(im, ax=ax, label="Proportion (normalisée par ligne)")
    plt.tight_layout()
    plt.savefig(f"figures/{filename}", dpi=200)
    plt.close()
    print(f"  -> figures/{filename}")

print("Génération des matrices de confusion...")
for result in data["results"]:
    name_slug = result["model"].lower().replace("-", "_")
    plot_confusion(result, f"confusion_{name_slug}.png")

# Graphique comparatif F1-macro
print("Génération du graphique comparatif F1-macro...")
names = [r["model"] for r in data["results"]]
f1_scores = [r["f1_macro"] for r in data["results"]]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(names, f1_scores, color=["#4C72B0", "#55A868", "#C44E52"])
ax.set_ylabel("F1-score macro")
ax.set_title("Comparaison des trois modèles — critère de sélection ENF-06")
ax.set_ylim(0, 1)
ax.axhline(y=max(f1_scores), color="gray", linestyle="--", linewidth=0.8)
for bar, score in zip(bars, f1_scores):
    ax.text(bar.get_x() + bar.get_width()/2, score + 0.02, f"{score:.3f}",
            ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/comparaison_f1.png", dpi=200)
plt.close()
print("  -> figures/comparaison_f1.png")

print("\nToutes les figures sont dans le dossier figures/ — prêtes à insérer dans le Word.")
