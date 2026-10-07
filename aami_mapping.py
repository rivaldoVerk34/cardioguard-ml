"""
Mapping des symboles d'annotation MIT-BIH vers les 5 super-classes AAMI.

Référence : AAMI EC57 (Association for the Advancement of Medical
Instrumentation), standard largement utilisé dans la littérature pour
regrouper les ~15 types de battements annotés dans MIT-BIH en 5 catégories
cliniquement significatives.

Pourquoi ce regroupement (à expliquer en soutenance) :
- MIT-BIH a des classes très déséquilibrées (certains types n'ont que
  quelques dizaines d'exemples sur 100 000+ battements).
- Le regroupement AAMI rend l'entraînement plus robuste et reste
  cliniquement pertinent (différencier V de S a un vrai sens médical).
- C'est la norme académique : nos résultats deviennent directement
  comparables à ceux d'autres travaux publiés sur MIT-BIH.
"""

# N - Normal
# S - Supraventriculaire (origine au-dessus des ventricules)
# V - Ventriculaire (origine dans les ventricules, potentiellement plus grave)
# F - Fusion (fusion entre un battement normal et un battement ventriculaire)
# Q - Inclassable / non analysable
AAMI_MAPPING = {
    # --- Classe N (Normal) ---
    "N": "N",  # Battement normal
    "L": "N",  # Bloc de branche gauche
    "R": "N",  # Bloc de branche droit
    "e": "N",  # Échappement atrial
    "j": "N",  # Échappement jonctionnel

    # --- Classe S (Supraventriculaire) ---
    "A": "S",  # Extrasystole auriculaire (atrial premature beat)
    "a": "S",  # Extrasystole auriculaire aberrante
    "J": "S",  # Extrasystole jonctionnelle
    "S": "S",  # Extrasystole supraventriculaire (terme générique)

    # --- Classe V (Ventriculaire) ---
    "V": "V",  # Extrasystole ventriculaire (premature ventricular contraction)
    "E": "V",  # Échappement ventriculaire

    # --- Classe F (Fusion) ---
    "F": "F",  # Fusion entre battement normal et ventriculaire

    # --- Classe Q (Inclassable) ---
    "/": "Q",  # Battement de pacemaker (paced beat)
    "f": "Q",  # Fusion entre battement de pacemaker et normal
    "Q": "Q",  # Battement non classifiable
}

# Symboles présents dans les fichiers .atr qui NE SONT PAS des battements
# (marqueurs techniques à ignorer lors du découpage en battements)
NON_BEAT_SYMBOLS = {
    "+",  # changement de rythme (marqueur de rythme, pas un battement)
    "~",  # bruit/qualité du signal
    "|",  # battement isolé non classifié par l'algorithme
    "x",  # battement non conduit (annotation de marqueur)
    '"',  # commentaire
    "[", "]",  # début/fin de fibrillation ventriculaire
    "!",  # flutter/fibrillation ventriculaire
}

AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
AAMI_CLASS_TO_INDEX = {c: i for i, c in enumerate(AAMI_CLASSES)}


def map_symbol_to_aami(symbol: str):
    """
    Convertit un symbole d'annotation MIT-BIH en classe AAMI.
    Retourne None si le symbole n'est pas un battement (marqueur technique)
    ou s'il n'est pas reconnu (cas rare, à logger pour vérification).
    """
    if symbol in NON_BEAT_SYMBOLS:
        return None
    return AAMI_MAPPING.get(symbol)  # None si symbole inconnu
