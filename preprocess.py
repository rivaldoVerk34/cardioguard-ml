"""
Prétraitement des signaux ECG MIT-BIH pour l'entraînement du modèle CNN-LSTM.

Étapes (à expliquer en soutenance) :
1. Lecture du signal brut + annotations (positions des pics R, type de battement)
2. Découpage en fenêtres centrées sur chaque pic R (un battement = un échantillon)
3. Filtrage des annotations qui ne sont pas de vrais battements (marqueurs techniques)
4. Mapping des ~15 types MIT-BIH vers les 5 classes AAMI
5. Normalisation de chaque fenêtre (amplitude variable d'un patient à l'autre)

Pourquoi centrer sur le pic R (justification scientifique) :
Le pic R est le point le plus reconnaissable et le plus énergétique du
complexe QRS. Centrer chaque fenêtre sur le pic R garantit que le réseau
voit toujours le battement dans la même position relative, ce qui facilite
l'apprentissage (le CNN n'a pas à "chercher" le battement dans la fenêtre).
"""
import numpy as np
import wfdb

from aami_mapping import map_symbol_to_aami, AAMI_CLASS_TO_INDEX

# Fenêtre autour de chaque pic R, en nombre d'échantillons.
# À 360 Hz, 0.25s avant + 0.45s après le pic R couvre un battement complet
# (complexe QRS + une partie de l'onde T) sans empiéter excessivement sur
# les battements voisins. Ces valeurs sont celles communément retenues dans
# la littérature sur la classification de battements MIT-BIH.
SAMPLES_BEFORE_R = 90   # 0.25s à 360 Hz
SAMPLES_AFTER_R = 160   # ~0.44s à 360 Hz
WINDOW_LENGTH = SAMPLES_BEFORE_R + SAMPLES_AFTER_R  # 250 échantillons


def load_record(record_name: str, data_dir: str = "data"):
    """Charge un enregistrement MIT-BIH (signal + annotations) depuis le disque local."""
    path = f"{data_dir}/{record_name}"
    record = wfdb.rdrecord(path)
    annotation = wfdb.rdann(path, "atr")
    return record, annotation


def extract_beats(record, annotation, channel: int = 0):
    """
    Découpe le signal en fenêtres centrées sur chaque pic R annoté.

    Retourne :
        beats: array de forme (n_beats, WINDOW_LENGTH) - les segments de signal
        labels: array de forme (n_beats,) - l'indice de classe AAMI (0-4)
        skipped: nombre de battements ignorés (trop proches du bord, ou
                 symbole non reconnu/non-battement)
    """
    signal = record.p_signal[:, channel]  # on utilise le canal MLII (dérivation principale)
    beats = []
    labels = []
    skipped = 0

    for sample_idx, symbol in zip(annotation.sample, annotation.symbol):
        aami_class = map_symbol_to_aami(symbol)
        if aami_class is None:
            skipped += 1
            continue

        start = sample_idx - SAMPLES_BEFORE_R
        end = sample_idx + SAMPLES_AFTER_R

        # On ignore les battements trop proches du début/fin de l'enregistrement,
        # pour lesquels la fenêtre complète n'est pas disponible.
        if start < 0 or end > len(signal):
            skipped += 1
            continue

        window = signal[start:end]
        beats.append(window)
        labels.append(AAMI_CLASS_TO_INDEX[aami_class])

    return np.array(beats), np.array(labels), skipped


def normalize_beats(beats: np.ndarray):
    """
    Normalisation par battement (z-score individuel).

    Pourquoi par battement et pas globalement sur tout l'enregistrement :
    l'amplitude du signal ECG varie selon le patient, l'électrode, et même
    au cours du temps pour un même patient (mouvement, transpiration...).
    Normaliser chaque battement individuellement rend le modèle robuste à
    ces variations d'amplitude, qui n'ont pas de sens diagnostique en elles-mêmes.
    """
    mean = beats.mean(axis=1, keepdims=True)
    std = beats.std(axis=1, keepdims=True)
    std[std == 0] = 1.0  # évite la division par zéro sur un signal plat
    return (beats - mean) / std


if __name__ == "__main__":
    record, annotation = load_record("100")
    beats, labels, skipped = extract_beats(record, annotation)
    beats_normalized = normalize_beats(beats)

    print(f"Battements extraits : {len(beats)}")
    print(f"Battements ignorés (marqueurs/bords) : {skipped}")
    print(f"Forme des données : {beats_normalized.shape}")

    unique, counts = np.unique(labels, return_counts=True)
    from aami_mapping import AAMI_CLASSES
    print("\nDistribution des classes AAMI :")
    for idx, count in zip(unique, counts):
        print(f"  {AAMI_CLASSES[idx]} : {count} battements")
