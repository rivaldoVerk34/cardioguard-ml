"""
Architecture CNN-LSTM hybride pour la classification de battements ECG.

Justification scientifique (à expliquer en soutenance) :
- Les couches CNN (Conv1D) extraient des motifs LOCAUX dans le signal :
  la forme caractéristique d'un complexe QRS, la présence ou l'absence
  d'une onde particulière. C'est ce qui distingue par exemple un battement
  normal d'un battement ventriculaire (forme du QRS très différente).
- Les couches LSTM capturent les dépendances TEMPORELLES dans la séquence :
  comment l'amplitude évolue progressivement le long du battement. C'est
  utile notamment pour les fusions (classe F), où la transition entre deux
  morphologies différentes est progressive.
- La littérature (2023-2026) montre que cette combinaison CNN+LSTM dépasse
  systématiquement les modèles utilisant l'une ou l'autre architecture seule,
  sur les tâches de classification de signaux physiologiques temporels.

Architecture, couche par couche :
  Entrée (250 échantillons, 1 canal)
    -> Conv1D (extraction de motifs locaux, niveau 1 : ondes simples)
    -> MaxPooling1D (réduction de dimension, garde l'information essentielle)
    -> Conv1D (motifs locaux, niveau 2 : combinaisons d'ondes -> complexes QRS)
    -> MaxPooling1D
    -> LSTM (capture la dynamique temporelle sur la séquence de motifs extraits)
    -> Dense (couche de décision)
    -> Sortie (5 probabilités, une par classe AAMI : N, S, V, F, Q)
"""
from tensorflow import keras
from tensorflow.keras import layers


def build_cnn_lstm_model(input_length: int, n_classes: int = 5):
    """
    Construit le modèle CNN-LSTM.

    input_length : nombre d'échantillons par battement (250 dans notre cas,
                   cf. WINDOW_LENGTH dans preprocess.py)
    n_classes : nombre de classes de sortie (5 pour le schéma AAMI)
    """
    model = keras.Sequential([
        layers.Input(shape=(input_length, 1)),

        # --- Bloc convolutif 1 : détection de motifs simples (ondes) ---
        # 16 filtres de taille 7 : chaque filtre apprend à détecter un motif
        # local différent (ex: une pente montante caractéristique d'une onde R)
        layers.Conv1D(filters=16, kernel_size=7, activation="relu", padding="same"),
        layers.BatchNormalization(),  # stabilise l'apprentissage, accélère la convergence
        layers.MaxPooling1D(pool_size=2),  # réduit la longueur de séquence de moitié

        # --- Bloc convolutif 2 : détection de motifs plus complexes (complexes QRS) ---
        # Plus de filtres (32) car on combine les motifs simples du bloc 1
        # en motifs plus riches et plus spécifiques
        layers.Conv1D(filters=32, kernel_size=5, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        # --- Bloc LSTM : capture la dynamique temporelle ---
        # Après les convolutions, on a une séquence plus courte mais avec
        # des "motifs" plus riches à chaque pas de temps. Le LSTM apprend
        # comment ces motifs s'enchaînent dans le temps.
        layers.LSTM(units=32, return_sequences=False),
        layers.Dropout(0.3),  # désactive aléatoirement 30% des neurones à
                              # l'entraînement, pour limiter le surapprentissage
                              # (un risque réel avec un dataset MIT-BIH limité)

        # --- Couche de décision ---
        layers.Dense(units=32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(units=n_classes, activation="softmax"),
        # softmax : transforme les sorties en probabilités qui somment à 1,
        # une par classe AAMI
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",  # adapté à des labels entiers (0-4), pas one-hot
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    from preprocess import WINDOW_LENGTH
    model = build_cnn_lstm_model(input_length=WINDOW_LENGTH, n_classes=5)
    model.summary()
