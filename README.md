# CardioGuard-ML : classification d'arythmies et étude de compression

Partie « apprentissage automatique » de CardioGuard-Bénin, mon mémoire de Master 2 (IFRI, Université d'Abomey-Calavi) : détection d'événements cardiaques sur un système embarqué (ESP32, ECG AD8232, MAX30102).

## Données et protocole

- Base MIT-BIH Arrhythmia Database, classes AAMI à 5 niveaux, 109 452 battements.
- Découpage inter-patients (38 patients d'entraînement, 10 de test, graine 42) : aucun patient du test n'est vu à l'entraînement.
- Le dossier data/ n'est pas versionné : télécharger MIT-BIH sur PhysioNet et le placer dans data/.

## Comparaison de modèles (F1 macro, jeu de test inter-patients)

| Modèle | F1 macro |
|---|---|
| SVM | 0,521 |
| Random Forest | 0,496 |
| CNN-LSTM | 0,442 |

Le CNN-LSTM n'est pas le meilleur ici. Une classe du jeu de test ne compte que 16 battements, ce qui rend le F1 macro bruité. Le modèle évalué est cardioguard_cnn_lstm_final_v2.keras (F1 de 0,441 sur ce test, vérifié par make_test_set.py).

## Étude de compression du CNN-LSTM (48,9 Ko en float32)

Un seul entraînement, un seul découpage : les écarts de l'ordre de 0,01 sont à considérer comme du bruit.

| Variante | Taille (Ko) | F1 macro |
|---|---|---|
| float32 (référence) | 48,9 | 0,441 |
| Poids int8 (simulé) | 17,3 | 0,443 |
| Élagage 50 %, sans réentraînement | 30,8 | 0,162 |
| Élagage progressif 70 % + réentraînement + int8 | 12,3 | 0,439 |
| Élagage progressif 80 % + réentraînement + int8 | 10,9 | 0,331 |

Factorisation SVD, sans réentraînement :

| Variante | Taille (Ko) | F1 macro |
|---|---|---|
| int8 seul | 17,7 | 0,441 |
| Rang 50 % + int8 | 15,7 | 0,442 |
| Rang 35 % + int8 | 13,8 | 0,382 |
| Rang 25 % + int8 | 12,4 | 0,290 |

Les tailles sont celles des poids sérialisés puis compressés (npz). La référence int8 vaut 17,3 Ko ou 17,7 Ko selon le script, car les petits tenseurs sont sérialisés différemment.

## Limites

- Quantification simulée (poids quantifiés puis reconvertis) : pas de noyaux entiers. Les latences mesurées sur Mac ne valent pas celles de l'ESP32.
- Un seul tirage : la variance entre graines n'est pas mesurée.
- La SVD n'est pas suivie d'un réentraînement : l'effet d'un entraînement orienté vers le rang faible n'est pas testé.
- La conversion TFLite du CNN-LSTM échoue (TensorFlow 2.16, couche BatchNorm) ; elle n'est pas utilisée.

## Scripts

- make_test_set.py : reconstruit le jeu de test inter-patients et vérifie le F1 du modèle (attendu : environ 0,441).
- compression_v2.py : quantification et élagage sans réentraînement.
- pruning_finetune.py : élagage progressif avec réentraînement.
- low_rank_quant.py : factorisation SVD puis int8.
- train_comparatif.py : comparaison Random Forest, SVM et CNN-LSTM.

Exemple :

    python make_test_set.py cardioguard_cnn_lstm_final_v2.keras
    python compression_v2.py cardioguard_cnn_lstm_final_v2.keras X_test.npy y_test.npy

Auteur : Rivaldo Verckys Senonkin, github.com/rivaldoVerk34
