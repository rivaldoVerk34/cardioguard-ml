"""
Script à lancer SUR TA MACHINE (pas dans le sandbox Claude), une fois le
dataset complet MIT-BIH téléchargé depuis PhysioNet.

PRÉREQUIS :
    pip install wfdb tensorflow scikit-learn numpy

ÉTAPE 1 - Télécharger le dataset complet (48 enregistrements, ~100 Mo) :
    python -c "import wfdb; wfdb.dl_database('mitdb', dl_dir='data')"

    Cette commande télécharge automatiquement les 48 enregistrements
    officiels de la MIT-BIH Arrhythmia Database depuis PhysioNet.
    Si la commande échoue, télécharge manuellement depuis :
    https://physionet.org/content/mitdb/1.0.0/

ÉTAPE 2 - Lancer ce script :
    python train_full_dataset.py
"""
import os
from train import prepare_dataset, train_and_evaluate

# Les 48 enregistrements officiels de la MIT-BIH Arrhythmia Database.
# Référence : https://physionet.org/content/mitdb/1.0.0/
ALL_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]


def check_data_available(data_dir: str = "data"):
    """Vérifie que les fichiers du dataset complet sont bien présents."""
    missing = []
    for name in ALL_RECORDS:
        for ext in (".dat", ".hea", ".atr"):
            if not os.path.exists(f"{data_dir}/{name}{ext}"):
                missing.append(f"{name}{ext}")
    return missing


if __name__ == "__main__":
    print("=== Vérification de la présence du dataset complet ===")
    missing = check_data_available()
    if missing:
        print(f"\nATTENTION : {len(missing)} fichier(s) manquant(s), par exemple : {missing[:5]}")
        print("\nTélécharge le dataset complet avec :")
        print('  python -c "import wfdb; wfdb.dl_database(\'mitdb\', dl_dir=\'data\')"')
        raise SystemExit(1)

    print(f"Dataset complet trouvé : {len(ALL_RECORDS)} enregistrements.\n")

    print("=== Chargement et prétraitement de l'ensemble du dataset ===")
    beats, labels, patient_ids = prepare_dataset(ALL_RECORDS)
    print(f"\nTotal : {len(beats)} battements, forme {beats.shape}")

    print("\n=== Entraînement du modèle CNN-LSTM sur le dataset complet ===")
    model, history = train_and_evaluate(beats, labels, patient_ids, epochs=30)

    model.save("cardioguard_cnn_lstm_final.keras")
    print("\nModèle final sauvegardé : cardioguard_cnn_lstm_final.keras")
    print("Ce modèle est celui à présenter dans le mémoire (chapitre Résultats).")
