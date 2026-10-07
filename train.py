"""
CORRECTIF — découpage inter-patient au lieu du split aléatoire par battement.

À appliquer dans train.py : remplacer la fonction prepare_dataset et la
partie split de train_and_evaluate par ce qui suit, pour que le code
corresponde exactement à ce qu'annonce le mémoire en 1.3.3 et 2.5.2.
"""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from preprocess import load_record, extract_beats, normalize_beats, WINDOW_LENGTH
from aami_mapping import AAMI_CLASSES
from model import build_cnn_lstm_model


def prepare_dataset(record_names: list, data_dir: str = "data"):
    """
    Charge et prétraite un ou plusieurs enregistrements, puis les concatène.
    Retourne désormais AUSSI un tableau patient_ids, indispensable pour le
    découpage inter-patient (cf. 1.3.3 du mémoire).
    """
    all_beats, all_labels, all_patient_ids = [], [], []
    for name in record_names:
        record, annotation = load_record(name, data_dir=data_dir)
        beats, labels, skipped = extract_beats(record, annotation)
        print(f"  Enregistrement {name} : {len(beats)} battements extraits, {skipped} ignorés")
        all_beats.append(beats)
        all_labels.append(labels)
        all_patient_ids.append(np.full(len(beats), name))  # ID = numéro d'enregistrement

    beats = np.concatenate(all_beats, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    patient_ids = np.concatenate(all_patient_ids, axis=0)
    beats = normalize_beats(beats)
    return beats, labels, patient_ids


def train_and_evaluate(beats, labels, patient_ids, epochs: int = 15, batch_size: int = 32):
    unique, counts = np.unique(labels, return_counts=True)
    rare_classes = unique[counts < 2]
    if len(rare_classes) > 0:
        rare_names = [AAMI_CLASSES[c] for c in rare_classes]
        print(f"\nAttention : classe(s) {rare_names} trop rare(s) -> exclue(s).")
        mask = ~np.isin(labels, rare_classes)
        beats, labels, patient_ids = beats[mask], labels[mask], patient_ids[mask]

    # DÉCOUPAGE INTER-PATIENT : aucun patient présent à la fois dans train et test.
    # C'est ce que GroupShuffleSplit garantit, contrairement à train_test_split classique.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(beats, labels, groups=patient_ids))

    X_train, X_test = beats[train_idx], beats[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    print(f"\nPatients en entraînement : {sorted(set(patient_ids[train_idx]))}")
    print(f"Patients en test         : {sorted(set(patient_ids[test_idx]))}")
    overlap = set(patient_ids[train_idx]) & set(patient_ids[test_idx])
    print(f"Chevauchement (doit être vide) : {overlap}")

    X_train = X_train.reshape(-1, WINDOW_LENGTH, 1)
    X_test = X_test.reshape(-1, WINDOW_LENGTH, 1)

    present_classes = np.unique(y_train)
    class_weights_array = compute_class_weight(
        class_weight="balanced", classes=present_classes, y=y_train
    )
    class_weight = dict(zip(present_classes, class_weights_array))
    print("\nPoids de classe appliqués :")
    for idx, weight in class_weight.items():
        print(f"  {AAMI_CLASSES[idx]} : {weight:.2f}")

    model = build_cnn_lstm_model(input_length=WINDOW_LENGTH, n_classes=5)

    early_stop = __import__("tensorflow").keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=2,
    )

    print("\n--- Évaluation sur l'ensemble de test (patients jamais vus en entraînement) ---")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    present_in_test = sorted(set(y_test) | set(y_pred))
    target_names = [AAMI_CLASSES[i] for i in present_in_test]

    report = classification_report(y_test, y_pred, labels=present_in_test,
                                    target_names=target_names, zero_division=0,
                                    output_dict=True)
    print(classification_report(y_test, y_pred, labels=present_in_test,
                                 target_names=target_names, zero_division=0))
    print("Matrice de confusion :")
    print("Classes:", target_names)
    cm = confusion_matrix(y_test, y_pred, labels=present_in_test)
    print(cm)

    # SAUVEGARDE DES RÉSULTATS — pour ne plus jamais perdre un entraînement (cf. ta question précédente)
    import json
    results = {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "target_names": target_names,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "patients_train": sorted(set(patient_ids[train_idx])),
        "patients_test": sorted(set(patient_ids[test_idx])),
    }
    with open("training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nRésultats sauvegardés dans training_results.json")

    return model, history
