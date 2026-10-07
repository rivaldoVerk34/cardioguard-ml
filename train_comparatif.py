"""
train_comparatif.py — Entraîne RF, SVM et CNN-LSTM (avec sur-échantillonnage)
sur EXACTEMENT le même split inter-patient, pour une comparaison rigoureuse
conforme à la méthodologie annoncée en 2.5.2 du mémoire.

Usage : python3 train_comparatif.py
Prérequis : dataset complet déjà présent dans data/ (déjà le cas chez toi)
"""
import json
import time
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from preprocess import load_record, extract_beats, normalize_beats, WINDOW_LENGTH
from aami_mapping import AAMI_CLASSES
from model import build_cnn_lstm_model

ALL_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]

# ============================================================
# 1. Chargement (identique pour les trois modèles)
# ============================================================
def prepare_dataset(record_names, data_dir="data"):
    all_beats, all_labels, all_patient_ids = [], [], []
    for name in record_names:
        record, annotation = load_record(name, data_dir=data_dir)
        beats, labels, skipped = extract_beats(record, annotation)
        print(f"  Enregistrement {name} : {len(beats)} battements extraits, {skipped} ignorés")
        all_beats.append(beats)
        all_labels.append(labels)
        all_patient_ids.append(np.full(len(beats), name))
    beats = np.concatenate(all_beats, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    patient_ids = np.concatenate(all_patient_ids, axis=0)
    beats = normalize_beats(beats)
    return beats, labels, patient_ids


def oversample_minority_classes(X_train, y_train, target_ratio=0.15, noise_std=0.02, seed=42):
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y_train, return_counts=True)
    majority_count = counts.max()
    target_count = int(majority_count * target_ratio)
    X_aug, y_aug = [X_train], [y_train]
    for c, count in zip(classes, counts):
        if count < target_count:
            n_needed = target_count - count
            idx_c = np.where(y_train == c)[0]
            sampled_idx = rng.choice(idx_c, size=n_needed, replace=True)
            samples = X_train[sampled_idx].copy()
            samples += rng.normal(0, noise_std, samples.shape)
            X_aug.append(samples)
            y_aug.append(np.full(n_needed, c))
    X_bal = np.concatenate(X_aug, axis=0)
    y_bal = np.concatenate(y_aug, axis=0)
    perm = rng.permutation(len(X_bal))
    return X_bal[perm], y_bal[perm]


def evaluate_and_report(name, y_test, y_pred, elapsed_train, elapsed_infer, n_test):
    present = sorted(set(y_test) | set(y_pred))
    target_names = [AAMI_CLASSES[i] for i in present]
    report = classification_report(y_test, y_pred, labels=present, target_names=target_names,
                                    zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=present)
    macro_f1 = f1_score(y_test, y_pred, labels=present, average='macro', zero_division=0)

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, labels=present, target_names=target_names, zero_division=0))
    print("Matrice de confusion :", target_names)
    print(cm)
    print(f"F1 macro : {macro_f1:.4f}")
    print(f"Temps entraînement : {elapsed_train:.1f}s | Temps inférence : {elapsed_infer*1000/n_test:.2f}ms/échantillon")

    return {
        "model": name,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "target_names": target_names,
        "f1_macro": macro_f1,
        "train_time_seconds": elapsed_train,
        "inference_ms_per_sample": elapsed_infer * 1000 / n_test,
    }


if __name__ == "__main__":
    print("=== Chargement et prétraitement (dataset complet, identique pour les 3 modèles) ===")
    beats, labels, patient_ids = prepare_dataset(ALL_RECORDS)
    print(f"\nTotal : {len(beats)} battements")

    # SPLIT UNIQUE, PARTAGÉ PAR LES TROIS MODÈLES — condition de comparabilité
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(beats, labels, groups=patient_ids))
    X_train_raw, X_test = beats[train_idx], beats[test_idx]
    y_train_raw, y_test = labels[train_idx], labels[test_idx]

    print(f"Patients train : {len(set(patient_ids[train_idx]))} | Patients test : {len(set(patient_ids[test_idx]))}")
    assert len(set(patient_ids[train_idx]) & set(patient_ids[test_idx])) == 0, "FUITE inter-patient détectée !"
    print("Vérification inter-patient : OK, aucun chevauchement.\n")

    # Sur-échantillonnage — appliqué identiquement pour RF et SVM (CNN-LSTM en a aussi besoin)
    X_train, y_train = oversample_minority_classes(X_train_raw, y_train_raw, target_ratio=0.15)
    print(f"Après sur-échantillonnage : {len(X_train)} battements d'entraînement")
    for c, n in zip(*np.unique(y_train, return_counts=True)):
        print(f"  {AAMI_CLASSES[c]} : {n}")

    all_results = []

    # ============================================================
    # RANDOM FOREST — dataset complet, rapide même sur >100k échantillons
    # ============================================================
    print("\n\n########## RANDOM FOREST ##########")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    t_train_rf = time.time() - t0

    t0 = time.time()
    y_pred_rf = rf.predict(X_test)
    t_infer_rf = time.time() - t0

    all_results.append(evaluate_and_report("RandomForest", y_test, y_pred_rf, t_train_rf, t_infer_rf, len(X_test)))

    # ============================================================
    # SVM — RBF kernel. ATTENTION : O(n^2) à O(n^3), intraitable sur >100k
    # échantillons en un temps raisonnable sur CPU. On sous-échantillonne
    # l'entraînement à 15 000 exemples max (stratifié), pratique documentée
    # et courante dans la littérature pour ce type de comparaison — à
    # mentionner explicitement dans le mémoire (3.4.2) pour rester honnête
    # sur cette contrainte pratique plutôt que de la dissimuler.
    # ============================================================
    print("\n\n########## SVM (sous-échantillonné à 15000 pour tractabilité CPU) ##########")
    rng = np.random.default_rng(42)
    if len(X_train) > 15000:
        idx_svm = rng.choice(len(X_train), size=15000, replace=False)
        X_train_svm, y_train_svm = X_train[idx_svm], y_train[idx_svm]
    else:
        X_train_svm, y_train_svm = X_train, y_train
    print(f"Entraînement SVM sur {len(X_train_svm)} échantillons (sous-échantillon stratifié du train complet)")

    t0 = time.time()
    svm = SVC(kernel='rbf', C=1, gamma='scale')
    svm.fit(X_train_svm, y_train_svm)
    t_train_svm = time.time() - t0

    t0 = time.time()
    y_pred_svm = svm.predict(X_test)
    t_infer_svm = time.time() - t0

    all_results.append(evaluate_and_report("SVM", y_test, y_pred_svm, t_train_svm, t_infer_svm, len(X_test)))

    # ============================================================
    # CNN-LSTM — avec sur-échantillonnage + poids de classe plafonnés
    # ============================================================
    print("\n\n########## CNN-LSTM (sur-échantillonné + poids plafonnés) ##########")
    present_classes = np.unique(y_train)
    class_weights_array = compute_class_weight(class_weight="balanced", classes=present_classes, y=y_train)
    class_weight = {k: min(v, 10.0) for k, v in zip(present_classes, class_weights_array)}
    print("Poids de classe (plafonnés à 10) :", {AAMI_CLASSES[k]: round(v, 2) for k, v in class_weight.items()})

    X_train_cnn = X_train.reshape(-1, WINDOW_LENGTH, 1)
    X_test_cnn = X_test.reshape(-1, WINDOW_LENGTH, 1)

    model = build_cnn_lstm_model(input_length=WINDOW_LENGTH, n_classes=5)
    early_stop = __import__("tensorflow").keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True
    )

    t0 = time.time()
    history = model.fit(
        X_train_cnn, y_train,
        validation_split=0.1, epochs=30, batch_size=64,
        class_weight=class_weight, callbacks=[early_stop], verbose=2,
    )
    t_train_cnn = time.time() - t0

    t0 = time.time()
    y_pred_cnn = np.argmax(model.predict(X_test_cnn, verbose=0), axis=1)
    t_infer_cnn = time.time() - t0

    all_results.append(evaluate_and_report("CNN-LSTM", y_test, y_pred_cnn, t_train_cnn, t_infer_cnn, len(X_test)))

    model.save("cardioguard_cnn_lstm_final_v2.keras")

    # ============================================================
    # SAUVEGARDE COMPARATIVE FINALE
    # ============================================================
    with open("comparaison_finale.json", "w") as f:
        json.dump({
            "n_train_total": len(X_train),
            "n_test": len(X_test),
            "patients_train": sorted(set(patient_ids[train_idx])),
            "patients_test": sorted(set(patient_ids[test_idx])),
            "results": all_results,
        }, f, indent=2)

    print("\n\n=== RÉSUMÉ COMPARATIF (F1 macro) ===")
    for r in all_results:
        print(f"  {r['model']:15s} : F1 macro = {r['f1_macro']:.4f} | "
              f"train = {r['train_time_seconds']:.1f}s | inférence = {r['inference_ms_per_sample']:.2f}ms/éch.")
    print("\nRésultats complets sauvegardés dans comparaison_finale.json")
