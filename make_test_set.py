import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from train_comparatif import ALL_RECORDS, prepare_dataset

model_path = sys.argv[1] if len(sys.argv) > 1 else "cardioguard_cnn_lstm_final_v2.keras"

beats, labels, patient_ids = prepare_dataset(ALL_RECORDS)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(beats, labels, groups=patient_ids))
assert not (set(patient_ids[train_idx]) & set(patient_ids[test_idx])), "Fuite inter-patient"

X_test = beats[test_idx].reshape(len(test_idx), -1, 1).astype(np.float32)
y_test = labels[test_idx].astype(int)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
print(f"X_test {X_test.shape} | y_test {y_test.shape} | classes {np.bincount(y_test)}")
print(f"Patients test : {sorted(set(patient_ids[test_idx]))}")

model = tf.keras.models.load_model(model_path)
pred = model.predict(X_test, batch_size=256, verbose=0).argmax(1)
present = sorted(set(y_test) | set(pred))
f1 = f1_score(y_test, pred, labels=present, average="macro", zero_division=0)
print(f"\nF1 macro de {model_path} sur ce test : {f1:.3f}  (memoire : 0,442)")
if abs(f1 - 0.442) > 0.03:
    print("[!] Ecart notable : ce modele ne vient probablement pas de train_comparatif.py.")
