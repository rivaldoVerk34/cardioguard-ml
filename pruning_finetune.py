import io, json, sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from train_comparatif import ALL_RECORDS, prepare_dataset, oversample_minority_classes

MODEL = sys.argv[1] if len(sys.argv) > 1 else "cardioguard_cnn_lstm_final_v2.keras"
STAGES = [0.3, 0.5, 0.7, 0.8]   # sparsite cumulative visee
EPOCHS = 5                      # epoques de reentrainement par etape
CAP = 10.0                      # plafond des poids de classe (a ajuster comme dans ton entrainement)

beats, labels, pids = prepare_dataset(ALL_RECORDS)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr, te = next(gss.split(beats, labels, groups=pids))
res = oversample_minority_classes(beats[tr], labels[tr])
Xtr, ytr = res[0], res[1]
Xtr = Xtr.reshape(len(Xtr), -1, 1).astype(np.float32); ytr = ytr.astype(int)
Xte = beats[te].reshape(len(te), -1, 1).astype(np.float32); yte = labels[te].astype(int)
cls = np.unique(ytr)
cw = {int(c): float(min(w, CAP)) for c, w in zip(cls, compute_class_weight("balanced", classes=cls, y=ytr))}

model = tf.keras.models.load_model(MODEL)
model.compile(optimizer=tf.keras.optimizers.Adam(5e-4),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

def kernels():
    return [v for v in model.weights if len(v.shape) > 1]

masks = {id(v): np.ones(v.shape, np.float32) for v in kernels()}

def apply_masks():
    for v in kernels():
        v.assign(v.numpy() * masks[id(v)])

def prune_to(s):
    for v in kernels():
        w = np.abs(v.numpy())
        thr = np.quantile(w, s)
        masks[id(v)] *= (w > thr).astype(np.float32)
    apply_masks()

class Mask(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        apply_masks()

def f1_test():
    p = model.predict(Xte, batch_size=512, verbose=0).argmax(1)
    present = sorted(set(yte) | set(p))
    return f1_score(yte, p, labels=present, average="macro", zero_division=0)

def zero_frac():
    ws = np.concatenate([v.numpy().ravel() for v in kernels()])
    return float((ws == 0).mean())

def size_kb(int8):
    arrs = []
    for v in model.weights:
        w = v.numpy()
        if int8 and w.ndim > 1:
            sc = max(float(np.abs(w).max()) / 127, 1e-12)
            arrs += [np.clip(np.round(w / sc), -127, 127).astype(np.int8), np.array(sc, np.float32)]
        else:
            arrs.append(w.astype(np.float32))
    buf = io.BytesIO(); np.savez_compressed(buf, *arrs)
    return buf.getbuffer().nbytes / 1024

def f1_int8():
    saved = [v.numpy() for v in model.weights]
    for v in kernels():
        w = v.numpy(); sc = max(float(np.abs(w).max()) / 127, 1e-12)
        v.assign(np.clip(np.round(w / sc), -127, 127) * sc)
    f = f1_test()
    for v, w in zip(model.weights, saved):
        v.assign(w)
    return f

rows = []
base = f1_test()
print(f"F1 de depart : {base:.3f} (attendu ~0.441)")
rows.append(dict(etape="reference", zeros=0.0, f1_avant=base, f1_apres=base,
                 kb_f32=size_kb(False), kb_int8=size_kb(True), f1_int8=f1_int8()))

for s in STAGES:
    prune_to(s)
    f_avant = f1_test()
    model.fit(Xtr, ytr, validation_split=0.1, epochs=EPOCHS, batch_size=64,
              class_weight=cw, callbacks=[Mask()], verbose=2)
    apply_masks()
    rows.append(dict(etape=f"{int(s*100)} %", zeros=zero_frac(), f1_avant=f_avant,
                     f1_apres=f1_test(), kb_f32=size_kb(False), kb_int8=size_kb(True),
                     f1_int8=f1_int8()))

print(f"\n{'Etape':<10}{'Zeros':>8}{'F1 apres elag.':>16}{'F1 reentraine':>15}{'Ko f32':>9}{'Ko int8':>9}{'F1 int8':>9}")
for r in rows:
    print(f"{r['etape']:<10}{r['zeros']:>8.2f}{r['f1_avant']:>16.3f}{r['f1_apres']:>15.3f}"
          f"{r['kb_f32']:>9.1f}{r['kb_int8']:>9.1f}{r['f1_int8']:>9.3f}")
json.dump(rows, open("pruning_finetune_results.json", "w"), indent=2)
model.save("cardioguard_cnn_lstm_pruned.keras")
