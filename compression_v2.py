import io, sys, time
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

model_path, xp, yp = sys.argv[1:4]
model = tf.keras.models.load_model(model_path)
X, y = np.load(xp), np.load(yp)
W0 = model.get_weights()
clone = tf.keras.models.clone_model(model)

def f1_of(weights):
    clone.set_weights([np.asarray(w, dtype=np.float32) for w in weights])
    pred = clone.predict(X, batch_size=512, verbose=0).argmax(1)
    present = sorted(set(y) | set(pred))
    return f1_score(y, pred, labels=present, average="macro", zero_division=0)

def prune(ws, s):
    out = []
    for w in ws:
        if w.ndim > 1:
            thr = np.quantile(np.abs(w), s)
            w = np.where(np.abs(w) < thr, 0.0, w).astype(np.float32)
        out.append(w)
    return out

def quant_int8(ws):
    deq, store = [], []
    for w in ws:
        if w.ndim > 1:
            scale = max(float(np.abs(w).max()) / 127, 1e-12)
            q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
            deq.append((q.astype(np.float32) * scale))
            store += [q, np.array(scale, dtype=np.float32)]
        else:
            deq.append(w)
            store.append(w)
    return deq, store

def to_f16(ws):
    st = [w.astype(np.float16) if w.ndim > 1 else w for w in ws]
    return [w.astype(np.float32) for w in st], st

def size_kb(arrays):
    buf = io.BytesIO()
    np.savez_compressed(buf, *[np.asarray(a) for a in arrays])
    return buf.getbuffer().nbytes / 1024

rows = []
t0 = time.perf_counter()
f_base = f1_of(W0)
ms = (time.perf_counter() - t0) / len(X) * 1000
rows.append(("Reference float32", size_kb(W0), f_base))

d, s = to_f16(W0)
rows.append(("Poids float16", size_kb(s), f1_of(d)))

d, s = quant_int8(W0)
rows.append(("Poids int8", size_kb(s), f1_of(d)))

for sp in (0.5, 0.8):
    P = prune(W0, sp)
    rows.append((f"Elagage {int(sp*100)} %", size_kb(P), f1_of(P)))
    d, s = quant_int8(P)
    rows.append((f"Elagage {int(sp*100)} % + int8", size_kb(s), f1_of(d)))

print(f"\n{'Variante':<28}{'Taille (Ko)':>13}{'F1 macro':>10}{'Delta F1':>10}")
for n, kb, f in rows:
    print(f"{n:<28}{kb:>13.1f}{f:>10.3f}{f - f_base:>+10.3f}")
print(f"\nLatence Keras float32 (Mac) : {ms:.3f} ms/battement")
print("Taille = poids stockes dans leur format, compresses (npz deflate).")
print("Quantification simulee (poids quantifies puis de-quantifies), sans noyaux entiers.")
