import io, sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

model_path, xp, yp = sys.argv[1:4]
model = tf.keras.models.load_model(model_path)
X, y = np.load(xp), np.load(yp)
W0 = model.get_weights()
clone = tf.keras.models.clone_model(model)

def f1_of(ws):
    clone.set_weights([np.asarray(w, dtype=np.float32) for w in ws])
    p = clone.predict(X, batch_size=512, verbose=0).argmax(1)
    present = sorted(set(y) | set(p))
    return f1_score(y, p, labels=present, average="macro", zero_division=0)

def q8(a):
    sc = max(float(np.abs(a).max()) / 127, 1e-12)
    q = np.clip(np.round(a / sc), -127, 127).astype(np.int8)
    return q, np.float32(sc)

def size_kb(arrs):
    b = io.BytesIO(); np.savez_compressed(b, *[np.asarray(a) for a in arrs])
    return b.getbuffer().nbytes / 1024

def run(frac, quant):
    rec, store = [], []
    for w in W0:
        if w.ndim >= 2 and min(int(np.prod(w.shape[:-1])), w.shape[-1]) > 8:
            m, n = int(np.prod(w.shape[:-1])), w.shape[-1]
            r = max(1, int(round(frac * min(m, n)))) if frac < 1 else None
            if r is not None and r * (m + n) < m * n:
                U, S, Vt = np.linalg.svd(w.reshape(m, n), full_matrices=False)
                A = (U[:, :r] * S[:r]).astype(np.float32); B = Vt[:r].astype(np.float32)
                if quant:
                    (qa, sa), (qb, sb) = q8(A), q8(B)
                    A, B = qa.astype(np.float32) * sa, qb.astype(np.float32) * sb
                    store += [qa, sa, qb, sb]
                else:
                    store += [A, B]
                rec.append((A @ B).reshape(w.shape)); continue
            if quant:
                q, s = q8(w); rec.append(q.astype(np.float32) * s); store += [q, s]
            else:
                rec.append(w); store.append(w)
        else:
            rec.append(w); store.append(w)
    return size_kb(store), f1_of(rec)

base_kb, base_f1 = size_kb(W0), f1_of(W0)
rows = [("float32 de reference", base_kb, base_f1)]
kb, f = run(1.0, True);  rows.append(("int8 seul", kb, f))
for frac in (0.75, 0.5, 0.35, 0.25):
    kb, f = run(frac, False); rows.append((f"rang {int(frac*100)} % (float32)", kb, f))
    kb, f = run(frac, True);  rows.append((f"rang {int(frac*100)} % + int8", kb, f))

print(f"\n{'Variante':<26}{'Taille (Ko)':>13}{'F1 macro':>10}{'Delta F1':>10}")
for n, kb, f in rows:
    print(f"{n:<26}{kb:>13.1f}{f:>10.3f}{f - base_f1:>+10.3f}")
print("\nSVD apres entrainement, sans reentrainement, quantification simulee.")
