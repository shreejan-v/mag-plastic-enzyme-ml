import numpy as np, itertools
AA = "ACDEFGHIKLMNPQRSTVWY"
IDX = {a: i for i, a in enumerate(AA)}
# conjoint-triad 7-class reduced alphabet
RED = {}
for k, grp in enumerate(["AGV", "ILFP", "YMTS", "HNQW", "RK", "DE", "C"]):
    for a in grp: RED[a] = k
TRI = {t: i for i, t in enumerate(itertools.product(range(7), repeat=3))}

def featurize(seq):
    n = len(seq)
    f1 = np.zeros(20); f2 = np.zeros(400); f3 = np.zeros(343)
    for a in seq: f1[IDX[a]] += 1
    for a, b in zip(seq, seq[1:]): f2[IDX[a] * 20 + IDX[b]] += 1
    r = [RED[a] for a in seq]
    for t in zip(r, r[1:], r[2:]): f3[TRI[t]] += 1
    return np.concatenate([f1 / n, f2 / max(n - 1, 1), f3 / max(n - 2, 1), [np.log(n)]])

def names():
    return ([f"aa_{a}" for a in AA] + [f"di_{a}{b}" for a in AA for b in AA]
            + [f"tri_{t}" for t in TRI] + ["log_len"])

def onehot_idx(seq, L=1000):
    x = np.zeros(L, dtype=np.int64)
    for i, a in enumerate(seq[:L]): x[i] = IDX[a] + 1
    return x
