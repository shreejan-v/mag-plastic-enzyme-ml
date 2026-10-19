"""Homology-grouped 5-fold CV: DIAMOND-homology baseline vs logistic regression vs XGBoost vs 1D-CNN.
Writes out-of-fold scores + results table."""
import json, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from features import featurize, onehot_idx

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(8)
D = Path(__file__).resolve().parent.parent / "data"; O = D.parent / "outputs"; O.mkdir(exist_ok=True)
df = pd.read_csv(D / "dataset.csv")
y = df.label.values
X = np.stack([featurize(s) for s in df.seq])
Xc = np.stack([onehot_idx(s) for s in df.seq])
np.save(D / "X.npy", X)
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, df.group))
fold_of = np.zeros(len(df), int)
for k, (_, te) in enumerate(folds): fold_of[te] = k

# --- M0: homology baseline (best bitscore to a PlasticDB positive in the TRAINING folds only)
av = pd.read_csv(D / "allvall.tsv", sep="\t", names=["q", "s", "pid", "bits", "qcov", "scov"])
idx = {i: k for k, i in enumerate(df.id)}
av = av[av.q.isin(idx) & av.s.isin(idx) & (av.q != av.s)]
av = av[df.label.values[av.s.map(idx)] == 1]
av["qf"] = fold_of[av.q.map(idx)]; av["sf"] = fold_of[av.s.map(idx)]
av = av[av.qf != av.sf]
best = av.groupby("q").bits.max()
oof = {"homology": df.id.map(best).fillna(0).values.astype(float)}

# --- M1/M2: classical ML
oof["logreg"] = np.zeros(len(df)); oof["xgboost"] = np.zeros(len(df)); oof["cnn"] = np.zeros(len(df))
spw = (y == 0).sum() / (y == 1).sum()
class CNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(21, 32, padding_idx=0)
        s.c1 = nn.Conv1d(32, 128, 9, padding=4); s.c2 = nn.Conv1d(128, 128, 9, padding=4)
        s.drop = nn.Dropout(0.3); s.fc = nn.Linear(128, 1)
    def forward(s, x):
        h = s.emb(x).transpose(1, 2)
        h = torch.relu(s.c2(torch.relu(s.c1(h))))
        return s.fc(s.drop(h.max(2).values)).squeeze(1)
def fit_cnn(xtr, ytr, xte, epochs=12):
    m = CNN(); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(spw) ** 0.5))
    xt, yt = torch.tensor(xtr), torch.tensor(ytr, dtype=torch.float32)
    for _ in range(epochs):
        m.train(); perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 64):
            b = perm[i:i + 64]; opt.zero_grad(); lossf(m(xt[b]), yt[b]).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(torch.cat([m(torch.tensor(xte[i:i + 256])) for i in range(0, len(xte), 256)])).numpy()

for k, (tr, te) in enumerate(folds):
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced"))
    lr.fit(X[tr], y[tr]); oof["logreg"][te] = lr.predict_proba(X[te])[:, 1]
    xg = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
                       scale_pos_weight=spw, random_state=SEED, eval_metric="logloss", n_jobs=8)
    xg.fit(X[tr], y[tr]); oof["xgboost"][te] = xg.predict_proba(X[te])[:, 1]
    oof["cnn"][te] = fit_cnn(Xc[tr], y[tr], Xc[te])
    print("fold", k, "done", flush=True)

def tpr_at_fpr(yt, s, fpr=0.01):
    neg = np.sort(s[yt == 0])[::-1]; thr = neg[int(np.ceil(fpr * len(neg))) - 1]
    return (s[yt == 1] > thr).mean()
rows = []
hard = df.hard_neg.values; neg = y == 0
for name, s in oof.items():
    m_hard = (y == 1) | hard  # positives vs hydrolase-like negatives only
    rows.append(dict(model=name, roc_auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s),
                     recall_at_1pct_fpr=tpr_at_fpr(y, s),
                     roc_auc_vs_hard_negatives=roc_auc_score(y[m_hard], s[m_hard]),
                     recall_at_1pct_fpr_hard=tpr_at_fpr(y[m_hard], s[m_hard])))
res = pd.DataFrame(rows); res.to_csv(O / "results.csv", index=False)
print(res.round(3).to_string(index=False))
print("positive rate:", round(y.mean(), 3), "| hard negatives:", int(hard.sum()))
out = df[["id", "label", "source", "product", "polymer", "hard_neg", "group"]].copy()
for k, v in oof.items(): out[f"oof_{k}"] = v
out["fold"] = fold_of; out.to_csv(O / "oof_scores.csv", index=False)
