"""Final models on all data -> (1) SHAP for XGBoost, (2) external check on MAG proteins WITH PlasticDB hits
(never used in training), (3) homology-independent candidate list from out-of-fold scores of no-hit MAG proteins."""
import numpy as np, pandas as pd, shap, matplotlib, torch
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from features import featurize, onehot_idx, names
import importlib; T = importlib.import_module("02_train_lib")

D = Path(__file__).resolve().parent.parent / "data"; O = D.parent / "outputs"
df = pd.read_csv(D / "dataset.csv"); y = df.label.values; X = np.load(D / "X.npy")
ext = pd.read_csv(D / "mag_hit_proteins.csv")
hits = pd.read_csv(Path.home() / "metagenomics_project/mag_reconstruction/publication/data/plasticdb/plasticdb_best_hits.tsv", sep="\t")
ext = ext.merge(hits[["qseqid", "db_enzyme", "db_polymer", "pident", "scovhsp", "Tier"]], left_on="id", right_on="qseqid", how="left").drop(columns="qseqid")
Xe = np.stack([featurize(s) for s in ext.seq]); Xce = np.stack([onehot_idx(s) for s in ext.seq])

spw = (y == 0).sum() / (y == 1).sum()
xg = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
                   scale_pos_weight=spw, random_state=42, n_jobs=8, eval_metric="logloss").fit(X, y)
ext["xgboost"] = xg.predict_proba(Xe)[:, 1]
ext["cnn"] = T.fit_cnn(np.stack([onehot_idx(s) for s in df.seq]), y, Xce, spw)
ext.drop(columns="seq").to_csv(O / "external_mag_hits_scored.csv", index=False)
print(ext.groupby("Tier")[["xgboost", "cnn"]].agg(["mean", "count"]).round(3).to_string())
oof = pd.read_csv(O / "oof_scores.csv")
bg = oof[oof.label == 0]
for m in ["xgboost", "cnn"]:
    thr = bg[f"oof_{m}"].quantile(0.99)
    print(m, "frac of external hits above background 99th pct:", round((ext[m] > thr).mean(), 3),
          "| Tier1/2:", round((ext[ext.Tier.str.startswith(("Tier 1", "Tier 2"), na=False)][m] > thr).mean(), 3))

# SHAP
sv = shap.TreeExplainer(xg).shap_values(X)
imp = pd.Series(np.abs(sv).mean(0), index=names()).sort_values(ascending=False)
imp.head(25).to_csv(O / "shap_top25.csv")
shap.summary_plot(sv, pd.DataFrame(X, columns=names()), max_display=15, show=False)
plt.tight_layout(); plt.savefig(O / "shap_summary.png", dpi=200); plt.close()

# Homology-independent candidates: no-hit MAG proteins scored out-of-fold
c = bg.assign(score=(bg.oof_cnn.rank(pct=True) + bg.oof_xgboost.rank(pct=True)) / 2).sort_values("score", ascending=False)
c.head(25)[["id", "source", "product", "oof_cnn", "oof_xgboost", "hard_neg"]].to_csv(O / "candidates_no_homology.csv", index=False)
print(c.head(15)[["id", "source", "product", "oof_cnn", "oof_xgboost"]].round(3).to_string(index=False))

# Figures: PR curves
from sklearn.metrics import precision_recall_curve
plt.figure(figsize=(5, 4))
for m, lab in [("homology", "DIAMOND homology"), ("logreg", "Logistic reg."), ("xgboost", "XGBoost"), ("cnn", "1D-CNN")]:
    p, r, _ = precision_recall_curve(oof.label, oof[f"oof_{m}"]); plt.plot(r, p, label=lab)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(frameon=False); plt.title("Homology-grouped 5-fold CV")
plt.tight_layout(); plt.savefig(O / "pr_curves.png", dpi=200)
