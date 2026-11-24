"""Sec. 5.1, 'first featurization': why the corrected features were needed.

Uses the ORIGINAL phase-1 featurizer kept in this folder (features.py: entity frequencies fitted on the
training split, hour and weekday). For the naive global time cut, a random event split and an
incident-grouped split, the featurizer and a LightGBM model are fitted on the training split and
scored on the test split; for the naive cut, the univariate ROC-AUC of dst_ip_freq is reported on
training and test rows. Run from the repository root (needs data/100m/rulehits.parquet):

    python extras/first_featurization/first_featurization_check.py

Output: results/first_featurization/first_featurization.csv
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from common import load_signals, make_target, split_temporal, split_random, split_incident_group
from config import DATA_DIR, SEED, TIME_COL
from features import TabularFeaturizer

df, y = make_target(load_signals(DATA_DIR / "100m" / "rulehits.parquet"), "escalation")
print(f"rows={len(df):,} pos={y.sum():,}")
rows = []
for name, (tr, te) in {"naive_global_temporal": split_temporal(df, y),
                       "random_events": split_random(df, y),
                       "incident_grouped": split_incident_group(df, y)}.items():
    d_tr, d_te = df.iloc[tr], df.iloc[te]
    t = d_tr[TIME_COL].values; fit = np.where(t < np.quantile(t, 0.85))[0]
    fz = TabularFeaturizer().fit(d_tr); fit = np.arange(len(d_tr))
    Xtr, Xte = fz.transform(d_tr), fz.transform(d_te)
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, is_unbalance=True,
                           random_state=SEED, verbose=-1).fit(Xtr.iloc[fit].to_numpy(np.float32), y[tr][fit])
    roc = roc_auc_score(y[te], m.predict_proba(Xte.to_numpy(np.float32))[:, 1])
    line = f"{name:22s} lgbm_tab ROC-AUC={roc:.3f}"
    if name == "naive_global_temporal":
        line += (f" | dst_ip_freq univariate ROC train={roc_auc_score(y[tr], Xtr['dst_ip_freq']):.3f}"
                 f" test={roc_auc_score(y[te], Xte['dst_ip_freq']):.3f}")
    print(line, flush=True)
    row = {"protocol": name, "model": "lgbm_tab", "roc_auc": roc}
    if name == "naive_global_temporal":
        row.update(dst_ip_freq_roc_train=roc_auc_score(y[tr], Xtr["dst_ip_freq"]),
                   dst_ip_freq_roc_test=roc_auc_score(y[te], Xte["dst_ip_freq"]))
    rows.append(row)
from pathlib import Path
out = Path("results") / "first_featurization"; out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out / "first_featurization.csv", index=False)
print("saved ->", out)
