"""Loading, task definition, splits and metrics shared by all experiments."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             roc_curve, precision_recall_curve)
from config import DATA_DIR, SEED, TIME_COL, ORG_COL


def load_signals(path=None, sample=None):
    path = path or DATA_DIR / "signals" / "signals.parquet"
    df = pd.read_parquet(path)
    if sample and len(df) > sample:
        # keep every malicious and suspicious row, subsample benign
        keep = df[df.label_binary != "benign"]
        ben = df[df.label_binary == "benign"]
        n_ben = max(sample - len(keep), 0)
        df = pd.concat([keep, ben.sample(min(n_ben, len(ben)), random_state=SEED)]).sort_values(TIME_COL)
    return df.reset_index(drop=True)


def make_target(df, task="mal_vs_benign"):
    """Primary task: malicious vs benign. 'suspicious' rows are *defined* by rule
    matches, so learning them is learning the rules -> excluded from the primary task
    and reported separately (task='three_class')."""
    if task == "mal_vs_benign":
        df = df[df.label_binary.isin(["malicious", "benign"])].copy()
        y = (df.label_binary == "malicious").astype(int).values
    elif task == "mal_vs_rest":
        y = (df.label_binary == "malicious").astype(int).values
    elif task == "escalation":
        # Alert triage: among events that matched a detection rule, which ones
        # did the correlation engine escalate into an incident?
        df = df[df.label_binary.isin(["malicious", "suspicious"])].copy()
        y = (df.label_binary == "malicious").astype(int).values
    else:
        y = df.label_binary.map({"benign": 0, "suspicious": 1, "malicious": 2}).values
    return df.reset_index(drop=True), y


def first_incident(s):
    try:
        ids = json.loads(s) if isinstance(s, str) else []
        return ids[0] if ids else None
    except Exception:
        return None


# ---------------------------------------------------------------- splits
def split_random(df, y, test_size=0.2):
    """The flawed protocol most papers use (row-level random split). Kept ONLY to
    show how much it inflates results."""
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(df))
    cut = int(len(df) * (1 - test_size))
    return idx[:cut], idx[cut:]


def split_temporal(df, y, frac=0.7, gap_seconds=3600):
    """Train on the past, test on the future, with a gap so that the context
    window around one incident (+/-5 min in the small subset) cannot straddle the cut."""
    t = df[TIME_COL].values
    cut = np.quantile(t, frac)
    tr = np.where(t < cut)[0]
    te = np.where(t >= cut + gap_seconds)[0]
    return tr, te


def split_org(df, y, test_org):
    """Leave-one-organization-out: deployment on a network never seen in training."""
    te = np.where(df[ORG_COL].values == test_org)[0]
    tr = np.where(df[ORG_COL].values != test_org)[0]
    return tr, te


def split_incident_group(df, y, test_size=0.2):
    """Random split, but all rows of one incident stay on the same side."""
    groups = df.get("incident_ids", pd.Series([None] * len(df))).map(first_incident)
    groups = groups.fillna(pd.Series(np.arange(len(df)).astype(str), index=df.index))
    uniq = groups.unique()
    rng = np.random.RandomState(SEED)
    test_groups = set(rng.choice(uniq, int(len(uniq) * test_size), replace=False))
    m = groups.isin(test_groups).values
    return np.where(~m)[0], np.where(m)[0]


# ---------------------------------------------------------------- metrics
def recall_at_fpr(y, s, fpr_max=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= fpr_max
    return float(tpr[ok].max()) if ok.any() else 0.0


def evaluate(y, s, thr=None):
    """Imbalance-aware metrics. Accuracy is deliberately NOT reported."""
    out = {"n_test": int(len(y)), "n_pos": int(y.sum())}
    if y.min() == y.max():
        out["note"] = "single class in test set"
        return out
    out["pr_auc"] = average_precision_score(y, s)
    out["roc_auc"] = roc_auc_score(y, s)
    out["recall@fpr1%"] = recall_at_fpr(y, s, 0.01)
    out["recall@fpr0.1%"] = recall_at_fpr(y, s, 0.001)
    if thr is None:
        p, r, t = precision_recall_curve(y, s)
        f = 2 * p * r / (p + r + 1e-12)
        thr = t[np.argmax(f[:-1])] if len(t) else 0.5
    out["f1"] = f1_score(y, (s >= thr).astype(int))
    out["thr"] = float(thr)
    return out


def bootstrap_ci(y, s, metric=average_precision_score, n=500, alpha=0.05):
    rng = np.random.RandomState(SEED)
    vals = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(metric(y[i], s[i]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
