"""Unit tests for the rules the paper relies on (run from the repository root):

    python -m pytest tests/        or        python tests/test_protocols.py

They use a small synthetic table built here, so no dataset is needed.
  1. splits: train and test never share a row; temporal splits respect the gap;
     incident-grouped and cross-source splits never put one incident / source on both sides
  2. causal context: features of an event do not change when future events are removed
  3. label-derived columns never reach the feature matrix
  4. incident-level aggregation (GUIDE): one row per (org, incident), label and score = max
  5. metrics: PR-AUC / ROC-AUC match scikit-learn; recall at a fixed FPR is computed correctly
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import split_temporal, split_incident_group, evaluate, recall_at_fpr  # noqa: E402
from config import LEAKY_COLS, TIME_COL  # noqa: E402
from features import TabularFeaturizer, context_features  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def toy(n=4000, seed=0):
    rng = np.random.RandomState(seed)
    t = 1721952000 + np.sort(rng.randint(0, 6 * 86400, n)).astype(float)
    lab = rng.choice(["suspicious", "malicious"], n, p=[.9, .1])
    inc = [json.dumps([f"inc-{rng.randint(40)}"]) if l == "malicious" else "[]" for l in lab]
    df = pd.DataFrame({
        TIME_COL: t, "org_id": rng.choice(["ORG-0004", "ORG-0005"], n),
        "stream_name": rng.choice(["cisco_asa", "aws_vpc_flow_log"], n),
        "message_type": rng.choice(["a", "b", "c"], n), "protocol": rng.choice(["6", "17"], n),
        "src_ip": [f"10.0.0.{i}" for i in rng.randint(0, 30, n)],
        "dst_ip": [f"10.1.0.{i}" for i in rng.randint(0, 30, n)],
        "src_host": [f"HOST-{i:04d}" for i in rng.randint(0, 50, n)],
        "dst_host": [f"HOST-{i:04d}" for i in rng.randint(0, 50, n)],
        "username": [f"USER-{i:04d}" for i in rng.randint(0, 20, n)],
        "src_port": rng.randint(1, 65000, n).astype(str), "dst_port": rng.choice(["22", "443"], n),
        "message_sanitized": ["event"] * n,
        "label_binary": lab, "incident_ids": inc,
        "label_confidence": rng.rand(n), "suspicion_score": rng.rand(n), "mo_name": "x",
    })
    y = (df.label_binary == "malicious").astype(int).to_numpy()
    return df, y


# ------------------------------------------------------------------ 1. splits
def test_temporal_split_disjoint_and_gap():
    df, y = toy()
    tr, te = split_temporal(df, y, frac=0.7, gap_seconds=3600)
    assert len(set(tr) & set(te)) == 0
    t = df[TIME_COL].to_numpy()
    assert t[tr].max() + 3600 <= t[te].min()


def test_incident_grouped_split_keeps_incidents_whole():
    df, y = toy()
    tr, te = split_incident_group(df, y)
    assert len(set(tr) & set(te)) == 0 and len(tr) + len(te) == len(df)
    first = df.incident_ids.map(lambda s: (json.loads(s) or [None])[0])
    inc_tr, inc_te = set(first.iloc[tr].dropna()), set(first.iloc[te].dropna())
    assert inc_tr.isdisjoint(inc_te)


def test_precinct6_protocols_do_not_mix_sources_or_time():
    B = _load("b04", "04_baselines.py")
    df, y = toy()
    P = B.protocols(df, y, vpc_cut="2024-07-29")
    st, t = df.stream_name.to_numpy(), df[TIME_COL].to_numpy()
    for name, (tr, te) in P.items():
        assert len(set(tr) & set(te)) == 0, name
    tr, te = P["asa_temporal"]
    assert set(st[tr]) == set(st[te]) == {"cisco_asa"} and t[tr].max() < t[te].min()
    tr, te = P["vpc_wave"]
    assert set(st[tr]) == set(st[te]) == {"aws_vpc_flow_log"} and t[tr].max() < t[te].min()
    tr, te = P["asa_to_vpc"]
    assert set(st[tr]) == {"cisco_asa"} and set(st[te]) == {"aws_vpc_flow_log"}


# ------------------------------------------------------------------ 2. causal context
def test_causal_context_ignores_future_events():
    df, _ = toy()
    full = context_features(df)                      # computed on the whole table
    k = len(df) // 2
    past = context_features(df.iloc[:k].copy())     # same events, future removed
    pd.testing.assert_frame_equal(full.iloc[:k].reset_index(drop=True),
                                  past.reset_index(drop=True), check_dtype=False)


def test_two_sided_context_does_use_future_events():
    # sanity check that the test above can fail: event-wide context must change
    df, _ = toy()
    k = len(df) // 2
    full = context_features(df, mode="two_sided").iloc[:k].reset_index(drop=True)
    past = context_features(df.iloc[:k].copy(), mode="two_sided").reset_index(drop=True)
    assert not full.equals(past)


# ------------------------------------------------------------------ 3. no label-derived features
def test_label_derived_columns_never_become_features():
    df, y = toy()
    cols = list(TabularFeaturizer().fit_transform(df).columns) + list(context_features(df).columns)
    for leak in LEAKY_COLS:
        assert not any(c == leak or c.startswith(leak + "=") or c.startswith(leak + "_") for c in cols), leak


# ------------------------------------------------------------------ 4. incident aggregation
def test_incident_aggregation_takes_max_per_org_and_incident():
    V = _load("g15", "15_guide_v2.py")
    org = np.array(["o1", "o1", "o1", "o2", "o2"])
    inc = np.array([1, 1, 2, 1, 1])                  # incident 1 exists in two organizations
    y = np.array([0, 1, 0, 0, 0])
    tab = V.to_incidents(org, inc, y, {"m": np.array([.2, .9, .4, .1, .3])})
    assert len(tab) == 3
    row = tab[(tab.org == "o1") & (tab.inc == 1)].iloc[0]
    assert row.y == 1 and row.m == .9
    assert tab[(tab.org == "o2")].iloc[0].m == .3


# ------------------------------------------------------------------ 5. metrics
def test_metrics_match_sklearn_and_recall_at_fpr():
    rng = np.random.RandomState(1)
    y = (rng.rand(2000) < .05).astype(int)
    s = rng.rand(2000) + y * .5
    r = evaluate(y, s)
    assert abs(r["pr_auc"] - average_precision_score(y, s)) < 1e-12
    assert abs(r["roc_auc"] - roc_auc_score(y, s)) < 1e-12
    y2 = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    s2 = np.array([.9, .2, .8, .1, .1, .1, .1, .1, .1, .1])
    assert recall_at_fpr(y2, s2, 0.0) == .5          # one positive above every negative
    assert recall_at_fpr(y2, s2, 0.12) == .5         # 1/8 FPR > 0.12 is not allowed
    assert recall_at_fpr(y2, s2, 1.0) == 1.0


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f()
        print("ok ", f.__name__)
    print(f"{len(fns)} tests passed")
