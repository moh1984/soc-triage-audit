"""Feature extraction from observable fields only (no label-derived columns).

Design rules (learned from 06_diagnose_temporal.py on the full capture):
  * No calendar features (hour, weekday): the live capture is one week, so they
    only memorise when each attack wave happened.
  * No entity frequencies fitted on the training split: rows in train always find
    their own IP in the table, unseen test IPs get 0, and the direction flips.
    Entity context is computed CAUSALLY instead (past events only, same rule for
    every row), so train and test features mean the same thing.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from config import CATEGORICAL, HIGH_CARD, PORTS, TEXT_COL, TIME_COL, ORG_COL

WINDOWS = {"5m": 300, "1h": 3600, "24h": 86400}


class TabularFeaturizer:
    """One-hot of low-cardinality fields (vocabulary fitted on train), ports, length."""

    def __init__(self, top_k=50):
        self.top_k = top_k

    def fit(self, df):
        self.vocab = {c: df[c].astype(str).value_counts().index[: self.top_k].tolist()
                      for c in CATEGORICAL if c in df}
        return self

    def transform(self, df):
        out = {}
        for c, vals in self.vocab.items():
            s = df[c].astype(str)
            for v in vals:
                out[f"{c}={v}"] = (s == v).astype(np.int8)
            out[f"{c}=__other__"] = (~s.isin(vals)).astype(np.int8)
        for c in HIGH_CARD:
            if c in df:
                out[f"{c}_missing"] = (df[c].isna() | (df[c].astype(str) == "")).astype(np.int8)
        for c in PORTS:
            if c in df:
                p = pd.to_numeric(df[c], errors="coerce")
                out[f"{c}_num"] = p.fillna(-1)
                out[f"{c}_wellknown"] = (p < 1024).astype(np.int8)
        if TEXT_COL in df:
            out["msg_len"] = df[TEXT_COL].fillna("").str.len()
        return pd.DataFrame(out, index=df.index)

    def fit_transform(self, df):
        return self.fit(df).transform(df)


def _prior_counts(t, codes, w):
    """For each row: number of EARLIER rows with the same code within w seconds.
    t must be sorted ascending within each code; computed with one global sort."""
    t0 = t.min()
    span = t.max() - t0 + w + 1.0                           # groups never overlap
    key = codes.astype(np.float64) * span + (t - t0)
    order = np.argsort(key, kind="mergesort")
    ks = key[order]
    left = np.searchsorted(ks, ks - w, side="left")
    pos = np.arange(len(ks))
    cnt = np.empty(len(ks), dtype=np.int64)
    cnt[order] = pos - left
    return cnt


def _two_sided_counts(t, codes, w):
    """For each row: number of OTHER rows with the same code within [t - w, t + w].
    Uses future events too, so it is NOT available at decision time; used only to
    measure how much non-causal (event-wide) context inflates results."""
    t0 = t.min()
    span = t.max() - t0 + 2 * w + 1.0
    key = codes.astype(np.float64) * span + (t - t0)
    order = np.argsort(key, kind="mergesort")
    ks = key[order]
    left = np.searchsorted(ks, ks - w, side="left")
    right = np.searchsorted(ks, ks + w, side="right")
    cnt = np.empty(len(ks), dtype=np.int64)
    cnt[order] = right - left - 1
    return cnt


def context_features(df, minute_counts=None, mode="causal"):
    """mode='causal' (default, used everywhere in the paper): past events only.
    mode='two_sided': same features over past AND future events (event-wide context,
    as in methods that aggregate over a whole cluster or security event)."""
    if mode == "two_sided":
        return _context_two_sided(df, minute_counts)
    return _context_causal(df, minute_counts)


def _context_two_sided(df, minute_counts=None):
    t = df[TIME_COL].to_numpy(dtype=np.float64)
    out = {}
    ents = [c for c in HIGH_CARD if c in df]
    pairs = {"pair_ip": (df["src_ip"].astype(str) + ">" + df["dst_ip"].astype(str))} \
        if {"src_ip", "dst_ip"} <= set(df.columns) else {}
    full = t.max() - t.min() + 1
    for name, col in [(c, df[c].astype(str)) for c in ents] + list(pairs.items()):
        codes = pd.factorize(col)[0]
        for wn, w in WINDOWS.items():
            out[f"ctx_{name}_{wn}"] = np.log1p(_two_sided_counts(t, codes, w))
        out[f"ctx_{name}_all"] = np.log1p(_two_sided_counts(t, codes, full))
    org = pd.factorize(df[ORG_COL].astype(str))[0]
    for wn in ["5m", "1h"]:
        out[f"ctx_org_{wn}"] = np.log1p(_two_sided_counts(t, org, WINDOWS[wn]))
    if minute_counts is not None:
        tot = minute_counts.groupby(["org_id", "minute"])["n"].sum()
        minute = (t // 60).astype(np.int64)
        win = []
        for k in range(-5, 6):                      # 5 minutes before AND after
            if k == 0:
                continue
            key = pd.MultiIndex.from_arrays([df[ORG_COL].values, minute + k])
            win.append(tot.reindex(key).fillna(0).to_numpy())
        out["vol_org_prev5m"] = np.log1p(np.sum(win, axis=0))
    return pd.DataFrame(out, index=df.index)


def _context_causal(df, minute_counts=None):
    """Causal context for each event, computed on the whole (time-sorted) table.
    Uses only events that happened BEFORE the row, so it is valid for any split.
      - per entity (src_ip, dst_ip, src_host, dst_host, username, src-dst pair):
        prior rule-hit events in 5 min / 1 h / 24 h, and log of all prior events
      - per org: prior rule-hit events in 5 min / 1 h
      - optional honest volume from the full capture (all events, benign included)
    """
    t = df[TIME_COL].to_numpy(dtype=np.float64)
    out = {}
    ents = [c for c in HIGH_CARD if c in df]
    pairs = {"pair_ip": (df["src_ip"].astype(str) + ">" + df["dst_ip"].astype(str))} \
        if {"src_ip", "dst_ip"} <= set(df.columns) else {}
    for name, col in [(c, df[c].astype(str)) for c in ents] + list(pairs.items()):
        codes = pd.factorize(col)[0]
        for wn, w in WINDOWS.items():
            out[f"ctx_{name}_{wn}"] = np.log1p(_prior_counts(t, codes, w))
        out[f"ctx_{name}_all"] = np.log1p(_prior_counts(t, codes, t.max() - t.min() + 1))
    org = pd.factorize(df[ORG_COL].astype(str))[0]
    for wn in ["5m", "1h"]:
        out[f"ctx_org_{wn}"] = np.log1p(_prior_counts(t, org, WINDOWS[wn]))
    if minute_counts is not None:
        tot = minute_counts.groupby(["org_id", "minute"])["n"].sum()
        minute = (t // 60).astype(np.int64)
        prev = []
        for k in range(1, 6):                       # the 5 minutes BEFORE this one
            key = pd.MultiIndex.from_arrays([df[ORG_COL].values, minute - k])
            prev.append(tot.reindex(key).fillna(0).to_numpy())
        out["vol_org_prev5m"] = np.log1p(np.sum(prev, axis=0))
    return pd.DataFrame(out, index=df.index)


def text_features(df, n_features=2 ** 16):
    """Stateless hashing of the sanitized raw log. Sanitized tokens (HOST-0042,
    USER-0007) and all digits are masked so the model cannot memorise machines."""
    txt = (df[TEXT_COL].fillna("")
           .str.replace(r"(HOST|USER|ORG)-\d+", r"\1", regex=True)
           .str.replace(r"\d+", "0", regex=True))
    hv = HashingVectorizer(n_features=n_features, ngram_range=(1, 2),
                           alternate_sign=False, norm="l2", token_pattern=r"[A-Za-z_]{2,}")
    return hv.transform(txt)


def combine(*blocks):
    mats = [b if sparse.issparse(b) else sparse.csr_matrix(np.asarray(b, dtype=np.float32))
            for b in blocks]
    return sparse.hstack(mats).tocsr()
