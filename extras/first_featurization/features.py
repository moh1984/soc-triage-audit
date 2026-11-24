"""Feature extraction from observable fields only (no label-derived columns).

Fit on the training split, then transform the test split: frequency/target
statistics computed on the whole dataset would be data snooping."""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from config import CATEGORICAL, HIGH_CARD, PORTS, TEXT_COL, TIME_COL


class TabularFeaturizer:
    def __init__(self, top_k=50):
        self.top_k = top_k

    def fit(self, df):
        self.vocab = {c: df[c].astype(str).value_counts().index[: self.top_k].tolist()
                      for c in CATEGORICAL if c in df}
        # how common an entity is in TRAIN (rare host/user = potentially interesting)
        self.freq = {c: df[c].astype(str).value_counts(normalize=True)
                     for c in HIGH_CARD if c in df}
        return self

    def transform(self, df):
        out = {}
        for c, vals in self.vocab.items():
            s = df[c].astype(str)
            for v in vals:
                out[f"{c}={v}"] = (s == v).astype(np.int8)
            out[f"{c}=__other__"] = (~s.isin(vals)).astype(np.int8)
        for c, f in self.freq.items():
            s = df[c].astype(str)
            out[f"{c}_freq"] = s.map(f).fillna(0.0).astype(float)
            out[f"{c}_missing"] = df[c].isna().astype(np.int8)
        for c in PORTS:
            if c in df:
                p = pd.to_numeric(df[c], errors="coerce")
                out[f"{c}_num"] = p.fillna(-1)
                out[f"{c}_wellknown"] = (p < 1024).astype(np.int8)
        if TIME_COL in df:
            ts = pd.to_datetime(df[TIME_COL], unit="s", utc=True)
            out["hour"] = ts.dt.hour
            out["weekday"] = ts.dt.weekday
        if TEXT_COL in df:
            out["msg_len"] = df[TEXT_COL].fillna("").str.len()
        X = pd.DataFrame(out, index=df.index)
        return X

    def fit_transform(self, df):
        return self.fit(df).transform(df)


def text_features(df, n_features=2 ** 18):
    """Stateless hashing of the sanitized raw log (char/word n-grams).
    Sanitized tokens like HOST-0042 / USER-0007 are masked so the model cannot
    memorize specific machines."""
    txt = (df[TEXT_COL].fillna("")
           .str.replace(r"(HOST|USER|ORG)-\d+", r"\1", regex=True)
           .str.replace(r"\d+", "0", regex=True))
    hv = HashingVectorizer(n_features=n_features, ngram_range=(1, 2),
                           alternate_sign=False, norm="l2", token_pattern=r"[A-Za-z_]{2,}")
    return hv.transform(txt)


def combine(X_tab, X_txt):
    return sparse.hstack([sparse.csr_matrix(X_tab.values.astype(np.float32)), X_txt]).tocsr()
