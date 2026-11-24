"""Step 8 - the proposed model: CtxGraph (temporal memory graph for alert escalation).

Each rule-matched event e = (src entity, dst entity, t, features x_e) is an edge in a
continuous-time graph. Design follows the baseline evidence (research log C13-C19):

  * Node memory (TGN-style, Rossi et al. 2020): every entity (src_ip, dst_ip, user)
    has a memory vector updated by a GRU from the messages of its past events.
    Memory is updated at the start of the NEXT batch from stored raw messages, so the
    GRU receives gradients and an event never sees itself.
  * Event features: tabular encoder (same observable fields as the baselines) +
    causal context counts + optional template bag (digits, hosts, users removed,
    256 hash bins). No raw-text n-grams (C14: raw text does not transfer).
  * Escalation head: MLP on [memory(src), memory(dst), time encodings, x_e] -> logit.
  * Anomaly head: link prediction with negative destinations (Kairos/TGN style);
    anomaly = 1 - p(real edge). Final score = z(logit) + lambda * z(anomaly), lambda
    picked on the validation slice (C15: anomaly-in-context carries signal).
  * Source adversary: gradient reversal on the event embedding predicting stream_name
    (DANN, Ganin et al. 2016). Active only with --uda, which adds the UNLABELED test-
    source events to the training pass (unsupervised domain adaptation; reported
    separately because baselines never saw target data).

Ablations: --no-memory, --no-ctx, --text none, --no-anomaly, --adv 0.
Protocols and splits are imported from 04_baselines.py, so numbers are comparable.

Usage (Colab GPU):
  python 08_ctxgraph.py --only asa_temporal,vpc_wave --seeds 0,1,2,3,4
  python 08_ctxgraph.py --only asa_to_vpc,vpc_to_asa --seeds 0,1,2,3,4 --uda
Outputs: results/ctxgraph/<tag>/runs.csv, summary.csv, scores_<protocol>_<seed>.npz
"""
import argparse
import importlib.util
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score

from common import load_signals, make_target, evaluate, bootstrap_ci, first_incident
from config import DATA_DIR, RESULTS_DIR, TIME_COL, TEXT_COL
from features import TabularFeaturizer, context_features

spec = importlib.util.spec_from_file_location("baselines", Path(__file__).with_name("04_baselines.py"))
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ data
def template_bag(df, n_bins=256):
    """Bag of template words: digits, IPs, sanitized HOST/USER/ORG tokens removed."""
    txt = (df[TEXT_COL].fillna("")
           .str.replace(r"(HOST|USER|ORG)-\d+", " ", regex=True)
           .str.replace(r"[0-9a-fA-F:.]{7,}", " ", regex=True)      # IPs, hashes, uuids
           .str.replace(r"\d+", " ", regex=True))
    hv = HashingVectorizer(n_features=n_bins, alternate_sign=False, norm="l2",
                           token_pattern=r"[A-Za-z_]{3,}", lowercase=True)
    return hv.transform(txt).astype(np.float32).toarray()


def build_nodes(df):
    """Global node ids for src and dst entities (scoped by org for private values)."""
    org = df["org_id"].astype(str)
    src = org + "/" + df["src_ip"].astype(str)
    dst = org + "/" + df["dst_ip"].astype(str)
    codes, uniq = pd.factorize(pd.concat([src, dst], ignore_index=True))
    n = len(df)
    return codes[:n].astype(np.int64), codes[n:].astype(np.int64), len(uniq)


# ------------------------------------------------------------------ model
class TimeEnc(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.w = nn.Linear(1, d)
        with torch.no_grad():
            self.w.weight.copy_(torch.from_numpy(1 / 10 ** np.linspace(0, 6, d)).float().view(d, 1))
            self.w.bias.zero_()

    def forward(self, dt):
        return torch.cos(self.w(torch.log1p(dt.clamp(min=0)).unsqueeze(-1)))


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a):
        ctx.a = a
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.a * g, None


class CtxGraph(nn.Module):
    def __init__(self, n_nodes, d_x, n_dom, mem=64, td=16, hid=128, use_memory=True):
        super().__init__()
        self.use_memory = use_memory
        self.mem_dim = mem
        self.tenc = TimeEnc(td)
        self.x_enc = nn.Sequential(nn.Linear(d_x, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.msg = nn.Linear(2 * mem + hid + td, mem)
        self.gru = nn.GRUCell(mem, mem)
        din = hid + (2 * mem + 2 * td if use_memory else 0)
        self.emb = nn.Sequential(nn.Linear(din, hid), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hid, hid), nn.ReLU())
        self.head = nn.Linear(hid, 1)
        self.link = nn.Sequential(nn.Linear(2 * mem + hid, hid), nn.ReLU(), nn.Linear(hid, 1))
        self.dom = nn.Sequential(nn.Linear(hid, hid // 2), nn.ReLU(), nn.Linear(hid // 2, max(n_dom, 2)))
        self.n_nodes = n_nodes
        self.reset_state()

    # ---- state that lives outside autograd
    def reset_state(self):
        self.memory = torch.zeros(self.n_nodes, self.mem_dim, device=DEV)
        self.last_t = torch.zeros(self.n_nodes, device=DEV)
        self.pend_other = torch.full((self.n_nodes,), -1, dtype=torch.long, device=DEV)
        self.pend_x = None                      # [n_nodes, hid] filled lazily
        self.pend_t = torch.zeros(self.n_nodes, device=DEV)

    def _apply_pending(self, nodes):
        """GRU update for `nodes` from their last stored message (with gradients)."""
        mem = self.memory[nodes]
        if self.pend_x is None:
            return mem
        other = self.pend_other[nodes]
        has = other >= 0
        if not has.any():
            return mem
        n_h, o_h = nodes[has], other[has]
        dt = self.pend_t[n_h] - self.last_t[n_h]
        m_in = torch.cat([self.memory[n_h], self.memory[o_h], self.pend_x[n_h], self.tenc(dt)], -1)
        new = self.gru(F.relu(self.msg(m_in)), self.memory[n_h])
        mem = mem.clone()
        mem[has] = new
        return mem

    def forward_batch(self, src, dst, t, x):
        """Score a batch with memory as it stood BEFORE the batch; queue its messages."""
        hx = self.x_enc(x)
        if self.use_memory:
            nodes = torch.unique(torch.cat([src, dst]))
            upd = self._apply_pending(nodes)                       # grads flow into GRU
            lut = torch.full((self.n_nodes,), -1, dtype=torch.long, device=DEV)
            lut[nodes] = torch.arange(len(nodes), device=DEV)
            ms, md = upd[lut[src]], upd[lut[dst]]
            ts, td = self.tenc(t - self.last_t[src]), self.tenc(t - self.last_t[dst])
            z = self.emb(torch.cat([hx, ms, md, ts, td], -1))
        else:
            nodes, upd, ms, md = None, None, None, None
            z = self.emb(hx)
        return z, hx, nodes, upd, ms, md

    @torch.no_grad()
    def commit(self, src, dst, t, hx, nodes, upd):
        """Write updated memory and store this batch's events as pending messages."""
        if not self.use_memory:
            return
        self.memory[nodes] = upd.detach()
        upd_nodes = nodes[self.pend_other[nodes] >= 0]
        self.last_t[upd_nodes] = self.pend_t[upd_nodes]
        if self.pend_x is None:
            self.pend_x = torch.zeros(self.n_nodes, hx.shape[1], device=DEV)
        self.pend_other[nodes] = -1
        # last event per node wins (TGN "last" aggregator). Duplicate indices in a plain
        # index_put have no defined winner on GPU, so pick the last occurrence explicitly.
        nb = len(src)
        all_n = torch.cat([src, dst])
        all_o = torch.cat([dst, src])
        pos = torch.cat([2 * torch.arange(nb, device=DEV), 2 * torch.arange(nb, device=DEV) + 1])
        last = torch.full((self.n_nodes,), -1, dtype=torch.long, device=DEV)
        last.scatter_reduce_(0, all_n, pos, reduce="amax")
        keep = pos == last[all_n]
        ev = torch.cat([torch.arange(nb, device=DEV)] * 2)[keep]
        self.pend_other[all_n[keep]] = all_o[keep]
        self.pend_x[all_n[keep]] = hx.detach()[ev]
        self.pend_t[all_n[keep]] = t[ev]


# ------------------------------------------------------------------ train / score
def run_pass(model, idx, S, train, y=None, dom=None, w_lab=None, opt=None, args=None, collect=None):
    """One chronological pass over events `idx` (sorted by time)."""
    model.train(train)
    pos_w = torch.tensor([args.pos_weight], device=DEV)
    for b0 in range(0, len(idx), args.batch):
        bi = idx[b0:b0 + args.batch]
        src, dst = S["src"][bi], S["dst"][bi]
        t, x = S["t"][bi], S["x"][bi]
        with torch.set_grad_enabled(train):
            z, hx, nodes, upd, ms, md = model.forward_batch(src, dst, t, x)
            logit = model.head(z).squeeze(-1)
            if model.use_memory and not args.no_anomaly:
                neg = torch.randint(0, model.n_nodes, (len(bi),), device=DEV)
                mneg = model.memory[neg]
                lp = model.link(torch.cat([ms, md, hx], -1)).squeeze(-1)
                ln = model.link(torch.cat([ms, mneg, hx], -1)).squeeze(-1)
            if train:
                loss = torch.zeros((), device=DEV)
                lab = w_lab[bi]
                if lab.any():
                    loss = loss + F.binary_cross_entropy_with_logits(
                        logit[lab], y[bi][lab].float(), pos_weight=pos_w)
                if model.use_memory and not args.no_anomaly:
                    loss = loss + args.w_link * (F.binary_cross_entropy_with_logits(lp, torch.ones_like(lp))
                                                 + F.binary_cross_entropy_with_logits(ln, torch.zeros_like(ln)))
                if args.adv > 0 and args.uda and dom is not None:
                    d_logit = model.dom(GradReverse.apply(z, args.adv))
                    loss = loss + F.cross_entropy(d_logit, dom[bi])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        if collect is not None:
            collect["logit"].append(logit.detach().float().cpu().numpy())
            if model.use_memory and not args.no_anomaly:
                collect["anom"].append((1 - torch.sigmoid(lp)).detach().float().cpu().numpy())
            collect["idx"].append(bi.cpu().numpy())
        model.commit(src, dst, t, hx, nodes, upd)


def zscore(a, ref):
    return (a - ref.mean()) / (ref.std() + 1e-8)


def score_all(model, order, S, args):
    """Fresh memory, no gradients, chronological pass over `order`; returns per-event scores."""
    model.reset_state()
    col = {"logit": [], "anom": [], "idx": []}
    with torch.no_grad():
        run_pass(model, order, S, train=False, collect=col, args=args)
    idx = np.concatenate(col["idx"])
    out_l = np.full(S["n"], np.nan, np.float32)
    out_l[idx] = np.concatenate(col["logit"])
    out_a = np.full(S["n"], np.nan, np.float32)
    if col["anom"]:
        out_a[idx] = np.concatenate(col["anom"])
    return out_l, out_a


def fit_and_score(df, y, S, tr, te, proto, seed, args):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t_tr = df[TIME_COL].values[tr]
    if args.val == "time":
        cut = np.quantile(t_tr, 0.85)
        fit_i, val_i = tr[t_tr < cut], tr[t_tr >= cut]
    else:
        # grouped validation: 20% of the escalated INCIDENTS in train (all their rows)
        # plus 15% of non-escalated train rows. A time slice can hold no positives at
        # all (vpc_wave, vpc_to_asa), which makes epoch/lambda selection random.
        vr = np.random.RandomState(1000 + seed)
        inc = df["incident_ids"].iloc[tr].map(first_incident).values
        pos_inc = pd.unique(inc[(y[tr] == 1) & pd.notna(inc)])
        v_inc = set(vr.choice(pos_inc, max(1, int(0.2 * len(pos_inc))), replace=False)) if len(pos_inc) else set()
        is_val = np.array([i in v_inc for i in inc]) | ((y[tr] == 0) & (vr.rand(len(tr)) < 0.15))
        fit_i, val_i = tr[~is_val], tr[is_val]
    if y[fit_i].sum() == 0:
        fit_i, val_i = tr, tr
    print(f"    validation: {len(val_i):,} rows, {int(y[val_i].sum())} positives ({args.val})", flush=True)
    rng = np.random.RandomState(seed)
    fit_i = np.sort(rng.choice(fit_i, int(len(fit_i) * args.train_frac), replace=False))

    # training pool: labeled train events (+ unlabeled target events with --uda)
    pool = fit_i
    if args.uda:
        pool = np.union1d(fit_i, te)
    pool = pool[np.argsort(S["t_np"][pool], kind="mergesort")]
    w_lab = torch.zeros(S["n"], dtype=torch.bool, device=DEV)
    w_lab[torch.from_numpy(fit_i).to(DEV)] = True
    yt = torch.from_numpy(y).to(DEV)

    model = CtxGraph(S["n_nodes"], S["x"].shape[1], S["n_dom"], mem=args.mem, hid=args.hid,
                     use_memory=not args.no_memory).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    pool_t = torch.from_numpy(pool).to(DEV)

    # every pass (train and score) sees events in time order from a fresh memory;
    # scoring passes run over train+val+test so test memory is built from the past
    score_order = np.union1d(np.union1d(fit_i, val_i), te)
    score_order = score_order[np.argsort(S["t_np"][score_order], kind="mergesort")]
    score_order_t = torch.from_numpy(score_order).to(DEV)

    best = (-1, None, None)
    for ep in range(args.epochs):
        t0 = time.time()
        model.reset_state()
        run_pass(model, pool_t, S, train=True, y=yt, dom=S["dom"], w_lab=w_lab, opt=opt, args=args)
        lg, an = score_all(model, score_order_t, S, args)
        v_ap = average_precision_score(y[val_i], lg[val_i]) if y[val_i].sum() else 0.0
        print(f"    epoch {ep + 1}/{args.epochs}  val PR-AUC(logit)={v_ap:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if v_ap > best[0]:
            best = (v_ap, lg.copy(), an.copy())
    _, lg, an = best

    # combine supervised logit with anomaly score; lambda chosen on validation
    lam_best, ap_best = 0.0, -1
    if not np.isnan(an[val_i]).all():
        for lam in [0.0, 0.25, 0.5, 1.0, 2.0]:
            s_val = zscore(lg[val_i], lg[fit_i]) + lam * zscore(an[val_i], an[fit_i])
            ap = average_precision_score(y[val_i], s_val) if y[val_i].sum() else 0
            if ap > ap_best:
                lam_best, ap_best = lam, ap
    final = zscore(lg, lg[fit_i])
    if lam_best > 0:
        final = final + lam_best * zscore(an, an[fit_i])
    s_te, s_val = final[te], final[val_i]
    r = evaluate(y[te], s_te, thr=B.best_thr(y[val_i], s_val))
    if "pr_auc" in r:
        r["pr_auc_ci95"] = bootstrap_ci(y[te], s_te, n=200)
        r["base_rate"] = float(y[te].mean())
        r["lift"] = r["pr_auc"] / r["base_rate"]
        r["pr_auc_logit_only"] = average_precision_score(y[te], lg[te])
        if not np.isnan(an[te]).all():
            r["pr_auc_anomaly_only"] = average_precision_score(y[te], an[te])
    r["lambda"] = lam_best
    return r, s_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["small", "100m"], default="100m")
    ap.add_argument("--only", default="asa_temporal,vpc_wave")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--vpc-cut", default="2024-07-28")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mem", type=int, default=64)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--pos-weight", type=float, default=20.0)
    ap.add_argument("--w-link", type=float, default=0.2)
    ap.add_argument("--adv", type=float, default=0.3, help="gradient-reversal weight; 0 = off")
    ap.add_argument("--uda", action="store_true", help="add unlabeled test-source events to training")
    ap.add_argument("--train-frac", type=float, default=0.9)
    ap.add_argument("--val", choices=["grouped", "time"], default="grouped",
                    help="validation slice for epoch/lambda/threshold selection")
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--no-ctx", action="store_true")
    ap.add_argument("--no-anomaly", action="store_true")
    ap.add_argument("--text", choices=["template", "none"], default="template")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    path = DATA_DIR / "100m" / "rulehits.parquet" if args.data == "100m" else None
    df, y = make_target(load_signals(path), "escalation")
    df = df.reset_index(drop=True)
    counts = pd.read_parquet(DATA_DIR / "100m" / "minute_counts.parquet") if args.data == "100m" else None
    print(f"device={DEV}  rows={len(df):,} positives={y.sum():,}", flush=True)

    P = B.protocols(df, y, args.vpc_cut)
    protos = [p for p in args.only.split(",") if p]
    blocks_static = []
    if not args.no_ctx:
        blocks_static.append(context_features(df, counts).to_numpy(np.float32))
    if args.text == "template":
        blocks_static.append(template_bag(df))
    src, dst, n_nodes = build_nodes(df)
    dom_codes = pd.factorize(df["stream_name"].astype(str))[0]

    tag = args.tag or ("_".join(protos) + ("_uda" if args.uda else ""))
    out = RESULTS_DIR / "ctxgraph" / tag
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for proto in protos:
        tr, te = P[proto]
        # tabular vocabulary fitted on this protocol's training rows only
        fz = TabularFeaturizer().fit(df.iloc[tr])
        tab = fz.transform(df).to_numpy(np.float32)
        X = np.hstack([tab] + blocks_static)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        X = (X - mu) / sd
        S = {"n": len(df), "n_nodes": n_nodes, "n_dom": int(dom_codes.max() + 1),
             "src": torch.from_numpy(src).to(DEV), "dst": torch.from_numpy(dst).to(DEV),
             "t": torch.from_numpy((df[TIME_COL].values - df[TIME_COL].values.min()).astype(np.float32)).to(DEV),
             "t_np": df[TIME_COL].values, "x": torch.from_numpy(X.astype(np.float32)).to(DEV),
             "dom": torch.from_numpy(dom_codes).to(DEV)}
        for sd_ in seeds:
            print(f"\n== {proto} seed={sd_}  train {len(tr):,} ({y[tr].sum():,} pos)  test {len(te):,} ({y[te].sum():,} pos)", flush=True)
            r, s_te = fit_and_score(df, y, S, tr, te, proto, sd_, args)
            np.savez_compressed(out / f"scores_{proto}_{sd_}.npz", idx=te, score=s_te, y=y[te])
            rows.append({"protocol": proto, "seed": sd_, "model": "ctxgraph", **r})
            print(f"  -> PR-AUC={r.get('pr_auc', np.nan):.4f} (base {r.get('base_rate', np.nan):.4f}, "
                  f"lift {r.get('lift', np.nan):.1f}x) ROC={r.get('roc_auc', np.nan):.3f} "
                  f"R@1%FPR={r.get('recall@fpr1%', np.nan):.3f} | logit-only {r.get('pr_auc_logit_only', np.nan):.4f} "
                  f"anomaly-only {r.get('pr_auc_anomaly_only', np.nan):.4f} lambda={r['lambda']}", flush=True)
            pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)

    runs = pd.DataFrame(rows)
    summ = runs.groupby("protocol").agg(pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
                                        roc_mean=("roc_auc", "mean"), r1_mean=("recall@fpr1%", "mean"),
                                        base_rate=("base_rate", "first")).reset_index()
    summ["lift_mean"] = summ.pr_auc_mean / summ.base_rate
    summ.to_csv(out / "summary.csv", index=False)
    print("\n##### CtxGraph SUMMARY (mean ± std over seeds)")
    for _, r in summ.iterrows():
        print(f"  {r.protocol:15s} PR-AUC {r.pr_auc_mean:.4f} ± {r.pr_auc_std:.4f}  (lift {r.lift_mean:.1f}x)"
              f"  ROC {r.roc_mean:.3f}  R@1%FPR {r.r1_mean:.3f}")
    with open(out / "config.txt", "w") as fh:
        fh.write(str(vars(args)))
    print("\nsaved ->", out)


if __name__ == "__main__":
    main()
