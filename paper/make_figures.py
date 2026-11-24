"""Figures 1-5 of the paper, drawn from the result files of the experiments.

Run from this folder:  python make_figures.py   (writes fig/fig1..fig5)

Sources (paths relative to the repository root):
  Fig. 1  results/eda/label_by_stream.csv (released subset, 02_eda.py)
          results/extract/escalation_by_group.csv (full capture, 05b_summarize_100m.py)
  Fig. 2  results/guide_robust/D_org_purity.csv (13_guide_robust.py, section D)
  Fig. 3  results/diagnostics/q2_daily_escalations_by_stream.csv (06_diagnose_temporal.py)
  Fig. 4  logs/precinct6_cluster_output.txt, SUMMARY block (07_repeat_seeds.py --tag cluster;
          the console log is the archived record of that run) and, for "GBDT all",
          results/seeds/*_gbdt_all_rerun/summary.csv (07_repeat_seeds.py --models lgbm_all)
  Fig. 5  results/guide_v2/summary.csv (15_guide_v2.py)
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "axes.titlesize": 9.5})
OUT = Path(__file__).resolve().parent / "fig"
OUT.mkdir(exist_ok=True)
DARK = "#1f3b57"; MID = "#6f8fae"; LIGHT = "#c9d6e3"; ACC = "#c0504d"; GREY = "#8c8c8c"
ASA, VPC = "cisco_asa", "aws_vpc_flow_log"

# ---------- Fig. 1: escalation rate, released subset vs full capture
small = pd.read_csv(R / "eda" / "label_by_stream.csv").set_index("stream_name")
small = small[(small.malicious + small.suspicious) > 0]
def rate_small(rows):
    return 100 * rows.malicious.sum() / (rows.malicious.sum() + rows.suspicious.sum())
full = pd.read_csv(R / "extract" / "escalation_by_group.csv").set_index("group")
cats = ["All rule-matched", VPC, ASA]
s_vals = [rate_small(small), rate_small(small.loc[[VPC]]), rate_small(small.loc[[ASA]])]
f_vals = [100 * full.loc[g, "escalation_rate"] for g in ["Total", VPC, ASA]]
x = np.arange(3); w = 0.36
fig, ax = plt.subplots(figsize=(4.6, 2.6))
b1 = ax.bar(x - w / 2, s_vals, w, color=MID, label="2M subset (released small version)")
b2 = ax.bar(x + w / 2, f_vals, w, color=DARK, label="Full capture (114M events)")
for b, v in zip(list(b1) + list(b2), s_vals + f_vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.2f}%", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(cats)
ax.set_ylabel("Escalation rate (%)"); ax.set_ylim(0, 30)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
plt.tight_layout(); plt.savefig(OUT / "fig1_sampling.png"); plt.close()

# ---------- Fig. 2: GUIDE organization label purity
pur = pd.read_csv(R / "guide_robust" / "D_org_purity.csv")
grp = ["TP < 5%", "5–20%", "20–80%", "80–95%", "TP > 95%"]
orgs = list(100 * pur.orgs_share); rows = list(100 * pur.rows_share)
n_orgs, n_rows = int(pur.orgs.sum()), pur.rows.sum()
x = np.arange(len(grp)); w = 0.36
fig, ax = plt.subplots(figsize=(4.6, 2.5))
b1 = ax.bar(x - w / 2, orgs, w, color=MID, label=f"share of organizations (n = {n_orgs:,})")
b2 = ax.bar(x + w / 2, rows, w, color=DARK, label=f"share of evidence rows ({n_rows / 1e6:.1f}M)")
for b, v in zip(list(b1) + list(b2), orgs + rows):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(grp)
ax.set_xlabel("Share of the organization's evidence graded TruePositive")
ax.set_ylabel("%"); ax.set_ylim(0, 75)
ax.legend(frameon=False, fontsize=7.5, loc="upper center")
plt.tight_layout(); plt.savefig(OUT / "fig2_org_purity.png"); plt.close()

# ---------- Fig. 3: escalated events per day and source
d = pd.read_csv(R / "diagnostics" / "q2_daily_escalations_by_stream.csv")
days = pd.to_datetime(d["_day"]).dt.strftime("%-d %b").tolist()
asa, vpc = d[ASA].to_numpy(), d[VPC].to_numpy()
other = d["ALL_escalated"].to_numpy() - asa - vpc
x = np.arange(len(days))
fig, ax = plt.subplots(figsize=(5.2, 2.7))
ax.bar(x, asa, 0.6, color=MID, label="cisco_asa (ORG-0005)")
ax.bar(x, vpc, 0.6, bottom=asa, color=DARK, label="aws_vpc_flow_log (ORG-0004)")
ax.bar(x, other, 0.6, bottom=asa + vpc, color=LIGHT, label="other sources")
ax.axvline(0.76, color=ACC, ls="--", lw=1)          # 27 Jul 06:19 UTC (Q1 of 06_diagnose_temporal.py)
ax.text(0.82, 3650, "naive global cut\n(27 Jul 06:19 UTC)", color=ACC, fontsize=7, va="top")
ax.axvline(1.5, color=GREY, ls=":", lw=1)           # --vpc-cut 2024-07-28
ax.text(1.56, 2600, "vpc_wave cut\n(28 Jul 00:00)", color="#555555", fontsize=7, va="top")
ax.set_xticks(x); ax.set_xticklabels(days)
ax.set_ylabel("Escalated events per day"); ax.set_ylim(0, 3800)
ax.legend(frameon=False, fontsize=7, loc="upper right")
plt.tight_layout(); plt.savefig(OUT / "fig3_timeline.png"); plt.close()

# ---------- Fig. 4: benchmark lift
log = (ROOT / "logs" / "precinct6_cluster_output.txt").read_text(encoding="utf-8")
summary = log.split("##### SUMMARY")[1].split("#####")[0]
res, base, proto = {}, {}, None
for line in summary.splitlines():
    m = re.match(r"^(\w+)\s+\(base rate ([\d.]+)\)", line)
    if m:
        proto = m.group(1); base[proto] = float(m.group(2)); continue
    m = re.match(r"^\s+(\S+)\s+PR-AUC ([\d.]+) ± ([\d.]+)", line)
    if m and proto:
        res[(proto, m.group(1))] = (float(m.group(2)), float(m.group(3)))
ga = next((R / "seeds").glob("*_gbdt_all_rerun")) / "summary.csv"
for _, r in pd.read_csv(ga).iterrows():
    res[(r.protocol, "lgbm_all")] = (r.pr_auc_mean, r.pr_auc_std)
protos = [("asa_temporal", "ASA temporal"), ("vpc_wave", "VPC wave-to-wave"),
          ("asa_to_vpc", "ASA → VPC"), ("vpc_to_asa", "VPC → ASA")]
models = [("logreg_tab", "LR tab"), ("logreg_tab+ctx", "LR tab+ctx"), ("lgbm_tab", "GBDT tab"),
          ("lgbm_tab+ctx", "GBDT tab+ctx"), ("logreg_text", "LR raw text"), ("lgbm_all", "GBDT all"),
          ("iforest_ctx", "iForest ctx")]
cols = ["#d9d9d9", "#1f3b57", "#bfbfbf", "#6f8fae", "#e8c1c0", "#9fb4c8", "#c0504d"]
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.6), sharey=False)
for ax, (p, title) in zip(axes, protos):
    vals = [res[(p, m)][0] / base[p] for m, _ in models]
    errs = [res[(p, m)][1] / base[p] for m, _ in models]
    ax.bar(range(len(models)), vals, yerr=errs, color=cols, capsize=1.5, error_kw={"lw": 0.6})
    ax.axhline(1, color="k", lw=0.6, ls="--")
    ax.set_title(title); ax.set_xticks([])
    if p == "asa_temporal":
        ax.set_ylabel("PR-AUC / base rate (lift)")
fig.legend([Patch(color=c) for c in cols], [n for _, n in models], ncol=7, frameon=False,
           fontsize=6.6, loc="lower center", bbox_to_anchor=(0.5, -0.01))
plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig(OUT / "fig4_benchmark.png"); plt.close()

# ---------- Fig. 5: GUIDE incident-level ROC-AUC by protocol and feature set
g = pd.read_csv(R / "guide_v2" / "summary.csv", header=[0, 1], index_col=[0, 1])
g = g["inc_roc_auc"]
gprotos = [("official", "Official\n(released split)"),
           ("incident_temporal", "Temporal\n(by incident start,\n10 Jun 2024)"),
           ("org_holdout", "Organization\nheld out")]
fsets = [("org_prior", "Org prior only", "#8c8c8c"), ("content", "Content", "#c9d6e3"),
         ("clean", "Clean (+ entity ids)", "#6f8fae"),
         ("all", "All (+ OrgId, incident/alert ids)", "#1f3b57")]
x = np.arange(len(gprotos)); w = 0.2
fig, ax = plt.subplots(figsize=(5.6, 2.4))
for k, (fs, lab, c) in enumerate(fsets):
    m = [g.loc[(p, fs), "mean"] for p, _ in gprotos]
    e = [g.loc[(p, fs), "std"] for p, _ in gprotos]
    bars = ax.bar(x + (k - 1.5) * w, m, w * 0.9, yerr=e, color=c, edgecolor="#333333", linewidth=0.4,
                  capsize=1.5, error_kw={"lw": 0.6}, label=lab)
    for b, v, er in zip(bars, m, e):
        ax.text(b.get_x() + b.get_width() / 2, v + er + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
ax.axhline(0.5, color="k", lw=0.6, ls="--")
ax.set_xticks(x); ax.set_xticklabels([t for _, t in gprotos], fontsize=7.5)
ax.set_ylabel("Incident-level ROC-AUC"); ax.set_ylim(0.4, 1.06)
ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
plt.tight_layout(); plt.savefig(OUT / "fig5_guide.png"); plt.close()
print("figures written to", OUT)
