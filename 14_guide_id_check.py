"""Step 14 - GUIDE: are identifiers shared ACROSS organisations? (no model training, ~5 min)

Why: in the org_holdout protocol, the 'all' feature set (with IncidentId, AlertId, Id, OrgId
and hashed entity ids) ranks incidents of UNSEEN organisations better than 'content'.
An identifier can only help there if its values are not local to one organisation.
For every identifier column we report:
  - share of values that occur in more than one organisation
  - share of rows carrying such a value
  - TP rate of rows whose value is shared vs not (does sharing carry label information?)
  - for IncidentId and AlertId: Spearman correlation of the id with the TP label and with
    time, inside organisations (ids that grow with time can leak the period)

Usage:  python 14_guide_id_check.py --dir guide
Output: results/guide_id_check/id_sharing.csv
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from config import RESULTS_DIR

spec = importlib.util.spec_from_file_location("g12", Path(__file__).with_name("12_guide_audit.py"))
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)

IDS = G.ID_COLS + G.ENTITY_IDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="guide")
    args = ap.parse_args()
    out = RESULTS_DIR / "guide_id_check"
    out.mkdir(parents=True, exist_ok=True)

    df, y = G.add_derived(G.load_all(args.dir))
    org = df["OrgId"].values
    rows = []
    for c in IDS:
        v = G._keys(df[c])
        n_org = pd.DataFrame({"v": v, "o": org}).groupby("v")["o"].nunique()
        shared_vals = n_org.index[n_org > 1]
        shared = pd.Series(v).isin(shared_vals).values
        top = pd.Series(v).value_counts()
        r = {"column": c, "n_values": int(len(n_org)),
             "share_values_multi_org": float((n_org > 1).mean()),
             "share_rows_multi_org_value": float(shared.mean()),
             "top_value": str(top.index[0]), "top_value_row_share": float(top.iloc[0] / len(v)),
             "tp_rate_shared": float(y[shared].mean()) if shared.any() else np.nan,
             "tp_rate_not_shared": float(y[~shared].mean()) if (~shared).any() else np.nan}
        rows.append(r)
        print(f"  {c:18s} values={r['n_values']:>9,}  multi-org values={r['share_values_multi_org']:.2%}  "
              f"rows={r['share_rows_multi_org_value']:.2%}  top={r['top_value']} ({r['top_value_row_share']:.1%})  "
              f"TP shared/not={r['tp_rate_shared']:.3f}/{r['tp_rate_not_shared']:.3f}", flush=True)

    print("\n== ids vs label and time, within organisations (Spearman, median over orgs with >=200 rows)")
    t = df["Timestamp"].values.astype("int64")
    for c in ["IncidentId", "AlertId", "Id"]:
        d = pd.DataFrame({"o": org, "id": df[c].values, "y": y, "t": t})
        big = d.groupby("o").filter(lambda g: len(g) >= 200)
        res = big.groupby("o").apply(lambda g: pd.Series({
            "rho_label": g["id"].corr(g["y"], method="spearman"),
            "rho_time": g["id"].corr(g["t"], method="spearman")}))
        med = res.median()
        print(f"  {c:10s} rho(id, TP)={med['rho_label']:+.3f}  rho(id, time)={med['rho_time']:+.3f}  "
              f"over {len(res):,} orgs")
        rows.append({"column": c + "_within_org", "rho_label_median": float(med["rho_label"]),
                     "rho_time_median": float(med["rho_time"]), "orgs": int(len(res))})
    pd.DataFrame(rows).to_csv(out / "id_sharing.csv", index=False)
    print("\nsaved ->", out)


if __name__ == "__main__":
    main()
