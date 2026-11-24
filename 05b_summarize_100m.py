"""Step 3b - summary of the full-capture extract (Table 1 and the full-capture bars of Fig. 1).

Reads data/100m/rulehits.parquet written by 05_extract_100m.py and writes
results/extract/escalation_by_group.csv: rule-matched events, malicious (escalated) and
suspicious counts and the escalation rate, by organization, by log source, and in total.

Usage: python 05b_summarize_100m.py
"""
import pandas as pd
from common import load_signals, make_target
from config import DATA_DIR, RESULTS_DIR, ORG_COL


def table(df, y, col):
    g = df.assign(_y=y).groupby(col)["_y"].agg(malicious="sum", events="size")
    g["suspicious"] = g.events - g.malicious
    g["escalation_rate"] = g.malicious / g.events
    return g.reset_index().rename(columns={col: "group"}).assign(by=col)


def main():
    df, y = make_target(load_signals(DATA_DIR / "100m" / "rulehits.parquet"), "escalation")
    out = RESULTS_DIR / "extract"
    out.mkdir(parents=True, exist_ok=True)
    tot = pd.DataFrame([{"group": "Total", "malicious": int(y.sum()), "events": len(y),
                         "suspicious": int(len(y) - y.sum()), "escalation_rate": y.mean(), "by": "all"}])
    res = pd.concat([table(df, y, ORG_COL), table(df, y, "stream_name"), tot], ignore_index=True)
    res = res[["by", "group", "malicious", "suspicious", "events", "escalation_rate"]]
    res.to_csv(out / "escalation_by_group.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
