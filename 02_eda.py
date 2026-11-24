"""Step 2 - exploratory analysis. Writes tables to results/eda/ ."""
import json
import pandas as pd
from common import load_signals
from config import RESULTS_DIR, ORG_COL, TIME_COL

out = RESULTS_DIR / "eda"
out.mkdir(parents=True, exist_ok=True)
df = load_signals()
print("rows:", len(df), "| columns:", len(df.columns))

tabs = {
    "label_overall": df.label_binary.value_counts().to_frame("rows"),
    "label_by_org": pd.crosstab(df[ORG_COL], df.label_binary),
    "label_by_day": pd.crosstab(pd.to_datetime(df[TIME_COL], unit="s").dt.date, df.label_binary),
    "label_by_stream": pd.crosstab(df.stream_name, df.label_binary).sort_values("malicious", ascending=False).head(30),
    "label_by_msgtype": pd.crosstab(df.message_type, df.label_binary).sort_values("malicious", ascending=False).head(30),
    "disposition_malicious": df[df.label_binary == "malicious"].disposition.value_counts().to_frame("rows"),
    "missing_rate": df.isna().mean().sort_values(ascending=False).to_frame("missing"),
}
mal = df[df.label_binary == "malicious"]
if "incident_ids" in df:
    inc = mal.incident_ids.map(lambda s: json.loads(s) if isinstance(s, str) else []).explode()
    tabs["rows_per_incident"] = inc.value_counts().describe().to_frame("rows_per_incident")
    tabs["mo_name_malicious"] = mal.mo_name.value_counts().to_frame("rows")

for name, t in tabs.items():
    t.to_csv(out / f"{name}.csv")
    print(f"\n== {name}\n", t.head(15).to_string())

# Shortcut check: how many streams / message types are (almost) pure in one class?
pure = (pd.crosstab(df.stream_name, df.label_binary, normalize="index").max(axis=1) > 0.99).mean()
print(f"\nShare of stream_name values that are >99% one label: {pure:.1%}")
