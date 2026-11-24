"""Build a small synthetic file with the real Precinct6 schema, so the pipeline
can be smoke-tested without downloading anything. NOT for results."""
import json
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.RandomState(0)
N = 40_000
t0 = 1721992223  # 2024-07-26
orgs = rng.choice(["ORG-0004", "ORG-0005", "ORG-0003", "ORG-0001"], N, p=[.5, .29, .2, .01])
ts = t0 + np.sort(rng.randint(0, 6 * 86400, N)).astype(float)
lab = rng.choice(["benign", "suspicious", "malicious"], N, p=[.93, .05, .02])
lab[orgs == "ORG-0001"] = "benign"
streams = ["aws_cloudtrail_events", "microsoft-windows-security-auditing", "aws_vpc_flow_log", "cisco_asa", "dnsmasq"]
mt = ["4624", "4625", "AssumeRole", "firewall_action", "account_logon"]
inc = [f"inc-{i}" for i in range(80)]
df = pd.DataFrame({
    "timestamp": ts, "event_time": ts - 2, "timestamp_source": "artifact.timeuuid", "origin": "live",
    "org_id": orgs, "artifact_id": [f"a{i}" for i in range(N)],
    "message_type": rng.choice(mt, N), "stream_name": rng.choice(streams, N), "pipeline": "p",
    "src_ip": [f"10.0.{rng.randint(5)}.{rng.randint(50)}" for _ in range(N)],
    "dst_ip": [f"10.1.0.{rng.randint(30)}" for _ in range(N)],
    "src_port": rng.randint(1, 65000, N).astype(str), "dst_port": rng.choice(["443", "22", "3389", "53"], N),
    "protocol": rng.choice(["6", "17"], N), "src_host": [f"HOST-{rng.randint(200):04d}" for _ in range(N)],
    "dst_host": [f"HOST-{rng.randint(200):04d}" for _ in range(N)],
    "username": [f"USER-{rng.randint(80):04d}" for _ in range(N)],
    "action": rng.choice(["permit", "block", "logon"], N), "severity": rng.choice(["1", "3", "5"], N),
    "vendor_code": rng.choice(["x", "y"], N),
    "message_sanitized": [f"event {m} by USER-{rng.randint(80):04d} on HOST-{rng.randint(200):04d} status {rng.choice(['ok','fail'])}"
                          for m in rng.choice(mt, N)],
    "label_binary": lab,
})
mal = df.label_binary == "malicious"
df["label_confidence"] = np.where(mal, .8, np.where(df.label_binary == "suspicious", .6, .5))
df["suspicion_score"] = np.where(mal, .25, 0.0)
df["incident_ids"] = [json.dumps([rng.choice(inc)]) if m else "[]" for m in mal]
df["matched_rules"] = [json.dumps(["r1"]) if l != "benign" else "[]" for l in df.label_binary]
for c in ["attack_techniques", "attack_tactics", "defense_techniques", "set_roles"]:
    df[c] = np.where(mal, '["T1078"]', "[]")
df["mo_name"] = np.where(mal, "Data Theft", "")
df["lifecycle_stage"] = np.where(mal, "initial-compromise", "")
df["disposition"] = np.where(mal, "Disrupted", "")
df["status_name"] = df["disposition"]
df["disposition_category"] = np.where(mal, "automated", "")
df["is_false_positive"] = False
df["product_name"] = rng.choice(["Windows Logs", "ASA Firewall"], N)
df["vendor_name"] = rng.choice(["Microsoft", "Cisco"], N)
# make malicious weakly learnable
df.loc[mal, "message_sanitized"] = df.loc[mal, "message_sanitized"] + " privilege escalation"
out = Path("data/signals"); out.mkdir(parents=True, exist_ok=True)
df.to_parquet(out / "signals.parquet", index=False)
print("mock written:", out / "signals.parquet", df.shape, df.label_binary.value_counts().to_dict())
