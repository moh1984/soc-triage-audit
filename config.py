"""Shared settings for the Precinct6 research pipeline."""
from pathlib import Path

REPO_ID = "witfoo/precinct6-cybersecurity"      # small version (~2M live signals)
REVISION = "v2.1.0"                             # pin the version for reproducibility
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
SEED = 42

# Columns that are produced by WitFoo's own labelling/correlation engine.
# Any of them in the feature set leaks the label (the model learns the rules,
# not the attack). They are kept only for the leakage audit.
LEAKY_COLS = [
    "label_binary", "label_confidence", "attack_techniques", "attack_tactics",
    "defense_techniques", "mo_name", "suspicion_score", "lifecycle_stage",
    "disposition", "disposition_category", "is_false_positive", "status_name",
    "incident_ids", "matched_rules", "set_roles",
]
# Bookkeeping columns: not features (ids / provenance of the row itself).
META_COLS = ["artifact_id", "origin", "timestamp_source"]

# Raw observable fields a deployed detector would actually see.
CATEGORICAL = [
    "message_type", "stream_name", "pipeline", "protocol", "action",
    "severity", "vendor_code", "product_name", "vendor_name",
]
HIGH_CARD = ["src_ip", "dst_ip", "src_host", "dst_host", "username"]
PORTS = ["src_port", "dst_port"]
TEXT_COL = "message_sanitized"
TIME_COL = "timestamp"
ORG_COL = "org_id"
