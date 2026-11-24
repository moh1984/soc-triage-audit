"""Step 1 - download the files we need from Hugging Face (pinned revision).

Usage:  python 01_download.py            # signals + metadata + rule catalog (~1-2 GB)
        python 01_download.py --graph    # also the provenance graph + incidents (larger)
"""
import argparse
from huggingface_hub import hf_hub_download
from config import REPO_ID, REVISION, DATA_DIR

CORE = [
    "signals/signals.parquet",
    "signals/incident_signals.parquet",
    "signals/metadata.json",
    "reference/lead_rules_catalog.json",
]
GRAPH = [
    "graph/nodes.jsonl",
    "graph/metadata.json",
    "graph/incidents.jsonl",
    "graph/attack_reports.jsonl",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", action="store_true", help="also download graph files")
    args = ap.parse_args()
    files = CORE + (GRAPH if args.graph else [])
    DATA_DIR.mkdir(exist_ok=True)
    for f in files:
        try:
            p = hf_hub_download(REPO_ID, f, repo_type="dataset", revision=REVISION,
                                local_dir=DATA_DIR)
            print("ok  ", p)
        except Exception as e:  # file names can change between versions
            print("FAIL", f, "->", e)
    if args.graph:
        print("\nEdges are sharded; list them with:\n"
              "  from huggingface_hub import list_repo_files\n"
              f"  [f for f in list_repo_files('{REPO_ID}', repo_type='dataset', revision='{REVISION}') if f.startswith('graph/edges')]")


if __name__ == "__main__":
    main()
