"""Step 5 - build an unbiased escalation dataset from the FULL capture (100M version).

Why: the small subset keeps suspicious rows only near incidents (+/-5 min window)
or in a 0.37% fill, so ~83% of suspicious events are missing and escalation looks
easier than it is. Here we stream every shard of witfoo/precinct6-cybersecurity-100m
and keep:
  1. every rule-matched row (label_binary in {malicious, suspicious}), all columns
  2. TRUE event counts per (org, minute) and per (org, minute, stream) computed
     from ALL rows, benign included -> an honest context/density signal
Each shard is deleted after processing, so disk use stays ~1 shard (~270 MB).

Usage:  python 05_extract_100m.py              (all shards, ~15.7 GB streamed)
        python 05_extract_100m.py --max-shards 3   (quick test)
Output: data/100m/rulehits.parquet, data/100m/minute_counts.parquet
"""
import argparse
import os
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from config import DATA_DIR

REPO = "witfoo/precinct6-cybersecurity-100m"
OUT = DATA_DIR / "100m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-shards", type=int, default=0)
    ap.add_argument("--revision", default="ef20229d046b03c02061c396cd66b711745021ae",
                    help="commit used in the paper (2026-09-22); pass 'main' for the latest")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = DATA_DIR / "_shards_tmp"   # outside OUT, so OUT can live on Google Drive

    files = sorted(f for f in list_repo_files(REPO, repo_type="dataset", revision=args.revision)
                   if f.startswith("signals/") and f.endswith(".parquet")
                   and "incident" not in f)
    if args.max_shards:
        files = files[: args.max_shards]
    print(f"{len(files)} signal shards to stream")

    hits, counts = [], []
    done_log = OUT / "done.txt"
    done = set(done_log.read_text().split()) if done_log.exists() else set()
    for i, f in enumerate(files, 1):
        part_hits = OUT / f"hits_{i:05d}.parquet"
        part_cnt = OUT / f"cnt_{i:05d}.parquet"
        if f in done:                       # resume after a Colab disconnect
            continue
        p = hf_hub_download(REPO, f, repo_type="dataset", revision=args.revision, local_dir=tmp)
        t = pq.read_table(p)
        df = t.to_pandas()
        # 1. rule-matched rows
        df[df.label_binary != "benign"].to_parquet(part_hits, index=False)
        # 2. honest minute counts from ALL rows
        minute = (df["timestamp"] // 60).astype("int64")
        c = (df.assign(minute=minute)
               .groupby(["org_id", "minute", "stream_name"], observed=True)
               .size().rename("n").reset_index())
        c.to_parquet(part_cnt, index=False)
        os.remove(p)
        with open(done_log, "a") as fh:
            fh.write(f + "\n")
        n_hit = (df.label_binary != "benign").sum()
        print(f"[{i}/{len(files)}] {f}: rows={len(df):,} rule-hits={n_hit:,}", flush=True)
        del df, t

    # merge parts
    hit_parts = sorted(OUT.glob("hits_*.parquet"))
    cnt_parts = sorted(OUT.glob("cnt_*.parquet"))
    rulehits = pd.concat([pd.read_parquet(p) for p in hit_parts], ignore_index=True)
    rulehits = rulehits.sort_values("timestamp").reset_index(drop=True)
    rulehits.to_parquet(OUT / "rulehits.parquet", index=False)
    cnt = pd.concat([pd.read_parquet(p) for p in cnt_parts], ignore_index=True)
    cnt = cnt.groupby(["org_id", "minute", "stream_name"], observed=True)["n"].sum().reset_index()
    cnt.to_parquet(OUT / "minute_counts.parquet", index=False)

    print("\nrule-hit rows:", len(rulehits))
    print(rulehits.label_binary.value_counts().to_string())
    print("\nby org:\n", pd.crosstab(rulehits.org_id, rulehits.label_binary).to_string())
    print("\nby stream (top 15):\n",
          pd.crosstab(rulehits.stream_name, rulehits.label_binary)
            .sort_values("malicious", ascending=False).head(15).to_string())
    print("\nsaved ->", OUT / "rulehits.parquet", "and", OUT / "minute_counts.parquet")


if __name__ == "__main__":
    main()
