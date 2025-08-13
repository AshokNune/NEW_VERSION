#!/usr/bin/env python3
import argparse, os, time, math, hashlib, gzip, pickle
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from logger_util import start_logger, get_worker_logger, stop_logger
from feedback_utils import normalize_snippet, build_cache_from_feedback, load_cache

def ensure_cache(feedback_csv: str, cache_path: str, logger) -> Dict[str,str]:
    if not os.path.exists(cache_path):
        logger.info("Feedback cache not found; building cache...")
        mapping, n = build_cache_from_feedback(feedback_csv, cache_path)
        logger.info(f"Built cache with {n} entries")
        return mapping
    # Check mtime
    if os.path.getmtime(feedback_csv) > os.path.getmtime(cache_path):
        logger.info("Feedback CSV newer than cache; rebuilding cache...")
        mapping, n = build_cache_from_feedback(feedback_csv, cache_path)
        logger.info(f"Rebuilt cache with {n} entries")
        return mapping
    else:
        logger.info("Loading feedback cache...")
        mapping = load_cache(cache_path)
        logger.info(f"Loaded cache with {len(mapping)} entries")
        return mapping

def hash_norm(s: str) -> str:
    import hashlib
    n = normalize_snippet(s)
    return hashlib.sha256(n.encode('utf-8')).hexdigest()

def apply_to_file(path: Path, out_dir: Path, mapping: Dict[str,str], logger) -> Tuple[str,int,int]:
    # Read input (expect Snippets_Normalized or Snippets_with_duplicates)
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    if "Snippets_Normalized" in df.columns and df["Snippets_Normalized"].notna().any():
        src_col = "Snippets_Normalized"
    elif "Snippets_with_duplicates" in df.columns:
        # normalize them
        df["Snippets_Normalized"] = df["Snippets_with_duplicates"].fillna("").astype(str).apply(
            lambda s: ";".join(sorted(set(normalize_snippet(x.strip()) for x in s.split(";") if x.strip())))
        )
        src_col = "Snippets_Normalized"
    else:
        raise ValueError(f"{path.name}: need Snippets_Normalized or Snippets_with_duplicates")

    # For each row, apply mapping
    applied = []
    for _, row in df.iterrows():
        sn = str(row.get(src_col, "")) if pd.notna(row.get(src_col, "")) else ""
        labels = []
        for piece in [x.strip() for x in sn.split(";") if x.strip()]:
            h = hashlib.sha256(piece.encode('utf-8')).hexdigest()
            if h in mapping:
                labels.append(mapping[h])
        applied.append(",".join(sorted(set(labels))) if labels else "")
    df["Applied_Classification"] = applied

    out_path = out_dir / (path.stem + "_with_feedback.xlsx")
    df.to_excel(out_path, index=False)
    logger.info(f"Wrote {out_path.name}")
    return path.name, len(df), df["Applied_Classification"].astype(bool).sum()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="single file with snippets")
    ap.add_argument("--input_folder", help="folder with files")
    ap.add_argument("--output_folder", default="output")
    ap.add_argument("--feedback_csv", required=True)
    ap.add_argument("--feedback_cache", default="feedback.pkl.gz")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    out_dir = Path(args.output_folder); out_dir.mkdir(parents=True, exist_ok=True)
    log_q, listener = start_logger("logs")
    logger = get_worker_logger(log_q, name="apply.worker")

    mapping = ensure_cache(args.feedback_csv, args.feedback_cache, logger)

    inp_files = []
    if args.input:
        p = Path(args.input); 
        if not p.exists(): raise FileNotFoundError(p)
        inp_files = [p]
    elif args.input_folder:
        p = Path(args.input_folder)
        if not p.exists(): raise FileNotFoundError(p)
        inp_files = sorted([x for x in p.iterdir() if x.suffix.lower() in (".csv",".xlsx")])
    else:
        raise SystemExit("Provide --input or --input_folder")

    jobs = os.cpu_count() if args.jobs in (-1,0) else max(1, args.jobs)

    if jobs == 1:
        for f in inp_files:
            apply_to_file(f, out_dir, mapping, logger)
    else:
        Parallel(n_jobs=jobs, prefer="processes")(
            delayed(apply_to_file)(f, out_dir, mapping, logger) for f in inp_files
        )

    stop_logger(log_q, listener)

if __name__ == "__main__":
    main()
