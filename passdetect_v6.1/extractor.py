#!/usr/bin/env python3
import argparse, os, re, shutil, time, math
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from colorama import Fore, Style
from logger_util import start_logger, get_worker_logger, stop_logger
from feedback_utils import normalize_snippet

IGNORE_DEFAULT = {"error","err"}

def load_custom_keywords(csv_path: str) -> List[str]:
    if not csv_path or not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path, header=None)
    kws = []
    for v in df.iloc[:,0].tolist():
        s = str(v).strip()
        if s:
            kws.append(s)
    # dedupe, keep lowercase for matching but preserve original for output? We'll store lowercase and output lowercase
    seen=set(); out=[]
    for k in kws:
        kl = k.lower()
        if kl not in seen:
            seen.add(kl); out.append(kl)
    return out

def flexible_keyword_regex(keyword: str) -> re.Pattern:
    # Build flexible pattern allowing non-word separators between chars
    parts = []
    for ch in keyword:
        parts.append(re.escape(ch) + r"[\W_]*")
    body = "".join(parts)
    return re.compile(body, re.IGNORECASE)

def find_snippets(text: str, row_kws: List[str], custom_kws: List[str], after_chars: int) -> Tuple[List[str], List[str], List[str]]:
    if not isinstance(text, str) or not text:
        return [], [], []
    snippets = []
    matched_custom = set()
    all_kws = [k.strip() for k in row_kws if k.strip()]
    # Search row-specific keywords
    for kw in all_kws:
        pat = flexible_keyword_regex(kw)
        for m in pat.finditer(text):
            end = m.end()
            snippet = text[m.start(): min(len(text), end + after_chars)]
            snippets.append(snippet)
    # Custom keywords simple case-insensitive containment
    low = text.lower()
    for ckw in custom_kws:
        if ckw and ckw in low:
            matched_custom.add(ckw)
    # Dedup snippets for the "without duplicates" version
    uniq = []
    seen = set()
    for s in snippets:
        k = s.lower()
        if k not in seen:
            seen.add(k); uniq.append(s)
    return snippets, uniq, sorted(matched_custom)

def should_ignore(sn: str, ignore_terms: set) -> bool:
    # Ignore if snippet normalized collapses to only placeholders or empty
    n = normalize_snippet(sn)
    if not n:
        return True
    # if purely placeholder like <date> or <num> etc.
    if n in {"<date>", "<time>", "<datetime>", "<num>", "<ip>", "<url>", "<email>"}:
        return True
    low = n.lower()
    for t in ignore_terms:
        if t in low:
            return True
    return False

def heuristic_tag(sn: str, keywords_all: List[str]) -> str:
    low = sn.lower()
    for kw in keywords_all:
        k = re.escape(kw.lower())
        if re.search(k + r'\s*[=:>]\s*', low) or re.search(k + r'\s+valuer\s*=\s*', low):
            return "PotentialTrue Positive"
    return ""

def process_file(path: Path, out_dir: Path, completed_dir: Path, summary_dir: Path,
                 custom_kws: List[str], after_chars: int, ignore_terms: set,
                 log_q, show_progress: bool) -> Tuple[str, int, float]:
    logger = get_worker_logger(log_q)
    t0 = time.time()
    # Read file
    ext = path.suffix.lower()
    if ext == ".xlsx":
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # Ensure columns
    if "Text" not in df.columns:
        raise ValueError(f"{path.name}: missing 'Text' column")
    if "Keyword" not in df.columns:
        df["Keyword"] = ""

    total_rows = len(df)
    pbar = tqdm(total=total_rows, desc=f"{path.name}", disable=not show_progress)
    rows = []
    total_snip_counts = {}

    for idx, row in df.iterrows():
        text = str(row.get("Text", "")) if pd.notna(row.get("Text", "")) else ""
        row_kw_cell = str(row.get("Keyword","")) if pd.notna(row.get("Keyword","")) else ""
        row_kws = [k.strip() for k in row_kw_cell.split(";") if k.strip()]
        snippets, snippets_uniq, matched_custom = find_snippets(text, row_kws, custom_kws, after_chars)

        # Filter ignored snippets
        snippets = [s for s in snippets if not should_ignore(s, ignore_terms)]
        snippets_uniq = []
        seen=set()
        for s in snippets:
            kl = s.lower()
            if kl not in seen:
                seen.add(kl); snippets_uniq.append(s)

        # Normalized unique snippets for per-row
        norm_uniq = []
        seen_norm = set()
        for s in snippets_uniq:
            n = normalize_snippet(s)
            if n and n.lower() not in seen_norm:
                seen_norm.add(n.lower())
                norm_uniq.append(n)
                total_snip_counts[n] = total_snip_counts.get(n, 0) + 1

        cust_col = "No" if not matched_custom else ",".join(matched_custom)
        heur = heuristic_tag(" ".join(snippets_uniq), row_kws + custom_kws)

        rows.append({
            **row.to_dict(),
            "Snippets_with_duplicates": ";".join(snippets) if snippets else "",
            "Snippets_without_duplicates": ";".join(snippets_uniq) if snippets_uniq else "",
            "Snippet_count": len(snippets_uniq),
            "CustomKeywordFound": cust_col,
            "Snippet_Original": ";".join(snippets) if snippets else "",
            "Snippets_Normalized": ";".join(norm_uniq) if norm_uniq else "",
            "Heuristic_Tag": heur,
        })
        pbar.update(1)
    pbar.close()

    # Write per-file output
    out_path = out_dir / (path.stem + "_processed.xlsx")
    pd.DataFrame(rows).to_excel(out_path, index=False)

    # Update snippet summary (append, dedupe by (snippet_normalized, file_name, run_timestamp))
    run_ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    summ_path = summary_dir / "snippet_summary.xlsx"
    new_summ = pd.DataFrame([{"run_timestamp": run_ts, "file_name": path.name,
                              "snippet_normalized": k, "occurrence_count": v}
                              for k,v in total_snip_counts.items()])
    if summ_path.exists():
        old = pd.read_excel(summ_path)
        all_summ = pd.concat([old, new_summ], ignore_index=True)
    else:
        all_summ = new_summ
    # sort by occurrence desc
    all_summ = all_summ.sort_values(by="occurrence_count", ascending=False)
    all_summ.to_excel(summ_path, index=False)

    # Move input file to completed
    shutil.move(str(path), str(completed_dir / path.name))

    dt_secs = time.time() - t0
    logger.info(f"Processed {path.name} rows={total_rows} in {dt_secs/60:.2f} min")
    return path.name, total_rows, dt_secs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="single file (.csv or .xlsx)")
    ap.add_argument("--input_folder", help="folder with files")
    ap.add_argument("--output_folder", default="output")
    ap.add_argument("--completed_folder", default="completed")
    ap.add_argument("--summary_folder", default="summary")
    ap.add_argument("--custom_keywords_csv")
    ap.add_argument("--chars_after_keyword", type=int, default=25)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--ignore_terms", default="error,err")
    args = ap.parse_args()

    base = Path(".")
    inp_files = []
    if args.input:
        p = Path(args.input)
        if not p.exists(): raise FileNotFoundError(p)
        inp_files = [p]
    elif args.input_folder:
        p = Path(args.input_folder)
        if not p.exists(): raise FileNotFoundError(p)
        inp_files = sorted([x for x in p.iterdir() if x.suffix.lower() in (".csv",".xlsx")])
    else:
        raise SystemExit("Provide --input or --input_folder")

    out_dir = Path(args.output_folder); out_dir.mkdir(parents=True, exist_ok=True)
    comp_dir = Path(args.completed_folder); comp_dir.mkdir(parents=True, exist_ok=True)
    sum_dir = Path(args.summary_folder); sum_dir.mkdir(parents=True, exist_ok=True)

    log_q, listener = start_logger("logs")
    # Initial ETA (rough): assume 25k rows/min per job
    total_rows = 0
    for f in inp_files:
        if f.suffix.lower() == ".xlsx":
            total_rows += len(pd.read_excel(f, nrows=1_000_000))
        else:
            total_rows += len(pd.read_csv(f))
    jobs = os.cpu_count() if args.jobs in (-1, 0) else max(1, args.jobs)
    rows_per_min_est = 25000 * jobs
    eta_min = total_rows / rows_per_min_est if rows_per_min_est > 0 else 0
    print(f"Estimated total time: {math.ceil(eta_min)} minutes ({eta_min:.2f} min) for {total_rows} rows using {jobs} workers.")

    custom_kws = load_custom_keywords(args.custom_keywords_csv)
    ignore_terms = set([t.strip().lower() for t in args.ignore_terms.split(",") if t.strip()])

    results = []
    if jobs == 1:
        for f in inp_files:
            results.append(process_file(f, out_dir, comp_dir, sum_dir, custom_kws, args.chars_after_keyword, ignore_terms, log_q, args.progress))
    else:
        # Parallel per-file
        results = Parallel(n_jobs=jobs, prefer="processes")(
            delayed(process_file)(f, out_dir, comp_dir, sum_dir, custom_kws, args.chars_after_keyword, ignore_terms, log_q, args.progress)
            for f in inp_files
        )

    # Timing summary append
    ts_csv = Path(args.summary_folder) / "timing_summary.csv"
    run_ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    recs = []
    for name, rows, secs in results:
        recs.append({
            "run_timestamp": run_ts,
            "file_name": name,
            "row_count": rows,
            "estimated_minutes_start": math.ceil(eta_min),
            "actual_minutes": round(secs/60, 2),
            "rows_per_min": round(rows / (secs/60) if secs>0 else 0, 2),
            "start_time": run_ts,
            "end_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
    df = pd.DataFrame(recs)
    if ts_csv.exists():
        old = pd.read_csv(ts_csv)
        all_df = pd.concat([old, df], ignore_index=True)
    else:
        all_df = df
    all_df.to_csv(ts_csv, index=False)

    stop_logger(log_q, listener)

if __name__ == "__main__":
    main()
