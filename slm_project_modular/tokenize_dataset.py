import argparse
import os
import json
import torch
import pandas as pd
from tokenizers import ByteLevelBPETokenizer

def read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    if path.lower().endswith(".xlsx"):
        return pd.read_excel(path)
    raise ValueError("Unsupported file type. Use .csv or .xlsx")

def make_label_map(series) -> dict:
    # Map LFP -> 0, remaining unique labels -> 1..N (sorted)
    uniq = sorted(set(str(x) for x in series.dropna().tolist()))
    label_map = {}
    if "LFP" in uniq:
        label_map["LFP"] = 0
        others = [u for u in uniq if u != "LFP"]
        for i, lab in enumerate(others, start=1):
            label_map[lab] = i
    else:
        # If no LFP present, just enumerate starting at 0
        for i, lab in enumerate(uniq):
            label_map[lab] = i
    return label_map

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True, help="CSV or XLSX with Text, Keyword, classification")
    ap.add_argument("--tokenizer_dir", default="artifacts/tokenizer")
    ap.add_argument("--text_col", default="Text")
    ap.add_argument("--keyword_col", default="Keyword")
    ap.add_argument("--label_col", default="classification")
    ap.add_argument("--out_dir", default="artifacts/data")
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    df = read_table(args.input_file)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load tokenizer (vocab + merges)
    tok = ByteLevelBPETokenizer(
        os.path.join(args.tokenizer_dir, "vocab.json"),
        os.path.join(args.tokenizer_dir, "merges.txt"),
    )
    with open(os.path.join(args.tokenizer_dir, "vocab.json"), "r", encoding="utf-8") as vf:
        vocab = json.load(vf)
    pad_id = vocab.get("[PAD]", 0)

    # Build label_map (LFP -> 0, others -> 1..)
    label_map = make_label_map(df[args.label_col])
    labels = []
    for v in df[args.label_col].astype(str):
        if v not in label_map:
            # unseen label gets a new id at the end
            label_map[v] = max(label_map.values()) + 1 if label_map else 0
        labels.append(label_map[v])

    # Tokenize
    input_ids = []
    for t, k in zip(df[args.text_col].astype(str), df[args.keyword_col].astype(str)):
        ids = tok.encode(f"{t} [SEP] {k}").ids[:args.max_length]
        input_ids.append(ids)

    # Save tokenized data (+ label map)
    out_path = os.path.join(args.out_dir, "tokenized.pt")
    torch.save(
        {
            "input_ids": input_ids,
            "labels": labels,
            "label_map": label_map,              # e.g., {"LFP":0, "LTP":1, ...}
            "pad_token_id": pad_id,
            "max_length": args.max_length,
            "tokenizer_dir": args.tokenizer_dir,
        },
        out_path,
    )
    with open(os.path.join(args.out_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    print(f"✅ Tokenized dataset saved to {out_path}")

if __name__ == "__main__":
    main()
