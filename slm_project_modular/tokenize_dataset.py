import argparse
import os
import pandas as pd
import torch
from tokenizers import ByteLevelBPETokenizer

def load_files(input_dir):
    dfs = []
    for fname in os.listdir(input_dir):
        fpath = os.path.join(input_dir, fname)
        if fname.endswith(".csv"):
            dfs.append(pd.read_csv(fpath))
        elif fname.endswith(".xlsx"):
            dfs.append(pd.read_excel(fpath, engine="openpyxl"))
    if not dfs:
        raise ValueError(f"No CSV/XLSX files found in {input_dir}")
    return pd.concat(dfs, ignore_index=True)

def main(args):
    # Load tokenizer
    tokenizer = ByteLevelBPETokenizer(
        os.path.join(args.tokenizer_dir, "vocab.json"),
        os.path.join(args.tokenizer_dir, "merges.txt"),
    )

    # Load all files from input_dir
    df = load_files(args.input_dir)
    if "Text" not in df.columns or "classification" not in df.columns:
        raise ValueError("Dataset must contain 'Text' and 'classification' columns")

    # Encode dataset
    input_ids = [tokenizer.encode(text).ids for text in df["Text"].astype(str)]
    labels = df["classification"].astype("category").cat.codes.values

    # Pad sequences (simple padding to max_len)
    max_len = max(len(seq) for seq in input_ids)
    padded_inputs = [seq + [0]*(max_len - len(seq)) for seq in input_ids]

    dataset = {
        "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(dataset, os.path.join(args.out_dir, "tokenized_dataset.pt"))
    print(f"✅ Saved tokenized dataset to {args.out_dir}/tokenized_dataset.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data", help="Folder with CSV/XLSX files")
    parser.add_argument("--tokenizer_dir", type=str, default="artifacts/tokenizer")
    parser.add_argument("--out_dir", type=str, default="artifacts")
    args = parser.parse_args()
    main(args)
