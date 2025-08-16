import argparse
import os
import pandas as pd
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
    # Load dataset
    df = load_files(args.input_dir)
    if "Text" not in df.columns:
        raise ValueError("Dataset must contain a 'Text' column")

    texts = df["Text"].astype(str).tolist()

    # Train tokenizer (BPE)
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(texts, vocab_size=args.vocab_size, min_frequency=2)

    # Save tokenizer
    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer.save_model(args.out_dir)

    vocab_file = os.path.join(args.out_dir, "vocab.json")
    merges_file = os.path.join(args.out_dir, "merges.txt")

    # ✅ Verify both files exist
    if not os.path.exists(vocab_file) or not os.path.exists(merges_file):
        raise RuntimeError("Tokenizer training failed: vocab.json or merges.txt not found!")

    print(f"✅ Tokenizer trained and saved to {args.out_dir}")
    print(f"   Files created: vocab.json, merges.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data", help="Folder with CSV/XLSX files")
    parser.add_argument("--out_dir", type=str, default="artifacts/tokenizer")
    parser.add_argument("--vocab_size", type=int, default=3000)
    args = parser.parse_args()
    main(args)
