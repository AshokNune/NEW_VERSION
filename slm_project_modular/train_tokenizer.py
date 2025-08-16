import argparse, os, json, pandas as pd
from tokenizers import ByteLevelBPETokenizer

def read_table(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".xlsx"):
        return pd.read_excel(path)
    else:
        raise ValueError("Unsupported file type. Use .csv or .xlsx")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True, help="CSV or XLSX with columns Text, Keyword")
    ap.add_argument("--text_col", default="Text")
    ap.add_argument("--keyword_col", default="Keyword")
    ap.add_argument("--out_dir", default="artifacts/tokenizer")
    ap.add_argument("--vocab_size", type=int, default=30000)
    ap.add_argument("--min_freq", type=int, default=2)
    args = ap.parse_args()

    df = read_table(args.input_file)
    texts = (df[args.text_col].astype(str) + " [SEP] " + df[args.keyword_col].astype(str)).tolist()

    os.makedirs(args.out_dir, exist_ok=True)
    tmp_corpus = os.path.join(args.out_dir, "corpus.txt")
    with open(tmp_corpus, "w", encoding="utf-8") as f:
        for t in texts:
            f.write((t or "").strip() + "\n")

    tok = ByteLevelBPETokenizer()
    tok.train(files=[tmp_corpus], vocab_size=args.vocab_size, min_frequency=args.min_freq,
              special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    tok.save_model(args.out_dir)

    # Save meta (vocab size + pad id)
    with open(os.path.join(args.out_dir, "vocab.json"), "r", encoding="utf-8") as vf:
        vocab = json.load(vf)
    pad_id = vocab.get("[PAD]", None)
    meta = {"vocab_size": len(vocab), "pad_token_id": pad_id}
    with open(os.path.join(args.out_dir, "tokenizer_meta.json"), "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)
    print(f"✅ Tokenizer saved to {args.out_dir} (vocab_size={len(vocab)}, pad_id={pad_id})")

if __name__ == "__main__":
    main()
