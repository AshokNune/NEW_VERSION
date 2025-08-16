import argparse, os, json, torch, pandas as pd
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
    ap.add_argument("--input_file", required=True, help="CSV or XLSX with columns Text, Keyword, classification")
    ap.add_argument("--tokenizer_dir", default="artifacts/tokenizer")
    ap.add_argument("--text_col", default="Text")
    ap.add_argument("--keyword_col", default="Keyword")
    ap.add_argument("--label_col", default="classification")
    ap.add_argument("--out_dir", default="artifacts/data")
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    df = read_table(args.input_file)
    os.makedirs(args.out_dir, exist_ok=True)

    tok = ByteLevelBPETokenizer(
        os.path.join(args.tokenizer_dir, "vocab.json"),
        os.path.join(args.tokenizer_dir, "merges.txt")
    )

    with open(os.path.join(args.tokenizer_dir, "vocab.json"), "r", encoding="utf-8") as vf:
        vocab = json.load(vf)
    pad_id = vocab.get("[PAD]", 0)

    labels_cat = df[args.label_col].astype("category")
    labels = labels_cat.cat.codes.tolist()
    label_map = dict(enumerate(labels_cat.cat.categories))

    input_ids = []
    for t, k in zip(df[args.text_col].astype(str), df[args.keyword_col].astype(str)):
        ids = tok.encode(f"{t} [SEP] {k}").ids[:args.max_length]
        input_ids.append(ids)

    out_path = os.path.join(args.out_dir, "tokenized.pt")
    torch.save({
        "input_ids": input_ids,
        "labels": labels,
        "label_map": label_map,
        "pad_token_id": pad_id,
        "max_length": args.max_length,
        "tokenizer_dir": args.tokenizer_dir
    }, out_path)

    with open(os.path.join(args.out_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    print(f"✅ Tokenized dataset saved to {out_path}")

if __name__ == "__main__":
    main()
