import argparse, torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="artifacts/data/tokenized.pt")
    ap.add_argument("--out_path", default="artifacts/data/batches.pt")
    args = ap.parse_args()

    data = torch.load(args.data_path)
    ids = data["input_ids"]
    pad_token_id = data["pad_token_id"]
    max_len = data["max_length"]

    padded = [seq + [pad_token_id] * (max_len - len(seq)) for seq in ids]
    dataset = {
        "input_ids": torch.tensor(padded, dtype=torch.long),
        "labels": torch.tensor(data["labels"], dtype=torch.long),
        "pad_token_id": pad_token_id,
        "tokenizer_dir": data["tokenizer_dir"],
        "label_map": data["label_map"]
    }
    torch.save(dataset, args.out_path)
    print(f"✅ Batches saved to {args.out_path} (N={len(padded)}, T={max_len})")

if __name__ == "__main__":
    main()
