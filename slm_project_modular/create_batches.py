import argparse
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="artifacts/data/tokenized.pt")
    ap.add_argument("--out_path", default="artifacts/data/batches.pt")
    args = ap.parse_args()

    blob = torch.load(args.data_path)
    ids_list = blob["input_ids"]
    labels = blob["labels"]
    pad_token_id = int(blob.get("pad_token_id", 0))
    max_length = int(blob.get("max_length", max(len(s) for s in ids_list)))
    label_map = blob["label_map"]

    # Pad to max_length saved at tokenization time
    padded = [seq + [pad_token_id] * (max_length - len(seq)) for seq in ids_list]

    out = {
        "input_ids": torch.tensor(padded, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "pad_token_id": pad_token_id,
        "tokenizer_dir": blob["tokenizer_dir"],
        "label_map": label_map,
    }
    torch.save(out, args.out_path)
    print(f"✅ Batches saved to {args.out_path} (N={len(padded)}, T={max_length})")

if __name__ == "__main__":
    main()
