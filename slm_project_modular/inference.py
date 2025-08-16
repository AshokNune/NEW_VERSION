import os
import json
import argparse
import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer
from model import TransformerSLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="artifacts/model/best_model.pt")
    ap.add_argument("--tokenizer_dir", default="artifacts/tokenizer")
    ap.add_argument("--label_map_path", default="artifacts/data/label_map.json")
    ap.add_argument("--text", required=True)
    ap.add_argument("--keyword", default="")
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    # Load tokenizer + vocab
    tok = ByteLevelBPETokenizer(
        os.path.join(args.tokenizer_dir, "vocab.json"),
        os.path.join(args.tokenizer_dir, "merges.txt"),
    )
    with open(os.path.join(args.tokenizer_dir, "vocab.json"), "r", encoding="utf-8") as vf:
        vocab = json.load(vf)
    pad_id = vocab.get("[PAD]", 0)
    vocab_size = len(vocab)

    # Load label map (e.g., {"LFP":0, "LTP":1, ...})
    with open(args.label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    id_to_label = {int(v): k for k, v in label_map.items()}
    num_classes = max(label_map.values()) + 1

    # Build input
    text = f"{args.text} [SEP] {args.keyword}"
    ids = tok.encode(text).ids[:args.max_length]
    ids = ids + [pad_id] * (args.max_length - len(ids))
    x = torch.tensor([ids], dtype=torch.long)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerSLM(vocab_size=vocab_size, num_classes=num_classes, max_len=args.max_length).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        _, cls_logits = model(x.to(device))
        probs = F.softmax(cls_logits, dim=-1).cpu().numpy()[0]
        pred_id = int(probs.argmax())
        pred_label = id_to_label.get(pred_id, str(pred_id))

    print("Predicted:", pred_label)
    print("Probs:", [float(f"{p:.4f}") for p in probs])

if __name__ == "__main__":
    main()
