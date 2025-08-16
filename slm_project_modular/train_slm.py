import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from model import TransformerSLM

def load_vocab_size(tokenizer_dir: str) -> int:
    with open(os.path.join(tokenizer_dir, "vocab.json"), "r", encoding="utf-8") as vf:
        vocab = json.load(vf)
    return len(vocab)

def train_single_process(
    data_path: str,
    out_dir: str,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 3e-4,
    lm_weight: float = 0.5,
    num_workers: int = 4,
    val_split: float = 0.1,
    seed: int = 42,
):
    torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "best_model.pt")

    # Load batches
    blob = torch.load(data_path)
    X: torch.Tensor = blob["input_ids"]
    y: torch.Tensor = blob["labels"]
    pad_token_id: int = int(blob.get("pad_token_id", 0))
    tokenizer_dir: str = blob["tokenizer_dir"]
    label_map = blob["label_map"]

    # num classes from label_map values (handles LFP->0, others->1..)
    num_classes = max(label_map.values()) + 1
    vocab_size = load_vocab_size(tokenizer_dir)
    max_len = X.size(1)

    # Dataset split
    ds = TensorDataset(X, y)
    n = len(ds)
    n_val = max(1, int(n * val_split))
    n_train = max(1, n - n_val)
    if n_train + n_val > n:
        n_val = n - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerSLM(vocab_size=vocab_size, num_classes=num_classes, max_len=max_len).to(device)

    # Losses & optimizer
    loss_cls = nn.CrossEntropyLoss()
    loss_lm = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    train_losses, val_losses, val_accs = [], [], []
    best_acc = -1.0

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            lm_logits, cls_logits = model(xb)

            # LM next-token prediction (shifted)
            lm_targets = xb[:, 1:].contiguous()
            lm_pred = lm_logits[:, :-1, :].contiguous()

            l_lm = loss_lm(lm_pred.view(-1, vocab_size), lm_targets.view(-1))
            l_cls = loss_cls(cls_logits, yb)
            loss = lm_weight * l_lm + l_cls

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total += float(loss.detach().item())

        avg_train = total / max(1, len(train_loader))
        train_losses.append(avg_train)

        # Validation (use classification metrics)
        model.eval()
        vtotal = 0.0
        vcount = 0
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                _, cls_logits = model(xb)
                l = loss_cls(cls_logits, yb)
                vtotal += float(l.item()) * xb.size(0)
                vcount += xb.size(0)
                pred = cls_logits.argmax(dim=-1)
                correct += (pred == yb).sum().item()
        avg_val = vtotal / max(1, vcount)
        acc = correct / max(1, vcount)
        val_losses.append(avg_val)
        val_accs.append(acc)

        print(f"Epoch {epoch+1}/{epochs} - train_loss: {avg_train:.4f} | val_loss: {avg_val:.4f} | val_acc: {acc:.3f}")

        # ✅ Save BEST ONLY (by highest validation accuracy)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)
            print(f"  ↳ Saved new best model to {best_path} (val_acc={best_acc:.3f})")

    # Plot losses/acc
    plt.figure()
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.plot(val_accs, label="val_acc")
    plt.xlabel("Epoch"); plt.title("Training (best checkpoint saved)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curve.png"))

    print(f"✅ Best model saved at {best_path} (best val_acc={best_acc:.3f})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="artifacts/data/batches.pt")
    ap.add_argument("--out_dir", default="artifacts/model")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lm_weight", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_single_process(
        data_path=args.data_path,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lm_weight=args.lm_weight,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
