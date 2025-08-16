import argparse, os, json, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt
from model import TransformerSLM

def ddp_setup(rank, world_size, backend):
    torch.distributed.init_process_group(
        backend=backend,
        init_method="tcp://127.0.0.1:29500",
        world_size=world_size,
        rank=rank,
    )

def load_vocab_size(tokenizer_dir):
    with open(os.path.join(tokenizer_dir, "vocab.json"), "r", encoding="utf-8") as vf:
        vocab = json.load(vf)
    return len(vocab)

def _train(rank, world_size, args):
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    ddp_setup(rank, world_size, backend)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    data = torch.load(args.data_path)
    X, y = data["input_ids"], data["labels"]
    pad_token_id = data["pad_token_id"]
    tokenizer_dir = data["tokenizer_dir"]
    label_map = data["label_map"]
    num_classes = len(label_map)
    vocab_size = load_vocab_size(tokenizer_dir)

    ds = TensorDataset(X, y)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
    dl = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)

    model = TransformerSLM(vocab_size=vocab_size, num_classes=num_classes, max_len=X.size(1))
    model = model.to(device)
    if torch.cuda.is_available():
        model = DDP(model, device_ids=[rank])
    else:
        model = DDP(model)

    loss_cls = nn.CrossEntropyLoss()
    loss_lm = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimz = optim.AdamW(model.parameters(), lr=args.lr)

    epoch_losses = []
    for epoch in range(args.epochs):
        model.train()
        sampler.set_epoch(epoch)
        running = 0.0
        for batch in dl:
            xb, yb = [t.to(device, non_blocking=True) for t in batch]
            lm_logits, cls_logits = model(xb)

            # LM next-token prediction
            lm_targets = xb[:, 1:].contiguous()
            lm_pred = lm_logits[:, :-1, :].contiguous()

            l_lm = loss_lm(lm_pred.view(-1, lm_pred.size(-1)), lm_targets.view(-1))
            l_cls = loss_cls(cls_logits, yb)
            loss = args.lm_weight * l_lm + l_cls

            optimz.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimz.step()

            running += loss.detach().item()

        avg = running / max(1, len(dl))
        if rank == 0:
            epoch_losses.append(avg)
            print(f"Epoch {epoch+1}/{args.epochs} - loss: {avg:.4f}")

    if rank == 0:
        os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
        torch.save(model.module.state_dict(), args.model_out)
        plt.figure()
        plt.plot(epoch_losses)
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("SLM Training Loss")
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(args.model_out), "training_loss.png"))
        print(f"✅ Saved model to {args.model_out}")

    torch.distributed.destroy_process_group()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="artifacts/data/batches.pt")
    ap.add_argument("--model_out", default="artifacts/model/final_model.pt")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lm_weight", type=float, default=0.5)
    ap.add_argument("--world_size", type=int, default=1, help="# processes (GPUs or CPU workers)")
    args = ap.parse_args()

    world_size = args.world_size
    torch.multiprocessing.spawn(_train, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
