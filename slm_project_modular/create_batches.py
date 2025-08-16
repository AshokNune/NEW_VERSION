import torch
import argparse
import os

def create_batches(tokenized_file, out_file, seq_len=64, batch_size=32):
    # Load tokenized dataset (list of IDs)
    dataset = torch.load(tokenized_file)
    data = torch.tensor(dataset, dtype=torch.long)

    # Number of batches
    num_batches = len(data) // (batch_size * seq_len)
    data = data[:num_batches * batch_size * seq_len + 1]

    # Create input and target sequences
    x = data[:-1].view(batch_size, -1)
    y = data[1:].view(batch_size, -1)

    # Split into chunks of seq_len
    x_batches = torch.split(x, seq_len, dim=1)
    y_batches = torch.split(y, seq_len, dim=1)

    # Store batches
    batches = list(zip(x_batches, y_batches))
    torch.save(batches, out_file)

    print(f"✅ Created {len(batches)} batches saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenized_file", type=str, default="artifacts/tokenized_dataset.pt")
    parser.add_argument("--out_file", type=str, default="artifacts/batches.pt")
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    create_batches(args.tokenized_file, args.out_file, args.seq_len, args.batch_size)
