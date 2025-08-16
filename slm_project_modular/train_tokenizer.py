import os
import argparse
import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

def read_input(path):
    """Read either a single file or all .xlsx files in a folder"""
    texts = []
    if os.path.isfile(path):
        if path.endswith(".xlsx"):
            df = pd.read_excel(path)
            if "Text" in df.columns:
                texts.extend(df["Text"].astype(str).tolist())
            if "Keyword" in df.columns:
                texts.extend(df["Keyword"].astype(str).tolist())
    elif os.path.isdir(path):
        for file in os.listdir(path):
            if file.endswith(".xlsx"):
                df = pd.read_excel(os.path.join(path, file))
                if "Text" in df.columns:
                    texts.extend(df["Text"].astype(str).tolist())
                if "Keyword" in df.columns:
                    texts.extend(df["Keyword"].astype(str).tolist())
    else:
        raise ValueError("Input path must be a .xlsx file or a folder containing .xlsx files")
    return texts

def train_tokenizer(input_path, vocab_size=1000, save_path="artifacts/tokenizer/tokenizer.json"):
    texts = read_input(input_path)

    # Define a BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.BPEDecoder()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    print(f"Tokenizer trained with vocab_size={vocab_size} and saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on .xlsx data")
    parser.add_argument("--input", type=str, required=True, help="Path to .xlsx file or folder of .xlsx files")
    parser.add_argument("--vocab_size", type=int, default=1000, help="Vocabulary size")
    parser.add_argument("--save_path", type=str, default="artifacts/tokenizer/tokenizer.json", help="Path to save tokenizer JSON")

    args = parser.parse_args()

    train_tokenizer(args.input, args.vocab_size, args.save_path)
