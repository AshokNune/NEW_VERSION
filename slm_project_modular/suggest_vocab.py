import os
import argparse
import pandas as pd
from collections import Counter
import re

def read_input(path):
    """Read Text + Keyword columns from file/folder of xlsx"""
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
        raise ValueError("Input must be a .xlsx file or a folder containing .xlsx files")
    return texts

def tokenize(text):
    """Basic whitespace + punctuation split"""
    return re.findall(r"\b\w+\b", text.lower())

def suggest_vocab_size(input_path, coverage=0.99):
    texts = read_input(input_path)

    # count word frequencies
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))

    total_tokens = sum(counter.values())
    unique_tokens = len(counter)

    print(f"🔹 Total tokens: {total_tokens:,}")
    print(f"🔹 Unique tokens: {unique_tokens:,}")

    # sort by frequency
    sorted_counts = counter.most_common()

    # coverage calculation
    running_total = 0
    cutoff = 0
    for i, (_, freq) in enumerate(sorted_counts, start=1):
        running_total += freq
        if running_total / total_tokens >= coverage:
            cutoff = i
            break

    print(f"✅ To cover {coverage*100:.1f}% of dataset → ~{cutoff} vocab size")
    print(f"💡 Rule of thumb:")
    print(f"   - Small dataset: {min(2000, cutoff)}")
    print(f"   - Medium dataset: {min(5000, cutoff)}")
    print(f"   - Large dataset: {min(20000, cutoff)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Suggest vocab size from dataset")
    parser.add_argument("--input", type=str, required=True, default = r"D:\SLM\slm_project_modular\data",help="Path to .xlsx file or folder of .xlsx files")
    parser.add_argument("--coverage", type=float, default=0.99, help="Fraction of tokens to cover (default 0.99)")
    args = parser.parse_args()

    suggest_vocab_size(args.input, args.coverage)
