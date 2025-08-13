#!/usr/bin/env python3
import re, hashlib, gzip, pickle, os, time
from typing import Dict, Tuple, List

# Regex patterns
RE_EMAIL = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
RE_IP = re.compile(r'(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.|$)){4}')
RE_URL = re.compile(r'(?:(?:https?://|www\.)[^\s]+|[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/[^\s]*)?)')
RE_TIME = re.compile(r'(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?:\s?(?:AM|PM|am|pm))?')
RE_DATE = re.compile(
    r'(?:\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})',
    re.IGNORECASE
)
RE_DATETIME = re.compile(r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}')
RE_NUMBER = re.compile(r'(?<![A-Za-z])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![A-Za-z])')

PLACEHOLDERS = [
    ("<datetime>", RE_DATETIME),
    ("<date>", RE_DATE),
    ("<time>", RE_TIME),
    ("<email>", RE_EMAIL),
    ("<ip>", RE_IP),
    ("<url>", RE_URL),
    ("<num>", RE_NUMBER),
]

def normalize_snippet(s: str) -> str:
    if not isinstance(s, str) or not s:
        return ""
    text = s

    # Replace entities with placeholders
    for token, pattern in PLACEHOLDERS:
        text = pattern.sub(token, text)

    # Compress repeated placeholders (e.g., "<num> <num>" -> "<num>")
    for token, _ in PLACEHOLDERS:
        # multiple separated by spaces/punct
        text = re.sub(r'(?:' + re.escape(token) + r'(?:[\s,;:/-]+)?){2,}', token, text)

    # Collapse whitespace and strip
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def hash_norm(s: str) -> str:
    n = normalize_snippet(s)
    return hashlib.sha256(n.encode('utf-8')).hexdigest()

def build_cache_from_feedback(feedback_csv: str, cache_path: str) -> Tuple[Dict[str, str], int]:
    import pandas as pd, os
    if not os.path.exists(feedback_csv):
        raise FileNotFoundError(f"feedback csv not found: {feedback_csv}")
    df = pd.read_csv(feedback_csv)
    # Expect columns: Snippet_Normalized, Classification
    snip_col = None
    for c in df.columns:
        if c.lower().startswith("snippet") and "normal" in c.lower():
            snip_col = c; break
    if snip_col is None:
        raise ValueError("feedback.csv must contain a Snippet_Normalized column")
    class_col = None
    for c in df.columns:
        if c.lower().startswith("class"):
            class_col = c; break
    if class_col is None:
        class_col = "Classification"
        if class_col not in df.columns:
            df[class_col] = ""

    mapping = {}
    for _, row in df.iterrows():
        sn = str(row[snip_col]) if pd.notna(row[snip_col]) else ""
        cl = str(row[class_col]) if pd.notna(row[class_col]) else ""
        h = hash_norm(sn)
        if h and cl:
            mapping[h] = cl

    with gzip.open(cache_path, "wb") as f:
        pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)
    return mapping, len(mapping)

def load_cache(cache_path: str) -> Dict[str, str]:
    import pickle, gzip, os
    if not os.path.exists(cache_path):
        return {}
    with gzip.open(cache_path, "rb") as f:
        return pickle.load(f)
