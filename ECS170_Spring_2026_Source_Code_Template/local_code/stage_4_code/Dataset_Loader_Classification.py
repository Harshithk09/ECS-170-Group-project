"""
Dataset_Loader_Classification.py
Stage 4 — IMDB Sentiment Classification

Dataset folder structure expected:
    text_classification/
    ├── train/
    │   ├── pos/   (*.txt files — positive reviews)
    │   └── neg/   (*.txt files — negative reviews)
    └── test/
        ├── pos/
        └── neg/

Each .txt file is one movie review. Labels: pos=1, neg=0.
"""

import os
import re
import random
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "is", "was", "are", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "this", "that", "these",
    "those", "i", "you", "he", "she", "it", "we", "they", "my",
    "your", "his", "her", "its", "our", "their", "what", "which",
    "who", "not", "no", "so", "if", "as", "up", "out", "by",
}


def clean_text(text: str) -> list:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return tokens


def build_vocab(all_token_lists: list, max_vocab_size: int = 20000) -> dict:
    counter = Counter(token for tokens in all_token_lists for token in tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_vocab_size - 2):
        vocab[word] = len(vocab)
    return vocab


def encode_tokens(tokens: list, vocab: dict, max_len: int) -> list:
    ids = [vocab.get(t, 1) for t in tokens[:max_len]]
    ids += [0] * (max_len - len(ids))
    return ids


class IMDBDataset(Dataset):
    def __init__(self, encoded: list, labels: list):
        self.x = torch.tensor(encoded, dtype=torch.long)
        self.y = torch.tensor(labels,  dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def _read_folder(folder_path: str, label: int):
    texts, labels = [], []
    for fname in os.listdir(folder_path):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(folder_path, fname)
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            texts.append(f.read().strip())
            labels.append(label)
    return texts, labels


def load_classification_data(
    data_root:      str,
    max_vocab_size: int   = 20000,
    max_seq_len:    int   = 200,
    batch_size:     int   = 64,
    val_split:      float = 0.1,
    num_workers:    int   = 0,
    seed:           int   = 42,
):
    """
    Load IMDB dataset and return train, val, and test DataLoaders.

    Args:
        data_root      : path to the text_classification/ folder
        max_vocab_size : maximum vocabulary size
        max_seq_len    : fixed sequence length (truncate / pad)
        batch_size     : samples per batch
        val_split      : fraction of training set used for validation (default 10%)
        num_workers    : DataLoader workers
        seed           : random seed for reproducibility

    Returns:
        train_loader, val_loader, test_loader, vocab, label2id
    """
    label2id = {"neg": 0, "pos": 1}

    # ── 1. Read raw files ──────────────────────────────────────
    print("Reading train/pos ...")
    tr_pos_texts, tr_pos_labels = _read_folder(os.path.join(data_root, "train", "pos"), label=1)
    print("Reading train/neg ...")
    tr_neg_texts, tr_neg_labels = _read_folder(os.path.join(data_root, "train", "neg"), label=0)
    print("Reading test/pos ...")
    te_pos_texts, te_pos_labels = _read_folder(os.path.join(data_root, "test",  "pos"), label=1)
    print("Reading test/neg ...")
    te_neg_texts, te_neg_labels = _read_folder(os.path.join(data_root, "test",  "neg"), label=0)

    train_texts  = tr_pos_texts  + tr_neg_texts
    train_labels = tr_pos_labels + tr_neg_labels
    test_texts   = te_pos_texts  + te_neg_texts
    test_labels  = te_pos_labels + te_neg_labels

    print(f"  Train samples : {len(train_texts):,}  (pos={len(tr_pos_texts):,}  neg={len(tr_neg_texts):,})")
    print(f"  Test  samples : {len(test_texts):,}  (pos={len(te_pos_texts):,}  neg={len(te_neg_texts):,})")

    # ── 2. Clean and tokenize ──────────────────────────────────
    print("Cleaning and tokenizing ...")
    train_tokens = [clean_text(t) for t in train_texts]
    test_tokens  = [clean_text(t) for t in test_texts]

    # ── 3. Build vocab from training set only ─────────────────
    print("Building vocabulary ...")
    vocab = build_vocab(train_tokens, max_vocab_size)
    print(f"  Vocabulary size : {len(vocab):,} tokens")

    # ── 4. Encode ──────────────────────────────────────────────
    train_encoded = [encode_tokens(t, vocab, max_seq_len) for t in train_tokens]
    test_encoded  = [encode_tokens(t, vocab, max_seq_len) for t in test_tokens]

    # ── 5. Split train → train / val ──────────────────────────
    random.seed(seed)
    indices   = list(range(len(train_encoded)))
    random.shuffle(indices)
    n_val     = int(len(indices) * val_split)
    val_idx   = indices[:n_val]
    train_idx = indices[n_val:]

    val_encoded = [train_encoded[i] for i in val_idx]
    val_labels  = [train_labels[i]  for i in val_idx]
    tr_encoded  = [train_encoded[i] for i in train_idx]
    tr_labels   = [train_labels[i]  for i in train_idx]

    print(f"  Train after split : {len(tr_encoded):,}")
    print(f"  Val               : {len(val_encoded):,}")

    # ── 6. DataLoaders ─────────────────────────────────────────
    train_loader = DataLoader(IMDBDataset(tr_encoded,    tr_labels),   batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(IMDBDataset(val_encoded,   val_labels),  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(IMDBDataset(test_encoded,  test_labels), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("Dataset ready.")
    return train_loader, val_loader, test_loader, vocab, label2id


if __name__ == "__main__":
    DATA_ROOT = "../../data/stage_4_data/text_classification"
    train_loader, val_loader, test_loader, vocab, label2id = load_classification_data(
        data_root=DATA_ROOT, max_vocab_size=20000, max_seq_len=200, batch_size=64)
    xb, yb = next(iter(train_loader))
    print(f"\nSample batch — x: {xb.shape}  y: {yb.shape}")
    print(f"Label mapping: {label2id}")