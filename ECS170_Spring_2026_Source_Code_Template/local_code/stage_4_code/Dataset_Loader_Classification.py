"""
Dataset_Loader_Classification.py
Stage 4 — IMDB Sentiment Classification

Dataset folder structure expected:
    aclImdb/
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
import string
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader

# ── Stop words to remove during cleaning ──────────────────────
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "is", "was", "are", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "this", "that", "these",
    "those", "i", "you", "he", "she", "it", "we", "they", "my",
    "your", "his", "her", "its", "our", "their", "what", "which",
    "who", "not", "no", "so", "if", "as", "up", "out", "by",
}


# ──────────────────────────────────────────────────────────────
# Text cleaning
# ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> list:
    """
    Clean and tokenize one review string.

    Steps:
      1. Lowercase everything
      2. Remove HTML tags (IMDB reviews often contain <br /> etc.)
      3. Remove punctuation and digits
      4. Split into tokens
      5. Remove stop words and single-character tokens

    Returns a list of clean word tokens.
    """
    # lowercase
    text = text.lower()
    # remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # remove punctuation and digits — keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)
    # split
    tokens = text.split()
    # remove stop words and very short tokens
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return tokens


# ──────────────────────────────────────────────────────────────
# Vocabulary builder
# ──────────────────────────────────────────────────────────────
def build_vocab(all_token_lists: list, max_vocab_size: int = 20000) -> dict:
    """
    Build a word → integer index mapping from a list of token lists.

    Reserved indices:
      0 = <PAD>  (used to pad shorter sequences)
      1 = <UNK>  (used for words not in vocabulary)

    Only the top (max_vocab_size - 2) most frequent words are kept.
    """
    counter = Counter(token for tokens in all_token_lists for token in tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_vocab_size - 2):
        vocab[word] = len(vocab)
    return vocab


# ──────────────────────────────────────────────────────────────
# Encoding helper
# ──────────────────────────────────────────────────────────────
def encode_tokens(tokens: list, vocab: dict, max_len: int) -> list:
    """
    Convert a token list to a fixed-length integer sequence.
    - Truncates if longer than max_len
    - Pads with 0 (<PAD>) if shorter than max_len
    - Unknown words map to index 1 (<UNK>)
    """
    ids = [vocab.get(t, 1) for t in tokens[:max_len]]
    ids += [0] * (max_len - len(ids))
    return ids


# ──────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────
class IMDBDataset(Dataset):
    """
    PyTorch Dataset wrapping encoded IMDB reviews.

    Args:
        encoded (list of list): integer-encoded token sequences
        labels  (list of int):  0 = negative, 1 = positive
    """
    def __init__(self, encoded: list, labels: list):
        self.x = torch.tensor(encoded, dtype=torch.long)
        self.y = torch.tensor(labels,  dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ──────────────────────────────────────────────────────────────
# File reading helper
# ──────────────────────────────────────────────────────────────
def _read_folder(folder_path: str, label: int):
    """
    Read all .txt files inside folder_path and return
    (list of raw text strings, list of integer labels).
    """
    texts, labels = [], []
    for fname in os.listdir(folder_path):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(folder_path, fname)
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            texts.append(f.read().strip())
            labels.append(label)
    return texts, labels


# ──────────────────────────────────────────────────────────────
# Main loader function — call this from your script
# ──────────────────────────────────────────────────────────────
def load_classification_data(
    data_root: str,
    max_vocab_size: int = 20000,
    max_seq_len:   int = 200,
    batch_size:    int = 64,
    num_workers:   int = 0,
):
    """
    Load the IMDB dataset from disk, clean it, build vocabulary,
    encode sequences, and return DataLoaders.

    Args:
        data_root      : path to the aclImdb/ folder
        max_vocab_size : maximum number of unique tokens to keep
        max_seq_len    : fixed sequence length (truncate / pad)
        batch_size     : samples per training batch
        num_workers    : DataLoader worker processes (0 = main thread)

    Returns:
        train_loader : DataLoader for training set
        test_loader  : DataLoader for test set
        vocab        : dict mapping word → index
        label2id     : {'neg': 0, 'pos': 1}
    """

    label2id = {"neg": 0, "pos": 1}

    # ── 1. Read raw text files ─────────────────────────────────
    print("Reading train/pos ...")
    tr_pos_texts, tr_pos_labels = _read_folder(
        os.path.join(data_root, "train", "pos"), label=1)

    print("Reading train/neg ...")
    tr_neg_texts, tr_neg_labels = _read_folder(
        os.path.join(data_root, "train", "neg"), label=0)

    print("Reading test/pos ...")
    te_pos_texts, te_pos_labels = _read_folder(
        os.path.join(data_root, "test", "pos"), label=1)

    print("Reading test/neg ...")
    te_neg_texts, te_neg_labels = _read_folder(
        os.path.join(data_root, "test", "neg"), label=0)

    train_texts  = tr_pos_texts + tr_neg_texts
    train_labels = tr_pos_labels + tr_neg_labels
    test_texts   = te_pos_texts + te_neg_texts
    test_labels  = te_pos_labels + te_neg_labels

    print(f"  Train samples : {len(train_texts):,}  "
          f"(pos={len(tr_pos_texts):,}  neg={len(tr_neg_texts):,})")
    print(f"  Test  samples : {len(test_texts):,}  "
          f"(pos={len(te_pos_texts):,}  neg={len(te_neg_texts):,})")

    # ── 2. Clean and tokenize ──────────────────────────────────
    print("Cleaning and tokenizing ...")
    train_tokens = [clean_text(t) for t in train_texts]
    test_tokens  = [clean_text(t) for t in test_texts]

    # ── 3. Build vocabulary from TRAINING set only ─────────────
    # (never use test set to build vocab — data leakage)
    print("Building vocabulary ...")
    vocab = build_vocab(train_tokens, max_vocab_size)
    print(f"  Vocabulary size : {len(vocab):,} tokens")

    # ── 4. Encode sequences ────────────────────────────────────
    train_encoded = [encode_tokens(t, vocab, max_seq_len) for t in train_tokens]
    test_encoded  = [encode_tokens(t, vocab, max_seq_len) for t in test_tokens]

    # ── 5. Wrap in Dataset and DataLoader ──────────────────────
    train_dataset = IMDBDataset(train_encoded, train_labels)
    test_dataset  = IMDBDataset(test_encoded,  test_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,       # shuffle training order each epoch
        num_workers = num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,      # keep test order consistent
        num_workers = num_workers,
    )

    print("Dataset ready.")
    return train_loader, test_loader, vocab, label2id


# ──────────────────────────────────────────────────────────────
# Quick test — run this file directly to verify it works
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":

    DATA_ROOT = "../../data/stage_4_data/test_classification"

    train_loader, test_loader, vocab, label2id = load_classification_data(
        data_root      = DATA_ROOT,
        max_vocab_size = 20000,
        max_seq_len    = 200,
        batch_size     = 64,
    )

    # Peek at one batch to confirm shapes
    xb, yb = next(iter(train_loader))
    print(f"\nSample batch — x shape: {xb.shape}  y shape: {yb.shape}")
    print(f"Label mapping: {label2id}")
    print(f"First 10 tokens of review 0: {xb[0, :10].tolist()}")
    print(f"Label of review 0: {'pos' if yb[0].item() == 1 else 'neg'}")