"""
Dataset_Loader_Generation.py
Stage 4 — Joke Text Generation

Dataset format expected:
    A single text file (e.g. shortjokes.csv or jokes.txt) containing
    1622 jokes, one per line. If it's a CSV the first column is an ID
    and the second column is the joke text — this loader handles both.

The loader builds a character/word-level language modelling dataset
using a sliding window: given SEQ_LEN tokens, predict the next token.
"""

import os
import re
import csv
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────────────────────
# Text cleaning (lighter than classification — keep punctuation
# so the model can learn to generate readable sentences)
# ──────────────────────────────────────────────────────────────
def clean_joke(text: str) -> str:
    """
    Light cleaning for a joke string:
      - Lowercase
      - Remove non-ASCII characters
      - Collapse multiple spaces into one
      - Strip leading/trailing whitespace

    We keep punctuation (. , ? ! ' ) because the model needs to
    learn sentence endings and question marks for joke structure.
    """
    text = text.lower()
    text = text.encode("ascii", errors="ignore").decode()   # drop non-ASCII
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list:
    """
    Simple word-level tokenizer.
    Splits on whitespace but keeps punctuation attached to words
    (e.g. "why?" stays as one token). This keeps the vocabulary
    small while preserving sentence-ending cues for generation.
    """
    return text.split()


# ──────────────────────────────────────────────────────────────
# File reading — handles both plain .txt and CSV formats
# ──────────────────────────────────────────────────────────────
def read_jokes_file(filepath: str) -> list:
    """
    Read the jokes file and return a list of raw joke strings.

    Supports:
      - Plain .txt  : one joke per line
      - .csv        : expects columns [ID, Joke] — skips the header
                      and reads the second column
    """
    jokes = []
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".csv":
        with open(filepath, newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            next(reader, None)                    # skip header row
            for row in reader:
                if len(row) >= 2 and row[1].strip():
                    jokes.append(row[1].strip())
    else:
        # plain text — one joke per line
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:                          # skip blank lines
                    jokes.append(line)

    return jokes


# ──────────────────────────────────────────────────────────────
# Vocabulary builder
# ──────────────────────────────────────────────────────────────
def build_vocab(all_tokens: list, max_vocab_size: int = 8000) -> dict:
    """
    Build word → integer index mapping.

    Reserved indices:
      0 = <PAD>  (padding token, not used in generation but kept for
                  consistency with the classification loader)
      1 = <UNK>  (unknown / rare words)
    """
    counter = Counter(all_tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_vocab_size - 2):
        vocab[word] = len(vocab)
    return vocab


# ──────────────────────────────────────────────────────────────
# PyTorch Dataset — sliding window over the full token sequence
# ──────────────────────────────────────────────────────────────
class JokesDataset(Dataset):
    """
    Language modelling dataset using a sliding window.

    The entire corpus is flattened into one long token sequence.
    Each sample is:
        input  = tokens[i   : i + seq_len]
        target = tokens[i+1 : i + seq_len + 1]

    So the model learns: given these seq_len tokens, predict the
    next token at each position (standard next-word prediction).

    Args:
        token_ids (list of int): full encoded corpus
        seq_len   (int)        : context window size
    """
    def __init__(self, token_ids: list, seq_len: int):
        self.ids     = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        # every position except the last seq_len is a valid starting point
        return len(self.ids) - self.seq_len

    def __getitem__(self, idx):
        x = self.ids[idx           : idx + self.seq_len]
        y = self.ids[idx + 1       : idx + self.seq_len + 1]
        return x, y


# ──────────────────────────────────────────────────────────────
# Main loader function — call this from your script
# ──────────────────────────────────────────────────────────────
def load_generation_data(
    data_path:     str,
    seq_len:       int  = 20,
    max_vocab_size: int = 8000,
    batch_size:    int  = 64,
    num_workers:   int  = 0,
):
    """
    Load the jokes dataset, clean it, build vocabulary, encode the
    full corpus, and return a DataLoader for language model training.

    Args:
        data_path      : path to the jokes .txt or .csv file
        seq_len        : sliding window context length
        max_vocab_size : max tokens in vocabulary
        batch_size     : samples per training batch
        num_workers    : DataLoader workers (0 = main thread)

    Returns:
        train_loader : DataLoader (input, target) pairs
        vocab        : dict word → index
        id2word      : dict index → word  (needed for generation)
        corpus_text  : full cleaned corpus string (for overlap eval)
    """

    # ── 1. Read raw jokes ──────────────────────────────────────
    print(f"Reading jokes from: {data_path}")
    raw_jokes = read_jokes_file(data_path)
    print(f"  Jokes loaded: {len(raw_jokes):,}")

    # ── 2. Clean and tokenize each joke ───────────────────────
    all_tokens = []
    cleaned_jokes = []
    for joke in raw_jokes:
        cleaned  = clean_joke(joke)
        tokens   = tokenize(cleaned)
        cleaned_jokes.append(cleaned)
        all_tokens.extend(tokens)

    corpus_text = " ".join(cleaned_jokes)
    print(f"  Total tokens in corpus: {len(all_tokens):,}")

    # ── 3. Build vocabulary ────────────────────────────────────
    vocab   = build_vocab(all_tokens, max_vocab_size)
    id2word = {idx: word for word, idx in vocab.items()}
    print(f"  Vocabulary size: {len(vocab):,} tokens")

    # ── 4. Encode full corpus as one long integer sequence ─────
    token_ids = [vocab.get(t, 1) for t in all_tokens]  # 1 = <UNK>

    # ── 5. Create sliding window dataset ──────────────────────
    dataset = JokesDataset(token_ids, seq_len)
    print(f"  Training samples (sliding windows): {len(dataset):,}")

    train_loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,       # shuffle window positions each epoch
        num_workers = num_workers,
        drop_last   = True,       # keeps all batches the same size
    )

    print("Dataset ready.")
    return train_loader, vocab, id2word, corpus_text


# ──────────────────────────────────────────────────────────────
# Quick test — run this file directly to verify it works
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_PATH = "../../data/stage_4_data/text_generation/data"

    train_loader, vocab, id2word, corpus_text = load_generation_data(
        data_path      = DATA_PATH,
        seq_len        = 20,
        max_vocab_size = 8000,
        batch_size     = 64,
    )

    # Peek at one batch to confirm shapes
    xb, yb = next(iter(train_loader))
    print(f"\nSample batch — x shape: {xb.shape}  y shape: {yb.shape}")
    print(f"Vocab size: {len(vocab)}")

    # Decode the first sample back to words to sanity-check
    sample_words = [id2word.get(i.item(), "<UNK>") for i in xb[0]]
    target_words = [id2word.get(i.item(), "<UNK>") for i in yb[0]]
    print(f"\nFirst input sequence  : {' '.join(sample_words)}")
    print(f"First target sequence : {' '.join(target_words)}")
    print("(Target should be input shifted one word to the right)")