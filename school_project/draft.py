from torchtext.data.utils import get_tokenizer

# Tokenizers
en_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
fr_tokenizer = get_tokenizer("spacy", language="fr_core_news_sm")

SPECIAL_TOKENS = ["<unk>", "<pad>", "<bos>", "<eos>"]
MAX_VOCAB = 10_004
UNK_TOKEN = "<unk>"

# --- Paths ---
train_file_en = "./Data/train.en"
train_file_fr = "./Data/train.fr"

# --- Function ---
def get_tokens(path, tokenizer):
    all_tokens = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            all_tokens.append(tokenizer(line.strip()))
    return all_tokens

# --- Load tokens ---
enToken = get_tokens(train_file_en, en_tokenizer)
frToken = get_tokens(train_file_fr, fr_tokenizer)
from collections import Counter
from torchtext.vocab import build_vocab_from_iterator

# ==== Build vocab function ====

def yield_tokens(token_list):
    for tokens in token_list:
        yield tokens

# ==== English vocab ====

en_vocab = build_vocab_from_iterator(
    yield_tokens(enToken),
    specials=SPECIAL_TOKENS,
    max_tokens=MAX_VOCAB
)

en_vocab.set_default_index(en_vocab[UNK_TOKEN])

# ==== French vocab ====

fr_vocab = build_vocab_from_iterator(
    yield_tokens(frToken),
    specials=SPECIAL_TOKENS,
    max_tokens=MAX_VOCAB
)

fr_vocab.set_default_index(fr_vocab[UNK_TOKEN])

# ==== Check vocab size ====
print("English vocab size:", len(en_vocab))
print("French vocab size:", len(fr_vocab))
print("Two: ",en_vocab(['Two']))
print("Young: ",en_vocab(['young']))
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

import torch
from torch.nn.utils.rnn import pad_sequence
def collate_fn(sentences, vocab, pad_token='<pad>'):
    pad_idx = vocab[pad_token]
    batch_indices = [torch.tensor([vocab[token] for token in sentence]) for sentence in sentences]
    
    lengths = torch.tensor([len(seq) for seq in batch_indices])
    
    src_padded = pad_sequence(batch_indices, batch_first=True, padding_value=pad_idx)
    return src_padded, lengths
from torch.utils.data import DataLoader
from functools import partial

loader = DataLoader(
    enToken,
    batch_size=128,
    shuffle=False,  # QUAN TRỌNG
    collate_fn=partial(collate_fn, vocab=en_vocab)
)

# for batch_idx, (padded_batch, lengths) in enumerate(loader):
#     print(f"\n=== Batch {batch_idx} ===")
#     print("Padded batch shape:", padded_batch.shape)
    # print("Lengths:", lengths)

# for batch_idx, (padded_batch, lengths) in enumerate(loader):
#     # Đây là lengths gốc của 5 câu trong batch
#     lengths_sorted, sorted_idx = torch.sort(lengths, descending=True)
#     padded_sorted = padded_batch[sorted_idx]
    
#     packed_input = pack_padded_sequence(padded_sorted, lengths_sorted, batch_first=True, enforce_sorted=True)


for padded_batch, lengths in loader:

    for i in range(len(lengths)):                      # 1 câu
        real_len = lengths[i]
        sentence_tokens = padded_batch[i, :real_len]   # bỏ pad

        # embed: numpy
        X = [E[token].reshape(-1,1) for token in sentence_tokens]

        outputs, (h, c) = lstm_numpy.forward_sequence(X)

