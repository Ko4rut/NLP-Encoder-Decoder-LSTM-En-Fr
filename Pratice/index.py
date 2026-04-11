import torch
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds_raw = load_dataset("thainq107/iwslt2015-en-vi")

VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<pad>","<s>","</s>","<unk>"]
PAD_ID, SOS_ID, EOS_ID, UNK_ID = 0,1,2,3

all_en = list(ds_raw["train"]["en"])
all_vi = list(ds_raw["train"]["vi"])

all_texts = all_en + all_vi

def build_word_tokenizer(texts, vocab_size = VOCAB_SIZE):
    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(
        vocab_size=vocab_size, special_tokens= SPECIAL_TOKENS
    )
    tok.train_from_iterator(texts, trainer)
    return tok

def build_bpe_tokenizer(texts, vocab_size = VOCAB_SIZE):
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens= SPECIAL_TOKENS
    )
    tok.train_from_iterator(texts, trainer)
    return tok

def build_byte_bpe_tokenizer(texts, vocab_size=VOCAB_SIZE):
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tok.train_from_iterator(texts, trainer)
    tok.decoder = decoders.ByteLevel()
    return tok

tokenizers_dict = {
    "Word": build_word_tokenizer(all_texts),
    "BPE" : build_bpe_tokenizer(all_texts),
    "Byte-BPE": build_byte_bpe_tokenizer(all_texts)
}

for name, tok in tokenizers_dict.items():
    tok: Tokenizer
    print(f"{name:12s} vocab size: {tok.get_vocab_size()}")

MAX_LEN = 75

def make_preprocess_fn(tok):
    def encode_text(text):
        ids = tok.encode(text).ids
        ids = [SOS_ID] + ids[:MAX_LEN - 2] + [EOS_ID]
        ids = ids + [PAD_ID] * (MAX_LEN - len(ids))
        return ids
    
    def preprocess(example):
        src_ids = encode_text(example["en"])
        tgt_ids = encode_text(example["vi"])
        return {"src_ids": src_ids, "tgt_ids": tgt_ids}
    
    return preprocess

datasets_dict = {}
for name, tok in tokenizers_dict.items():
    ds_copy = ds_raw.map(make_preprocess_fn(tok), remove_columns=["en","vi"])
    datasets_dict["name"] = ds_copy
    print(f"{name:12s} preprocessed train: {len(ds_copy['train'])}")

import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)   # (S, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (S, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, S, D)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, S, D)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(d_model, max_len=MAX_LEN, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=False
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=False
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def encode(self, src):
        src_pad_mask = (src == PAD_ID)              # (B, S)
        src_emb = self.pos(self.embed(src))         # (B, S, D)
        memory = self.encoder(
            src_emb.transpose(0, 1),               # (S, B, D)
            src_key_padding_mask=src_pad_mask
        )
        return memory, src_pad_mask

    def decode_step(self, tgt, memory, src_pad_mask):
        tgt_pad_mask = (tgt == PAD_ID)             # (B, T)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1), device=tgt.device
        ).bool()

        tgt_emb = self.pos(self.embed(tgt))        # (B, T, D)
        out = self.decoder(
            tgt_emb.transpose(0, 1),               # (T, B, D)
            memory,                                # (S, B, D)
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )
        return self.fc_out(out.transpose(0, 1))    # (B, T, V)

    def forward(self, src, tgt):
        memory, src_pad_mask = self.encode(src)
        return self.decode_step(tgt[:, :-1], memory, src_pad_mask)
    
BATCH = 512
EPOCHS = 30
# =========================
# FIX DATASET DICT
# =========================
datasets_dict = {}
for name, tok in tokenizers_dict.items():
    ds_copy = ds_raw.map(make_preprocess_fn(tok), remove_columns=["en", "vi"])
    ds_copy.set_format(type="torch", columns=["src_ids", "tgt_ids"])
    datasets_dict[name] = ds_copy
    print(f"{name:12s} preprocessed train: {len(ds_copy['train'])}")

# =========================
# HYPERPARAMS
# =========================
BATCH = 128
EPOCHS = 10
LR = 1e-3

# =========================
# DATALOADER
# =========================
def collate_fn(batch):
    src = torch.stack([item["src_ids"] for item in batch])
    tgt = torch.stack([item["tgt_ids"] for item in batch])
    return src, tgt

def make_loaders(ds):
    train_loader = DataLoader(
        ds["train"], batch_size=BATCH, shuffle=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        ds["validation"], batch_size=BATCH, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        ds["test"], batch_size=BATCH, shuffle=False, collate_fn=collate_fn
    )
    return train_loader, valid_loader, test_loader

# =========================
# LOSS / TRAIN / EVAL
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()

        # logits: (B, T-1, V)
        logits = model(src, tgt)

        # target thật: bỏ token đầu <s>
        target = tgt[:, 1:]

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        logits = model(src, tgt)
        target = tgt[:, 1:]

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1)
        )
        total_loss += loss.item()

    return total_loss / len(loader)

# =========================
# GREEDY DECODING
# =========================
@torch.no_grad()
def greedy_decode(model, src, max_len=MAX_LEN):
    model.eval()

    memory, src_pad_mask = model.encode(src)

    batch_size = src.size(0)
    ys = torch.full((batch_size, 1), SOS_ID, dtype=torch.long, device=src.device)

    for _ in range(max_len - 1):
        logits = model.decode_step(ys, memory, src_pad_mask)   # (B, T, V)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
        ys = torch.cat([ys, next_token], dim=1)

        # nếu tất cả batch đều đã EOS thì dừng
        if (next_token.squeeze(1) == EOS_ID).all():
            break

    return ys

# =========================
# IDS -> TEXT
# =========================
def decode_ids(ids, tok):
    # bỏ pad, sos, cắt tại eos
    clean_ids = []
    for i in ids:
        if i == EOS_ID:
            break
        if i not in [PAD_ID, SOS_ID]:
            clean_ids.append(int(i))

    try:
        return tok.decode(clean_ids)
    except:
        return ""

# =========================
# BLEU
# =========================
@torch.no_grad()
def compute_bleu(model, loader, tok, device, max_batches=None):
    model.eval()

    references = []
    hypotheses = []

    for batch_idx, (src, tgt) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        src = src.to(device)
        pred_ids = greedy_decode(model, src, max_len=MAX_LEN)

        pred_ids = pred_ids.cpu().tolist()
        tgt = tgt.cpu().tolist()

        for pred_seq, tgt_seq in zip(pred_ids, tgt):
            pred_text = decode_ids(pred_seq, tok).strip()
            tgt_text = decode_ids(tgt_seq, tok).strip()

            pred_tokens = pred_text.split()
            tgt_tokens = tgt_text.split()

            if len(pred_tokens) == 0:
                pred_tokens = ["<empty>"]
            if len(tgt_tokens) == 0:
                tgt_tokens = ["<empty>"]

            hypotheses.append(pred_tokens)
            references.append([tgt_tokens])

    smoothie = SmoothingFunction().method1
    bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
    return bleu

# =========================
# TRAIN ALL TOKENIZERS
# =========================
results = {}

for name, tok in tokenizers_dict.items():
    print(f"\n==================== {name} ====================")

    ds = datasets_dict[name]
    train_loader, valid_loader, test_loader = make_loaders(ds)

    vocab_size = tok.get_vocab_size()
    model = TransformerSeq2Seq(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = evaluate_loss(model, valid_loader, criterion, DEVICE)

        print(f"[{name}] Epoch {epoch:02d}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    bleu = compute_bleu(model, test_loader, tok, DEVICE, max_batches=100)
    print(f"[{name}] TEST BLEU = {bleu:.4f}")

    results[name] = {
        "best_val_loss": best_val_loss,
        "bleu": bleu
    }

print("\n===== FINAL RESULTS =====")
for name, info in results.items():
    print(f"{name:12s} | best_val_loss = {info['best_val_loss']:.4f} | BLEU = {info['bleu']:.4f}")