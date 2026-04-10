import torch
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

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

