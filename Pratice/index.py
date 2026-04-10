import torch
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds_raw = load_dataset("thainq07/iwslt2015-en-vi")

VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<pad>","<s>","</s>","<unk>"]
PAD_ID, SOS_ID, EOS_ID, UNK_ID = 0,1,2,3

all_en = list(ds_raw["train"]["en"])
all_vi = list(ds_raw["train"]["vi"])

all_texts = all_en + all_vi

def build_word_tokenizer(texts, vocab_size = VOCAB_SIZE):
    tok = Tokenizer
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(
        vocab_size=vocab_size, special_tokens= SPECIAL_TOKENS
    )
    tok.train_from_iterator(texts, trainer)
    return tok

def build_bpe_tokenizer(texts, vocab_size = VOCAB_SIZE):
    tok = Tokenizer
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens= SPECIAL_TOKENS
    )
    tok.train_from_iterator(texts, trainer)
    return tok

def build_byte_bpe_tokenizer(texts, vocab_size = VOCAB_SIZE):
    tok = Tokenizer
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space= False)
    trainer = trainers.BpeTrainer(
        vocab_size= vocab_size, 
        special_tokens= SPECIAL_TOKENS,
        initial_alphabet= pre_tokenizers.ByteLevel.alphabet()
    )
    tok.train_from_iterator(texts, trainer)
    tok.decode = decoders.ByteLevel()
    return tok

tokenizers_dict = {
    "Word": build_word_tokenizer,
    "BPE" : build_bpe_tokenizer,
    "Byte-BPE": build_byte_bpe_tokenizer
}

for name, tok in tokenizers_dict.items():
    print(f"{name:12s} vocab size: {tok.get}")