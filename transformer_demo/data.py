import torch
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

try:
    import spacy  # type: ignore
except ImportError:
    spacy = None


def _load_tokenizer(model_name):
    if spacy is None:
        return None

    try:
        return spacy.load(model_name).tokenizer
    except OSError:
        return None


spacy_de = _load_tokenizer("de_core_news_sm")
spacy_en = _load_tokenizer("en_core_web_sm")

def tokenize_de(text):
    if spacy_de is not None:
        return [tok.text.lower() for tok in spacy_de(text)]
    return text.lower().split()

def tokenize_en(text):
    if spacy_en is not None:
        return [tok.text.lower() for tok in spacy_en(text)]
    return text.lower().split()

def build_vocab(sentences, tokenizer):
    counter = Counter()
    for s in sentences:
        counter.update(tokenizer(s))

    vocab = {"<pad>":0, "<s>":1, "</s>":2, "<unk>":3}
    for word in counter:
        vocab[word] = len(vocab)

    return vocab

def numericalize(sentence, vocab, tokenizer):
    return [vocab.get(t, vocab["<unk>"]) for t in tokenizer(sentence)]

def collate_fn(batch, src_vocab, tgt_vocab):
    src_batch, tgt_batch = [], []

    for src, tgt in batch:
        src_ids = [1] + numericalize(src, src_vocab, tokenize_de) + [2]
        tgt_ids = [1] + numericalize(tgt, tgt_vocab, tokenize_en) + [2]

        src_batch.append(torch.tensor(src_ids))
        tgt_batch.append(torch.tensor(tgt_ids))

    src_batch = pad_sequence(src_batch, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, padding_value=0)

    return src_batch, tgt_batch
