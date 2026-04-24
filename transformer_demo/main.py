import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import *
from model import SimpleTransformer
from utils import subsequent_mask

# ====== 伪数据（你可以换真实数据集） ======
data = [
    ("ein mann spielt gitarre", "a man plays guitar"),
    ("zwei frauen laufen", "two women are running"),
    ("ein kind spielt", "a child plays"),
]

src_sentences = [x[0] for x in data]
tgt_sentences = [x[1] for x in data]

src_vocab = build_vocab(src_sentences, tokenize_de)
tgt_vocab = build_vocab(tgt_sentences, tokenize_en)

dataset = list(zip(src_sentences, tgt_sentences))

loader = DataLoader(
    dataset,
    batch_size=2,
    collate_fn=lambda b: collate_fn(b, src_vocab, tgt_vocab)
)

# ====== 模型 ======
model = SimpleTransformer(len(src_vocab), len(tgt_vocab))
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ====== 训练 ======
for epoch in range(10):
    for src, tgt in loader:
        optimizer.zero_grad()

        tgt_input = tgt[:-1]
        tgt_output = tgt[1:]

        src_mask = (src != 0).unsqueeze(1)
        tgt_mask = subsequent_mask(tgt_input.size(0))

        out = model(src, tgt_input, src_mask, tgt_mask)

        loss = criterion(
            out.reshape(-1, out.shape[-1]),
            tgt_output.reshape(-1)
        )

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ====== 推理 ======
def greedy_decode(model, src, max_len=10):
    src_mask = (src != 0).unsqueeze(1)
    memory = model.transformer.encoder(model.src_embed(src))

    ys = torch.ones(1, 1).fill_(1).long()

    for _ in range(max_len):
        tgt_mask = subsequent_mask(ys.size(0))
        out = model.transformer.decoder(
            model.tgt_embed(ys), memory, tgt_mask=tgt_mask[0]
        )
        prob = model.fc(out[-1])
        next_word = prob.argmax().item()

        ys = torch.cat([ys, torch.tensor([[next_word]])], dim=0)

    return ys

# 测试
test_sentence = "ein mann spielt gitarre"
src = torch.tensor(
    [[1] + numericalize(test_sentence, src_vocab, tokenize_de) + [2]]
).T

result = greedy_decode(model, src)

id2word = {v: k for k, v in tgt_vocab.items()}

def decode_tokens(token_ids, id2word):
    words = []
    for tid in token_ids:
        word = id2word.get(tid, "<unk>")
        if word == "</s>":
            break
        if word not in ["<s>", "<pad>"]:
            words.append(word)
    return " ".join(words)

tokens = result.squeeze().tolist()
sentence = decode_tokens(tokens, id2word)

print("输入:", test_sentence)
print("输出:", sentence)

test_sentences = [
    "ein mann spielt gitarre",
    "zwei frauen laufen",
    "ein kind spielt",
]

for s in test_sentences:
    src = torch.tensor(
        [[1] + numericalize(s, src_vocab, tokenize_de) + [2]]
    ).T

    result = greedy_decode(model, src)
    tokens = result.squeeze().tolist()
    sentence = decode_tokens(tokens, id2word)

    print(f"\n输入: {s}")
    print(f"输出: {sentence}")
