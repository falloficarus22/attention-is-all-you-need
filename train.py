import torch
import torch.nn as nn
import torch.optim as optim
from transformer import Transformer

# Hyperparameters
src_vocab = tgt_vocab = 50
d_model = 64
num_heads = 4
d_ff = 128
num_layers = 2
max_len = 20

model = Transformer(src_vocab, tgt_vocab, d_model, num_heads, d_ff, num_layers, max_len)
criterion = nn.CrossEntropyLoss(ignore_index = 0)
optimizer = optim.AdamW(model.parameters(), lr = 0.001)

# Toy data: random integer sequences
batch_size = 16
src_seq_len = 10
tgt_seq_len = 10
# Random input (values 1 to vocab-1)
src = torch.randint(1, src_vocab, (batch_size, src_seq_len))
# Let target = input (copy task), add a start token 1 at beginning
tgt_input = torch.cat([torch.ones(batch_size, 1, dtype=torch.long), src[:, :-1]], dim=1)
tgt_output = src  # model should predict the source sequence

# Masks (None here for simplicity; in real tasks handle padding and look-ahead)
src_mask = None
tgt_mask = None

for epoch in range(100):
    # Training step
    model.train()
    optimizer.zero_grad()
    logits = model(src, tgt_input, src_mask, tgt_mask)  
    # logits: (batch, tgt_len, vocab)
    loss = criterion(logits.view(-1, tgt_vocab), tgt_output.view(-1))
    loss.backward()
    optimizer.step()
    print(f"Training loss: {loss.item():.4f}")

