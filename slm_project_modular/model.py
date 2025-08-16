import torch
import torch.nn as nn

class TransformerSLM(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=256, n_heads=4, n_layers=3, max_len=256, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
                                               dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.cls_head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B, T = x.size()
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos)
        h = self.encoder(h)
        lm_logits = self.lm_head(h)          # (B, T, V)
        cls_logits = self.cls_head(h[:, 0])  # (B, C)
        return lm_logits, cls_logits
