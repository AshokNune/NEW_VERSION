import torch
import torch.nn as nn

class TransformerSLM(nn.Module):
    """
    Small Transformer encoder with:
      - token + positional embeddings
      - encoder layers (batch_first=True)
      - LM head (next-token)
      - classification head on first token ([CLS])
    """
    def __init__(self, vocab_size, num_classes, d_model=256, n_heads=4, n_layers=3, max_len=256, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.lm_head = nn.Linear(d_model, vocab_size)
        self.cls_head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B, T = x.size()
        pos_idx = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos_idx)
        h = self.encoder(h)

        lm_logits = self.lm_head(h)          # (B, T, V)
        cls_logits = self.cls_head(h[:, 0])  # (B, C) using first token as [CLS]
        return lm_logits, cls_logits
