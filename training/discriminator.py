import torch
import torch.nn as nn


class UntokenDiscriminator(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 4,
                 num_layers: int = 4, dim_feedforward: int = 1024, dropout: float = 0.1,
                 max_seq_len: int = 512):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_embed(input_ids) + self.pos_embed(positions)

        src_key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        if attention_mask is not None:
            mask_f = attention_mask.float().unsqueeze(-1)
            pooled = (x * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        else:
            pooled = x.mean(1)

        return self.classifier(pooled)
