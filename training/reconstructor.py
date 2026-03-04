import torch
import torch.nn as nn
from transformers import AutoTokenizer


class UntokenReconstructor(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 max_seq_len: int = 512):
        super().__init__()
        self.d_model = d_model

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def build_memory(self, compressed_ids: torch.Tensor, compressed_positions: torch.Tensor) -> torch.Tensor:
        return self.token_embed(compressed_ids) + self.pos_embed(compressed_positions)

    def forward(self, compressed_ids, compressed_positions, target_ids, compressed_mask=None):
        memory = self.build_memory(compressed_ids, compressed_positions)

        B, T = target_ids.shape
        positions = torch.arange(T, device=target_ids.device).unsqueeze(0).expand(B, -1)
        tgt = self.token_embed(target_ids) + self.pos_embed(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=target_ids.device)

        memory_key_padding_mask = ~compressed_mask.bool() if compressed_mask is not None else None

        out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        out = self.norm(out)
        return self.output_proj(out)
