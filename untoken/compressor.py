import torch
import torch.nn as nn
from transformers import DistilBertConfig, DistilBertModel, PretrainedConfig, PreTrainedModel


class UntokenCompressorConfig(PretrainedConfig):
    model_type = "untoken_compressor"

    def __init__(self, dropout=0.1, head_hidden=256, target_ratio=0.3, **kwargs):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.head_hidden = head_hidden
        self.target_ratio = target_ratio


class UntokenCompressor(PreTrainedModel):
    config_class = UntokenCompressorConfig

    def __init__(self, config: UntokenCompressorConfig):
        super().__init__(config)
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        self.encoder = DistilBertModel(DistilBertConfig())
        hidden = self.encoder.config.dim

        self.head = nn.Sequential(
            nn.Linear(hidden, config.head_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden, 1),
            nn.Sigmoid(),
        )
        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad_(True)

    def forward(self, input_ids, attention_mask, special_tokens_mask=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        scores = self.head(hidden).squeeze(-1)

        if special_tokens_mask is not None:
            scores = scores.masked_fill(special_tokens_mask.bool(), 0.0)

        scores = scores * attention_mask.float()
        return scores

    def get_keep_mask(self, scores, target_ratio, attention_mask):
        B, L = scores.shape
        mask = torch.zeros_like(scores)
        for i in range(B):
            seq_len = attention_mask[i].sum().item()
            k = max(1, int(seq_len * target_ratio))
            topk_idx = scores[i, :seq_len].topk(k).indices
            mask[i, topk_idx] = 1.0
        return mask
