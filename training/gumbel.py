import torch
import torch.nn.functional as F


def gumbel_soft_select(keep_probs: torch.Tensor, temperature: float) -> torch.Tensor:
    keep_probs = keep_probs.clamp(1e-6, 1 - 1e-6)
    log_probs = torch.stack([keep_probs.log(), (1 - keep_probs).log()], dim=-1)
    soft = F.gumbel_softmax(log_probs, tau=temperature, hard=False, dim=-1)
    return soft[..., 0]


def straight_through_select(keep_probs: torch.Tensor) -> torch.Tensor:
    hard = (keep_probs > 0.5).float()
    return keep_probs + (hard - keep_probs).detach()


def apply_compression_mask(
    input_ids: torch.Tensor,
    mask: torch.Tensor,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, L = input_ids.shape
    binary = (mask > 0.5).float()

    max_kept = int(binary.sum(dim=1).max().item())
    max_kept = max(max_kept, 1)

    compressed_ids = torch.full((B, max_kept), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
    compressed_positions = torch.zeros(B, max_kept, dtype=torch.long, device=input_ids.device)
    compressed_mask = torch.zeros(B, max_kept, dtype=torch.long, device=input_ids.device)

    for i in range(B):
        indices = binary[i].nonzero(as_tuple=True)[0]
        k = len(indices)
        if k == 0:
            # Guarantee at least the top-scoring token so the reconstructor
            # never receives an all-padding sequence (which causes NaN attention).
            indices = mask[i].argmax(keepdim=True)
            k = 1
        compressed_ids[i, :k] = input_ids[i, indices]
        compressed_positions[i, :k] = indices
        compressed_mask[i, :k] = 1

    return compressed_ids, compressed_positions, compressed_mask


def get_temperature(step: int, total_steps: int, t_start: float = 1.0, t_end: float = 0.1) -> float:
    if total_steps <= 0:
        return t_end
    progress = min(step / total_steps, 1.0)
    return t_start * (t_end / t_start) ** progress
