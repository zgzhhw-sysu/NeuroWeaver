from typing import Optional

import torch

# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)

def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])

def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])

def generate_position_ids(attention_mask):
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    position_ids.masked_fill_(attention_mask == 0, 0)
    return position_ids


# --- NeuroWeave v2.2: Gated R-GRPO (2.4) ---

def compute_length_penalty(
    completion_lengths: torch.Tensor,
    L_target: float,
    tau: float,
) -> torch.Tensor:
    """P_len = tanh((L - L_target) / tau). Shape (N,)."""
    L = completion_lengths.float()
    return torch.tanh((L - L_target) / max(tau, 1e-6))


def compute_repetition_penalty(
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    ngram: int = 3,
) -> torch.Tensor:
    """N-gram repetition ratio per sequence. Shape (N,). Higher = more repetitive."""
    N, L = completion_ids.shape
    device = completion_ids.device
    out = torch.zeros(N, device=device, dtype=torch.float32)
    for b in range(N):
        ids = completion_ids[b][completion_mask[b] > 0].tolist()
        if len(ids) < ngram:
            continue
        ngrams = set()
        repeats = 0
        for i in range(len(ids) - ngram + 1):
            ng = tuple(ids[i : i + ngram])
            if ng in ngrams:
                repeats += 1
            ngrams.add(ng)
        total = max(len(ids) - ngram + 1, 1)
        out[b] = repeats / total
    return out


def dynamic_beta(
    acc_batch: float,
    acc_target: float,
    beta_max: float,
    k: float,
) -> float:
    """β_t = β_max * σ(k * (acc_batch - acc_target))."""
    return float(beta_max * torch.sigmoid(torch.tensor(k * (acc_batch - acc_target))).item())


def gated_reward(
    rewards_per_func: torch.Tensor,
    outcome_idx: int,
    completion_lengths: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    length_penalty_target: float,
    length_penalty_tau: float,
    beta_t: float,
    gamma_rep: float = 0.0,
    ngram_rep: int = 3,
    alpha_density: float = 0.0,
    R_density: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    R = R_outcome + I_correct * (α * R_density - β_t * P_len) - γ * P_rep.
    R_density optional; if None, α*R_density is 0.
    """
    R_outcome = rewards_per_func[:, outcome_idx].float()
    I_correct = (R_outcome > 0.5).float()
    P_len = compute_length_penalty(
        completion_lengths, length_penalty_target, length_penalty_tau
    )
    R = R_outcome + I_correct * (alpha_density * (R_density if R_density is not None else torch.zeros_like(R_outcome)) - beta_t * P_len)
    if gamma_rep > 0 and ngram_rep >= 2:
        P_rep = compute_repetition_penalty(completion_ids, completion_mask, ngram=ngram_rep)
        R = R - gamma_rep * P_rep
    return R