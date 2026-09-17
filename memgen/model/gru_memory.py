"""
G-Mem: GRU-based gated memory evolution (NeuroWeave v2.2).
State S_t is updated at each augment point; S_t is fed back into the next weaver input.
"""
import torch
import torch.nn as nn


class GMemCell(nn.Module):
    """
    Single-step GRU cell for memory evolution.
    Input: x_t (current weaver output summary), S_{t-1} (previous state).
    Output: S_t = (1 - z_t) * S_{t-1} + z_t * S̃_t.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = nn.GRUCell(input_size, hidden_size)

    def forward(
        self,
        x_t: torch.Tensor,
        s_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_t: (batch, input_size) current step input (e.g. pooled weaver hidden).
            s_prev: (batch, hidden_size) previous memory state.
        Returns:
            s_t: (batch, hidden_size) new memory state.
        """
        return self.cell(x_t, s_prev)

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return zero initial state (batch, hidden_size)."""
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
