from typing import Optional

from peft import LoraConfig, PeftModel
import torch
import torch.nn as nn
import torch.nn.functional as F


def entropy_from_logits_topk(logits: torch.Tensor, top_k: int = 50) -> torch.Tensor:
    """Entropy H(P) from logits using only top-K logits (numerically stable, faster).
    logits: (batch, vocab_size). Returns (batch,) entropy in nats.
    """
    if top_k <= 0 or top_k >= logits.size(-1):
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1).clamp(min=-1e6)
        return -(probs * log_probs).sum(dim=-1)
    k = min(top_k, logits.size(-1))
    top_logits, _ = torch.topk(logits, k, dim=-1)
    probs = F.softmax(top_logits, dim=-1)
    log_probs = F.log_softmax(top_logits, dim=-1).clamp(min=-1e6)
    return -(probs * log_probs).sum(dim=-1)


class MemGenTrigger(nn.Module):
    adapter_name = "trigger"

    def __init__(
        self,
        model: PeftModel,
        active: bool,
        config: Optional[object] = None,
    ):
        super().__init__()
        
        self.active = active
        self.model = model
        self.output_layer = nn.Linear(model.base_model.config.hidden_size, 2)

        # NeuroWeave v2.2: Sparse Varentropy — τ_v (trainable or fixed)
        use_sparse = getattr(config, "use_sparse_varentropy_trigger", False) if config else False
        tau_trainable = getattr(config, "varentropy_tau_trainable", False) if config else False
        tau_init = float(getattr(config, "varentropy_threshold_tau", 2.0) if config else 2.0)
        if use_sparse and tau_trainable:
            self.varentropy_tau = nn.Parameter(torch.tensor(tau_init, dtype=torch.float32))
        else:
            self.register_buffer("varentropy_tau", torch.tensor(tau_init, dtype=torch.float32))
        self.use_sparse_varentropy_trigger = use_sparse
        self.varentropy_top_k = int(getattr(config, "varentropy_top_k", 50) if config else 50)

    def get_varentropy_threshold(self) -> torch.Tensor:
        return self.varentropy_tau

    def forward(
        self, 
        input_ids: torch.LongTensor, 
        attention_mask: torch.LongTensor,
        position_ids: torch.Tensor
    ) -> torch.FloatTensor:
        
        if self.active:
            self.model.set_adapter(self.adapter_name)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,  
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1]
            logits = self.output_layer(hidden_states)
            self.model.disable_adapter()
        
        else:
            batch_size, seq_len = input_ids.shape
            logits = torch.zeros(batch_size, seq_len, 2, device=input_ids.device)  # logits: [batch_size, seq_len, 2]
            logits[..., 1] = 1.0  

        return logits



