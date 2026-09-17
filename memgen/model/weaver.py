from peft import LoraConfig, PeftModel
from typing import Optional

import torch
import torch.nn as nn


def _zero_init_linear(linear: nn.Linear) -> None:
    """Zero-initialize output layer so training starts from W_t = W_base (no dynamic delta)."""
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class MemGenWeaver(nn.Module):

    adapter_name = "weaver"

    def __init__(
        self,
        model: PeftModel,
        prompt_latents_len: int,
        inference_latents_len: int,
        use_hyper_lora: bool = False,
        hyper_lora_rank: int = 16,
    ):
        super().__init__()
        self.model = model
        d = model.base_model.config.hidden_size
        self._use_hyper_lora = use_hyper_lora
        self._hyper_lora_rank = hyper_lora_rank

        # prompt augmentation (always used as base latents; when hyper_lora, add dynamic delta)
        self.prompt_query_latents = nn.Parameter(
            torch.randn(prompt_latents_len, d),
            requires_grad=True,
        )
        self.inference_query_latents = nn.Parameter(
            torch.randn(inference_latents_len, d),
            requires_grad=True,
        )

        if use_hyper_lora:
            r = hyper_lora_rank
            # H_φ^A: h_t -> A_t (d, r). Output layer zero-init => A_t=0 at start.
            self._hyper_net_A = nn.Sequential(
                nn.Linear(d, d),
                nn.GELU(),
                nn.Linear(d, d * r),
            )
            _zero_init_linear(self._hyper_net_A[-1])
            # H_φ^B: h_t -> B_t (r, d). Output layer zero-init => B_t=0 at start.
            self._hyper_net_B = nn.Sequential(
                nn.Linear(d, d),
                nn.GELU(),
                nn.Linear(d, r * d),
            )
            _zero_init_linear(self._hyper_net_B[-1])
            self._b_t = nn.Parameter(torch.zeros(d))
            # Expand M_t (B, d) -> (B, latent_len, d). Zero-init => at start latents = base only.
            self._expand_prompt = nn.Linear(d, prompt_latents_len * d)
            _zero_init_linear(self._expand_prompt)
            self._expand_inference = nn.Linear(d, inference_latents_len * d)
            _zero_init_linear(self._expand_inference)
    
    @property
    def prompt_latents_num(self) -> int:
        return self.prompt_query_latents.size(0)

    @property
    def inference_latents_num(self) -> int:
        return self.inference_query_latents.size(0)

    @property
    def device(self):
        assert self.prompt_query_latents.device == self.inference_query_latents.device
        return self.prompt_query_latents.device

    def _compute_hyper_lora_latents(
        self,
        h_t: torch.Tensor,
        base_latents: torch.Tensor,
        expand_layer: nn.Linear,
        latent_len: int,
    ) -> torch.Tensor:
        """Compute (B, latent_len, d) = base_latents + expand(M_t), M_t = σ(B_t A_t h_t + b_t). Zero-init => start from base only."""
        B, d = h_t.shape
        r = self._hyper_lora_rank
        A_flat = self._hyper_net_A(h_t)
        A_t = A_flat.view(B, d, r)
        B_flat = self._hyper_net_B(h_t)
        B_t = B_flat.view(B, r, d)
        W_t = torch.bmm(A_t, B_t)
        M_t = torch.sigmoid(torch.bmm(W_t, h_t.unsqueeze(-1)).squeeze(-1) + self._b_t)
        delta = expand_layer(M_t).view(B, latent_len, d)
        base = base_latents.unsqueeze(0).expand(B, -1, -1)
        return base + delta

    def _augment(
        self,
        latents: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.model.set_adapter(self.adapter_name)

        batch_size = attention_mask.shape[0]
        if latents.dim() == 2:
            latents_num = latents.size(0)
            latents = latents.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            assert latents.dim() == 3 and latents.size(0) == batch_size
            latents_num = latents.size(1)
        
        # inputs_embeds
        inputs_embeds = torch.cat([inputs_embeds, latents], dim=1)

        # attention_mask: (B, L_total)
        latents_mask = torch.ones(latents.shape[:-1], dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, latents_mask], dim=1)
        
        # get position ids
        last_position_ids = position_ids.max(dim=1)[0]
        latents_relative_positions = torch.arange(latents_num, device=attention_mask.device)
        latents_position_ids = last_position_ids.unsqueeze(1) + latents_relative_positions + 1
        position_ids = torch.cat([position_ids.long(), latents_position_ids.long()], dim=1) 

        # the processor only outputs the hidden states
        assert inputs_embeds.shape[:2] == attention_mask.shape == position_ids.shape

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,  
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        latents_hidden_states = hidden_states[:, -latents_num:, :]

        self.model.disable_adapter()  

        return latents_hidden_states, latents_mask, latents_position_ids

    def augment_prompt(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        h_t: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._use_hyper_lora:
            if h_t is None:
                h_t = inputs_embeds[:, -1, :]
            latents = self._compute_hyper_lora_latents(
                h_t,
                self.prompt_query_latents,
                self._expand_prompt,
                self.prompt_latents_num,
            )
            return self._augment(
                latents=latents,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        return self._augment(
            latents=self.prompt_query_latents,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    def augment_inference(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        h_t: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._use_hyper_lora:
            if h_t is None:
                h_t = inputs_embeds[:, -1, :]
            latents = self._compute_hyper_lora_latents(
                h_t,
                self.inference_query_latents,
                self._expand_inference,
                self.inference_latents_num,
            )
            return self._augment(
                latents=latents,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        return self._augment(
            latents=self.inference_query_latents,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )