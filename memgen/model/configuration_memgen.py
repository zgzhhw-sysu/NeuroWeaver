from transformers import PretrainedConfig
from typing import Optional


class MemGenConfig(PretrainedConfig):
    model_type = "memgen"

    def __init__(
        self,
        # weaver configs
        weaver_lora_config: Optional[dict] = None,
        prompt_latents_len: int = 0,
        inference_latents_len: int = 0,
        # NeuroWeave v2.2: Hyper-LoRA Weaver
        weaver_use_hyper_lora: bool = False,
        hyper_lora_rank: int = 16,
        # NeuroWeave v2.2: G-Mem (GRU memory)
        use_gru_memory: bool = False,
        gru_hidden_size: int = 0,
        bptt_trigger_steps: int = 4,
        # trigger configs
        trigger_active: bool = False,
        trigger_lora_config: Optional[dict] = None,
        max_prompt_aug_num: int = 1,
        max_inference_aug_num: int = 5,
        # NeuroWeave v2.2: Sparse Varentropy Trigger (2.3)
        use_sparse_varentropy_trigger: bool = False,
        varentropy_period_k: int = 1,
        varentropy_threshold_tau: float = 2.0,
        varentropy_tau_trainable: bool = False,
        varentropy_top_k: int = 50,
        punct_token_ids: Optional[list] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # weaver configs
        self.weaver_lora_config = weaver_lora_config
        self.prompt_latents_len = prompt_latents_len
        self.inference_latents_len = inference_latents_len
        self.weaver_use_hyper_lora = weaver_use_hyper_lora
        self.hyper_lora_rank = hyper_lora_rank
        self.use_gru_memory = use_gru_memory
        self.gru_hidden_size = gru_hidden_size
        self.bptt_trigger_steps = bptt_trigger_steps

        # trigger configs
        self.trigger_active = trigger_active
        self.trigger_lora_config = trigger_lora_config
        self.max_prompt_aug_num = max_prompt_aug_num
        self.max_inference_aug_num = max_inference_aug_num
        # NeuroWeave v2.2: Sparse Varentropy Trigger
        self.use_sparse_varentropy_trigger = use_sparse_varentropy_trigger
        self.varentropy_period_k = varentropy_period_k
        self.varentropy_threshold_tau = varentropy_threshold_tau
        self.varentropy_tau_trainable = varentropy_tau_trainable
        self.varentropy_top_k = varentropy_top_k
        self.punct_token_ids = punct_token_ids or []