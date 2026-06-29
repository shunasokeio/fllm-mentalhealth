"""Model loading, tokenizer, data collator, dual-LoRA adapter management."""

import copy
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Sequence

import bitsandbytes as bnb
import numpy as np
import torch
import transformers
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft.tuners.lora import LoraLayer
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaTokenizer,
)

from .config import DEFAULT_PAD_TOKEN, IGNORE_INDEX, ExperimentConfig


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def find_all_linear_names(model, bits=4):
    """Find all linear layer names for LoRA target modules."""
    cls = bnb.nn.Linear4bit if bits == 4 else (
        bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    lora_module_names.discard("lm_head")
    return list(lora_module_names)


def get_model(config: ExperimentConfig, gpu_id: int = 0):
    """Load model with 4-bit quantization and a single LoRA adapter ('default').

    gpu_id pins all layers to a single device so accelerate's prepare() doesn't
    see a multi-device model, which is unsupported with 4-bit quantization.
    """
    use_cuda = torch.cuda.is_available()
    compute_dtype = torch.bfloat16 if use_cuda else torch.float32
    model_cfg = config.model
    lora_cfg = config.lora

    quantization_config = None
    if use_cuda:
        if model_cfg.quantization == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            raise ValueError(f"Only 4-bit quantization supported, got {model_cfg.quantization}")

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
        device_map={"": gpu_id} if use_cuda else None,
    )

    setattr(model, "model_parallel", True)
    setattr(model, "is_parallelizable", True)

    if use_cuda:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
        )

    # Determine target modules
    target_modules = lora_cfg.target_modules
    if target_modules == "all-linear":
        target_modules = find_all_linear_names(model, bits=model_cfg.quantization)
    elif isinstance(target_modules, str):
        target_modules = [target_modules]

    peft_config = LoraConfig(
        r=lora_cfg.rank,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        bias=lora_cfg.bias,
    )

    peft_model = get_peft_model(model, peft_config)
    if not use_cuda:
        peft_model.enable_input_require_grads()

    # Dtype adjustments
    for name, module in peft_model.named_modules():
        if isinstance(module, LoraLayer):
            module = module.to(compute_dtype)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight") and module.weight.dtype == torch.float32:
                module = module.to(compute_dtype)

    if model_cfg.gradient_checkpointing:
        peft_model.config.use_cache = False

    return peft_model


def get_base_model_only(config: ExperimentConfig, gpu_id: int = 0):
    """Load 4-bit quantized Qwen model with no PEFT adapters.

    Used by FedALTClient before manually injecting MMOE LoRA layers.
    Mirrors get_model() but skips the LoraConfig / get_peft_model() step.
    """
    use_cuda = torch.cuda.is_available()
    compute_dtype = torch.bfloat16 if use_cuda else torch.float32
    model_cfg = config.model

    quantization_config = None
    if use_cuda:
        if model_cfg.quantization == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
        device_map={"": gpu_id} if use_cuda else None,
    )

    setattr(model, "model_parallel", True)
    setattr(model, "is_parallelizable", True)

    if use_cuda:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
        )
        # Needed for gradient flow through checkpointed layers when first trainable
        # param is not the embedding (our MMOE LoRA layers sit inside attention).
        model.enable_input_require_grads()

    # Same dtype adjustments as get_model()
    for name, module in model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight") and module.weight.dtype == torch.float32:
                module = module.to(compute_dtype)

    if model_cfg.gradient_checkpointing:
        model.config.use_cache = False

    return model


def load_model_for_inference(config: ExperimentConfig):
    """Load base model without quantization for inference/generation."""
    use_cuda = torch.cuda.is_available()
    compute_dtype = torch.bfloat16 if use_cuda else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if use_cuda else None,
    )
    device = "cuda" if use_cuda else "cpu"
    return model, device


# ---------------------------------------------------------------------------
# Dual LoRA management
# ---------------------------------------------------------------------------

def get_lora_config(config: ExperimentConfig) -> LoraConfig:
    """Create a LoRA config from experiment config (for adding adapters)."""
    lora_cfg = config.lora
    return LoraConfig(
        r=lora_cfg.rank,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        task_type="CAUSAL_LM",
        target_modules=lora_cfg.target_modules,
        bias=lora_cfg.bias,
    )


def add_local_adapter(model, config: ExperimentConfig) -> None:
    """Add a second LoRA adapter named 'local' to an existing PEFT model.

    Reads from ``config.lora_local`` so the personalized adapter can have a
    smaller capacity (rank / target modules) than the global adapter.
    """
    lora_cfg = config.lora_local

    target_modules = lora_cfg.target_modules
    if target_modules == "all-linear":
        target_modules = find_all_linear_names(
            model.get_base_model(), bits=config.model.quantization
        )
    elif isinstance(target_modules, str):
        target_modules = [target_modules]
    else:
        target_modules = list(target_modules)

    local_config = LoraConfig(
        r=lora_cfg.rank,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        bias=lora_cfg.bias,
    )
    model.add_adapter("local", local_config)


def get_adapter_params(model, adapter_name: str = "default") -> List[np.ndarray]:
    """Extract named adapter parameters as a list of numpy arrays."""
    state_dict = get_peft_model_state_dict(model, adapter_name=adapter_name)
    return [val.detach().to(torch.float32).cpu().numpy() for val in state_dict.values()]


def get_adapter_param_keys(model, adapter_name: str = "default") -> List[str]:
    """Get the keys for a named adapter's state dict."""
    state_dict = get_peft_model_state_dict(model, adapter_name=adapter_name)
    return list(state_dict.keys())


def set_adapter_params(model, params: List[np.ndarray], adapter_name: str = "default") -> None:
    """Set named adapter parameters from numpy arrays."""
    keys = get_adapter_param_keys(model, adapter_name)
    # Use bfloat16 on CUDA to match the model's compute dtype.
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    state_dict = OrderedDict({k: torch.tensor(v, dtype=dtype) for k, v in zip(keys, params)})
    set_peft_model_state_dict(model, state_dict, adapter_name=adapter_name)


def activate_adapters(model, names) -> None:
    """Activate one or more adapters for training/inference.

    names: str or List[str]
    """
    if isinstance(names, str):
        model.set_adapter(names)
    else:
        # PeftModel.set_adapter only accepts a single string in PEFT 0.17.x,
        # but the underlying LoraModel.set_adapter supports a list.
        model.base_model.set_adapter(names)


# ---------------------------------------------------------------------------
# Tokenizer and data collator
# ---------------------------------------------------------------------------

def get_tokenizer_and_data_collator(config: ExperimentConfig):
    """Initialize tokenizer and data collator."""
    model_cfg = config.model
    train_cfg = config.train

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.name,
        use_fast=model_cfg.use_fast_tokenizer,
        padding_side="right",
    )

    if getattr(tokenizer, "pad_token", None) is None or tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": DEFAULT_PAD_TOKEN})

    if "llama" in model_cfg.name.lower() or isinstance(tokenizer, LlamaTokenizer):
        tokenizer.add_special_tokens({
            "eos_token": tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id) if tokenizer.eos_token_id else "</s>",
            "bos_token": tokenizer.convert_ids_to_tokens(tokenizer.bos_token_id) if tokenizer.bos_token_id else "<s>",
            "unk_token": tokenizer.convert_ids_to_tokens(0) if hasattr(tokenizer, "convert_ids_to_tokens") else "<unk>",
        })

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=train_cfg.source_max_len,
        target_max_len=train_cfg.target_max_len,
        train_on_source=train_cfg.train_on_source,
        predict_with_generate=False,
    )

    return tokenizer, data_collator


@dataclass
class DataCollatorForCausalLM:
    """Data collator for causal language modeling."""
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool = False

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = []
        targets = []
        use_chat_template = hasattr(self.tokenizer, "apply_chat_template")

        for example in instances:
            messages = []
            if "messages" in example and isinstance(example["messages"], list):
                for msg in example["messages"]:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        messages.append({"role": msg["role"], "content": str(msg["content"])})
            else:
                if "instruction" in example:
                    user_input = str(example.get("input", "") or "").strip()
                    target = str(example.get("output", "") or "").strip()
                elif "input" in example and "output" in example:
                    user_input = str(example["input"])
                    target = str(example["output"])
                else:
                    vals = list(example.values())
                    user_input = str(vals[0]) if vals else ""
                    target = str(vals[1]) if len(vals) > 1 else ""
                messages = [{"role": "user", "content": user_input}]

            if use_chat_template:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                sources.append(prompt)
                if len(messages) == 1 and "target" in locals():
                    targets.append(f"{target}{self.tokenizer.eos_token}")
                else:
                    last_assistant = next(
                        (m["content"] for m in reversed(messages) if m["role"] == "assistant"), None
                    )
                    targets.append(
                        f"{last_assistant}{self.tokenizer.eos_token}" if last_assistant
                        else f"{self.tokenizer.eos_token}"
                    )
            else:
                sources.append(f"{self.tokenizer.bos_token}User: {user_input}\nAssistant:")
                targets.append(f"{target}{self.tokenizer.eos_token}")

        tokenized_sources = self.tokenizer(
            sources, max_length=self.source_max_len, truncation=True, add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets, max_length=self.target_max_len, truncation=True, add_special_tokens=False,
        )

        input_ids = []
        labels = []
        for tok_src, tok_tgt in zip(tokenized_sources["input_ids"], tokenized_targets["input_ids"]):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tok_src + tok_tgt))
                if not self.train_on_source:
                    labels.append(torch.tensor(
                        [IGNORE_INDEX] * len(tok_src) + copy.deepcopy(tok_tgt)
                    ))
                else:
                    labels.append(torch.tensor(copy.deepcopy(tok_src + tok_tgt)))
            else:
                input_ids.append(torch.tensor(tok_src))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None

        data_dict = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict["labels"] = labels
        return data_dict


# ---------------------------------------------------------------------------
# Cosine annealing LR schedule
# ---------------------------------------------------------------------------

def cosine_annealing(
    current_round: int,
    total_rounds: int,
    lr_max: float = 0.001,
    lr_min: float = 0.0,
) -> float:
    cos_inner = math.pi * current_round / total_rounds
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(cos_inner))
