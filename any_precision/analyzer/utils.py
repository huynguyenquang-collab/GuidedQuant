import os
import torch
import logging
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoTokenizer, PreTrainedTokenizerBase
from .splitted_models import (
    get_splitted_gemma3_text_model,
    get_splitted_llama_model,
    get_splitted_mistral_model,
    get_splitted_qwen3_model,
)


def get_target_cuda_device():
    return f"cuda:{os.environ.get('GUIDEDQUANT_CUDA_DEVICE', '0')}"


def load_model(model_str_or_model, dtype=torch.float16):
    """Returns a model from a string or a model object. If a string is passed, it will be loaded from the HuggingFace"""
    if isinstance(model_str_or_model, str):
        # Qwen / Gemma models are more stable in bfloat16; LLaMA/Mistral paths
        # use fp16 to match the original GuidedQuant kernels and cache layout.
        model_name = model_str_or_model.lower()
        if not any(name in model_name for name in ("llama", "mistral")):
            dtype = torch.bfloat16

        kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }
        attn_impl = os.environ.get("GUIDEDQUANT_ATTN_IMPLEMENTATION", "").strip()
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl
        try:
            model = AutoModelForCausalLM.from_pretrained(model_str_or_model, **kwargs)
        except Exception as exc:
            if not attn_impl:
                raise
            logging.warning(
                "Falling back from attn_implementation=%s because model loading failed: %s",
                attn_impl,
                exc,
            )
            kwargs.pop("attn_implementation", None)
            fallback_impl = os.environ.get("GUIDEDQUANT_ATTN_FALLBACK", "sdpa").strip()
            if fallback_impl:
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_str_or_model,
                        attn_implementation=fallback_impl,
                        **kwargs,
                    )
                except Exception as fallback_exc:
                    logging.warning(
                        "Falling back from attn_implementation=%s because model loading failed: %s",
                        fallback_impl,
                        fallback_exc,
                    )
                    model = AutoModelForCausalLM.from_pretrained(model_str_or_model, **kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_str_or_model, **kwargs)
    else:
        assert isinstance(model_str_or_model, PreTrainedModel), "model must be a string or a PreTrainedModel"
        model = model_str_or_model
    return model


def dispatch_model(model):
    target_device = get_target_cuda_device()
    if model.config.architectures[0] == 'LlamaForCausalLM':
        SplittedLlamaModel = get_splitted_llama_model()
        model.model.__class__ = SplittedLlamaModel
        model.model.config.use_cache = False
        model.model.set_devices()
        model.lm_head.to(target_device)
        return model
    elif model.config.architectures[0] == 'MistralForCausalLM':
        SplittedMistralModel = get_splitted_mistral_model()
        model.model.__class__ = SplittedMistralModel
        model.model.config.use_cache = False
        model.model.set_devices()
        model.lm_head.to(target_device)
        return model
    elif model.config.architectures[0] == 'Qwen3ForCausalLM':
        SplittedQwen3Model = get_splitted_qwen3_model()
        model.model.__class__ = SplittedQwen3Model
        model.model.config.use_cache = False
        model.model.set_devices()
        model.lm_head.to(target_device)
        return model
    elif model.config.architectures[0] == 'Gemma3ForConditionalGeneration':
        SplittedGemma3TextModel = get_splitted_gemma3_text_model()
        model.to(target_device)
        model.model.language_model.__class__ = SplittedGemma3TextModel
        model.model.language_model.config.use_cache = False
        model.model.language_model.set_devices()
        model.lm_head.to(target_device)
        return model
    else:
        raise NotImplementedError(f"Model {model.config.architectures[0]} is not supported")


def load_tokenizer(model_str_or_model_or_tokenizer):
    """Returns a tokenizer from the model string or model object or tokenizer object"""
    if isinstance(model_str_or_model_or_tokenizer, str):
        model_str = model_str_or_model_or_tokenizer
        return AutoTokenizer.from_pretrained(model_str, trust_remote_code=True)
    elif isinstance(model_str_or_model_or_tokenizer, PreTrainedModel):
        model_str = model_str_or_model_or_tokenizer.name_or_path
        return AutoTokenizer.from_pretrained(model_str, trust_remote_code=True)
    else:
        assert isinstance(model_str_or_model_or_tokenizer, PreTrainedTokenizerBase), \
            f"Unsupported type for model_str_or_model_or_tokenizer: {type(model_str_or_model_or_tokenizer)}"
        return model_str_or_model_or_tokenizer
