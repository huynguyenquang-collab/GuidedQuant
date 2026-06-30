import torch
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoTokenizer, PreTrainedTokenizerBase
from .splitted_models import SplittedLlamaModel, SplittedQwen3Model, SplittedGemma3TextModel


def load_model(model_str_or_model, dtype=torch.float16):
    """Returns a model from a string or a model object. If a string is passed, it will be loaded from the HuggingFace"""
    if isinstance(model_str_or_model, str):
        # Qwen / Gemma models are more stable in bfloat16
        if not "llama" in model_str_or_model.lower():
            dtype = torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(
            model_str_or_model,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map='cpu',
        )
    else:
        assert isinstance(model_str_or_model, PreTrainedModel), "model must be a string or a PreTrainedModel"
        model = model_str_or_model
    return model


def dispatch_model(model):
    architecture = model.config.architectures[0]
    if architecture == 'LlamaForCausalLM':
        model.model.__class__ = SplittedLlamaModel
        model.model.config.use_cache = False
        model.model.set_devices()
        model.lm_head.to("cuda:0")
        return model
    elif architecture == 'Qwen3ForCausalLM':
        model.model.__class__ = SplittedQwen3Model
        model.model.config.use_cache = False
        model.model.set_devices()
        model.lm_head.to("cuda:0")
        return model
    elif architecture == 'Gemma3ForConditionalGeneration':
        model.to("cuda:0")
        model.model.language_model.__class__ = SplittedGemma3TextModel
        model.model.language_model.config.use_cache = False
        model.model.language_model.set_devices()
        model.lm_head.to("cuda:0")
        return model
    elif architecture in {'Qwen3_5ForConditionalGeneration', 'Qwen3_5ForCausalLM'}:
        raise NotImplementedError(
            f"Model {architecture} is not supported by this GuidedQuant SqueezeLLM/LNQ path. "
            "Qwen3.5 uses hybrid linear-attention/full-attention layers with different module names per layer, "
            "while this quantizer expects the same quantized module list in every decoder layer."
        )
    else:
        raise NotImplementedError(f"Model {architecture} is not supported")


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
