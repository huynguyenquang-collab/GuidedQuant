def get_splitted_llama_model():
    from .llama import SplittedLlamaModel
    return SplittedLlamaModel


def get_splitted_mistral_model():
    from .mistral import SplittedMistralModel
    return SplittedMistralModel


def get_splitted_qwen3_model():
    from .qwen3 import SplittedQwen3Model
    return SplittedQwen3Model


def get_splitted_gemma3_text_model():
    from .gemma3 import SplittedGemma3TextModel
    return SplittedGemma3TextModel
