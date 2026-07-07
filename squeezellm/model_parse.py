from transformers import AutoModelForCausalLM


def load_model(model, model_type, cache_dir=None):
    return AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype="auto",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )


def parse_model(model):
    if "opt" in str(type(model)).lower():
        model_type = "opt"
    elif "qwen" in str(type(model)).lower():
        model_type = "qwen"
    elif "mistral" in str(type(model)).lower():
        model_type = "mistral"
    else:
        # additional rules should be added to support other models
        model_type = "llama"
    print(f"Model type : {model_type}")

    return model_type


def get_module_names(model_type):
    if model_type == "opt":
        return ["q", "k", "v", "o", "up", "down"]
    else:
        assert model_type in ("llama", "mistral", "qwen")
        return ["q", "k", "v", "o", "gate", "up", "down"]


def get_named_modules(layer, model_type):
    if model_type == "opt":
        return [
            ("q", layer.self_attn.q_proj),
            ("k", layer.self_attn.k_proj),
            ("v", layer.self_attn.v_proj),
            ("o", layer.self_attn.out_proj),
            ("up", layer.fc1),
            ("down", layer.fc2),
        ]

    assert model_type in ("llama", "mistral", "qwen")
    modules = []
    if hasattr(layer, "self_attn"):
        modules.extend(
            [
                ("q", layer.self_attn.q_proj),
                ("k", layer.self_attn.k_proj),
                ("v", layer.self_attn.v_proj),
                ("o", layer.self_attn.o_proj),
            ]
        )
    elif hasattr(layer, "linear_attn"):
        modules.extend(
            [
                ("la_qkv", layer.linear_attn.in_proj_qkv),
                ("la_z", layer.linear_attn.in_proj_z),
                ("la_b", layer.linear_attn.in_proj_b),
                ("la_a", layer.linear_attn.in_proj_a),
                ("la_o", layer.linear_attn.out_proj),
            ]
        )
    else:
        raise AttributeError(
            f"Unsupported {type(layer).__name__}: expected self_attn or linear_attn. "
            f"Children: {list(layer._modules.keys())}"
        )

    modules.extend(
        [
            ("gate", layer.mlp.gate_proj),
            ("up", layer.mlp.up_proj),
            ("down", layer.mlp.down_proj),
        ]
    )
    return modules


def get_named_sequential(layer, model_type):
    if model_type == "opt":
        return [
            ("q", "self_attn.q_proj"),
            ("k", "self_attn.k_proj"),
            ("v", "self_attn.v_proj"),
            ("o", "self_attn.out_proj"),
            ("up", "fc1"),
            ("down", "fc2"),
        ]

    assert model_type in ("llama", "mistral", "qwen")
    paths = []
    if hasattr(layer, "self_attn"):
        paths.extend(
            [
                ("q", "self_attn.q_proj"),
                ("k", "self_attn.k_proj"),
                ("v", "self_attn.v_proj"),
                ("o", "self_attn.o_proj"),
            ]
        )
    elif hasattr(layer, "linear_attn"):
        paths.extend(
            [
                ("la_qkv", "linear_attn.in_proj_qkv"),
                ("la_z", "linear_attn.in_proj_z"),
                ("la_b", "linear_attn.in_proj_b"),
                ("la_a", "linear_attn.in_proj_a"),
                ("la_o", "linear_attn.out_proj"),
            ]
        )
    else:
        raise AttributeError(
            f"Unsupported {type(layer).__name__}: expected self_attn or linear_attn. "
            f"Children: {list(layer._modules.keys())}"
        )

    paths.extend(
        [
            ("gate", "mlp.gate_proj"),
            ("up", "mlp.up_proj"),
            ("down", "mlp.down_proj"),
        ]
    )
    return paths


def get_modules(layer, model_type):
    return [module for _name, module in get_named_modules(layer, model_type)]


def get_sequential(model_type):
    if model_type == "opt":
        return [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
            "fc1",
            "fc2",
        ]
    else:
        assert model_type in ("llama", "mistral", "qwen")
        return [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]


def get_model(model, model_type):
    if model_type == "opt":
        return model.model.decoder
    else:
        assert model_type in ("llama", "mistral", "qwen")
        return model.model


def get_layers(model, model_type):
    _model = get_model(model, model_type)
    if model_type == "opt":
        return _model.layers
    else:
        assert model_type in ("llama", "mistral", "qwen")
        return _model.layers


def get_layers_name(model_type):
    if model_type == "opt":
        return "model.decoder.layers"
    else:
        assert model_type in ("llama", "mistral", "qwen")
        return "model.layers"


def get_embedding(model, model_type):
    _model = get_model(model, model_type)
    if model_type == "opt":
        return [_model.embed_tokens, _model.embed_positions]
    else:
        assert model_type in ("llama", "mistral", "qwen")
        return [_model.embed_tokens]


def get_norm(model, model_type):
    _model = get_model(model, model_type)
    if model_type == "opt":
        return _model.final_layer_norm
    else:
        assert model_type in ("llama", "mistral", "qwen")
        return _model.norm
