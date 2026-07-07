import os
import torch
from transformers.models.mistral.modeling_mistral import MistralModel
from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import Optional, Union, List, Tuple

try:
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask
except ImportError:
    DynamicCache = None
    create_causal_mask = None

class SplittedMistralModel(MistralModel):

    def set_devices(self):
        forced_device = os.environ.get("GUIDEDQUANT_CUDA_DEVICE")
        if forced_device is not None:
            self.split_gpus = False
            self.primary_device = f"cuda:{forced_device}"
            print(f"using {self.primary_device}")
            self.to(self.primary_device)
            return

        num_visible_devices = torch.cuda.device_count()
        assert num_visible_devices > 0, "Must use at least one GPU"
        self.split_gpus = num_visible_devices > 1
        self.primary_device = "cuda:0"
        print(f"splitting into {num_visible_devices} GPUs")
        if not self.split_gpus:
            self.to(self.primary_device)
        else:
            # For larger model, we need to split the model into multiple GPUs
            # assign the embedding and norm onto the 1st devide
            self.embed_tokens.to(self.primary_device)
            self.rotary_emb.to(self.primary_device)
            self.norm.to(self.primary_device)
            # layers are divided into #(num GPUs) chunks
            self.split_indices = []
            prev_device = 0
            nums = len(self.layers) // num_visible_devices
            for i, layer in enumerate(self.layers):
                device = min(num_visible_devices - 1, i // nums)
                if prev_device != device:
                    self.split_indices.append(i)
                print(f"Moving layer {i} to cuda:{device}")
                layer.to(f"cuda:{device}")
                prev_device = device


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if hasattr(self, "_update_causal_mask"):
            causal_mask = self._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
            )
            use_new_mask_api = False
        else:
            if create_causal_mask is None:
                raise RuntimeError("This transformers version requires create_causal_mask, but it is not available.")
            if use_cache and past_key_values is None:
                if DynamicCache is None:
                    raise RuntimeError("This transformers version requires DynamicCache, but it is not available.")
                past_key_values = DynamicCache(config=self.config)
            causal_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            use_new_mask_api = True
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        device = 0
        for idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Move activations to the next device at the split points
            if self.split_gpus and idx in self.split_indices:
                device += 1
                hidden_states = hidden_states.to(f"cuda:{device}")
                new_position_embeddings = []
                for position_embedding in position_embeddings:
                    new_position_embeddings.append(position_embedding.to(f"cuda:{device}"))
                position_embeddings = tuple(new_position_embeddings)
                if causal_mask is not None:
                    causal_mask = causal_mask.to(f"cuda:{device}")
                position_ids = position_ids.to(f"cuda:{device}")
                cache_position = cache_position.to(f"cuda:{device}")

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            elif use_new_mask_api:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

            # Move activations back to the 1st device at the end
            if self.split_gpus and idx == len(self.layers) - 1:
                hidden_states = hidden_states.to(self.primary_device)
                new_position_embeddings = []
                for position_embedding in position_embeddings:
                    new_position_embeddings.append(position_embedding.to(self.primary_device))
                position_embeddings = tuple(new_position_embeddings)
                if causal_mask is not None:
                    causal_mask = causal_mask.to(self.primary_device)
                position_ids = position_ids.to(self.primary_device)
                cache_position = cache_position.to(self.primary_device)

            if use_cache and isinstance(layer_outputs, tuple):
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions and isinstance(layer_outputs, tuple):
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache and use_new_mask_api else next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
