from dataclasses import dataclass
from typing import Any, Dict, OrderedDict
from transformers.models.llama.modeling_llama import *
import torch
import torch.nn as nn
import time, gc
from pathlib import Path
from torch.distributed.rpc import RRef
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from dataclasses import dataclass
from core.load_config import Config
from core.distributed import DistributedModel
from utils.monitor import MemoryMonitor


@dataclass
class IntermediateOutput(ModelOutput):
    hidden_states: torch.FloatTensor  # Only required field
    attention_mask: Optional[torch.Tensor] = None
    position_ids: Optional[torch.LongTensor] = None
    past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # Align with HF Cache type
    output_attentions: Optional[bool] = None
    use_cache: Optional[bool] = None
    cache_position: Optional[torch.LongTensor] = None
    position_embeddings: Optional[torch.FloatTensor] = None
    flash_attn_kwargs: Optional[Dict[str, Any]] = None


class CustomLlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, runtime_config):
        super().__init__(config)
        self.runtime_config = runtime_config
        self.memory_monitor = MemoryMonitor()
        
        # Initialize your custom model instead of the original LlamaModel
        self.model = DistributedModel(config, runtime_config, CustomLlamaModel)  # Replace with your custom model
        
        # The following lines are copied from the original LlamaForCausalLM __init__
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).to("cuda")
        
        # Initialize weights and apply final processing
        self._initialize_weights()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def _initialize_weights(self):
        print("Initializing output layer weights")
        old_dict = torch.load(self.runtime_config.master.lm_head_weight_path)
        lm_head_state_dict = OrderedDict()
        lm_head_state_dict['weight'] = old_dict['lm_head.weight']
        self.lm_head.load_state_dict(lm_head_state_dict)
        del old_dict
        print("Output layer weights initialized")
        gc.collect()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        token_count = input_ids.shape[-1] if input_ids is not None else 0

        try:
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

            hidden_states = outputs.last_hidden_state
            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
                logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

            loss = None
            if labels is not None:
                loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **loss_kwargs)

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.last_hidden_state,
                attentions=outputs.attentions,
            )
        
        finally:
            if torch.cuda.is_available():
                self.memory_monitor.record_memory(token_count)
    
    def reset_cache(self):
        """Reset cache across all pipeline stages"""
        if hasattr(self.model, 'reset_kv_cache'):
            self.model.reset_kv_cache()
        elif hasattr(self.model, 'node_rrefs'):
            # Distributed model case
            futures = []
            for rref in self.model.node_rrefs:
                futures.append(rref.rpc_async().reset_kv_cache())
            torch.futures.wait_all(futures)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    

class CustomLlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, offset: Tuple[int, int], ckpt_path: str):
        super().__init__(config)

        self.max_cache_size = config.max_position_embeddings  # Or set appropriate limit
        self.current_cache_size = 0

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.offset = offset
        if offset[0] == 0:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx).to("cuda")
            
        self._kv_cache: Optional[Cache] = None
        self.rotary_emb = LlamaRotaryEmbedding(config=config).to("cuda")
        
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx).to("cuda") for layer_idx in range(offset[0], offset[1])]
        )
        
        if offset[1] == config.num_hidden_layers:
            self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to("cuda")

        self.gradient_checkpointing = False
        self.ckpt_path = ckpt_path

        # Initialize weights and apply final processing
        self._initialize_weights()

    def _initialize_weights(self):
        prev_time = time.time()
        print(f'Loading checkpoint "{self.ckpt_path}"')
        checkpoint = torch.load(self.ckpt_path, map_location="cpu")
        print(f"Loaded checkpoint in {time.time() - prev_time:.6f}s")
        prev_time = time.time()
        new_state_dict = OrderedDict()
        for k,v in checkpoint.items():
            if 'layers' in k:
                names_list = k.split('.')[1:]
                if self.offset[0] <= int(names_list[1]) < self.offset[1]:
                    names_list[1] = str(int(names_list[1]) - self.offset[0])
                name = '.'.join(names_list)
                new_state_dict[name] = v
            else:
                names_list = k.split('.')[1:]
                name = '.'.join(names_list)
                new_state_dict[name] = v

        self.load_state_dict(new_state_dict, strict=True)
        print(f"Loaded state dict in {time.time() - prev_time:.2f}s")

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_data: RRef,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs,
    ) -> Union[Tuple, Dict[str, Any]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        if self.offset[0] == 0:
            input_data: dict = input_data.to_here()
            input_ids = input_data['input_ids']
            attention_mask = input_data['attention_mask']
            position_ids = input_data['position_ids']
            past_key_values = input_data['past_key_values']
            inputs_embeds = input_data['inputs_embeds']
            cache_position = input_data['cache_position']
            use_cache = input_data['use_cache']
            output_attentions = input_data['output_attentions']
            output_hidden_states = input_data['output_hidden_states']
            return_dict = input_data['return_dict']

            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

            if self.gradient_checkpointing and self.training and use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
                )
                use_cache = False

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)

            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )

            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)

            past_len = self._kv_cache.get_seq_length() if self._kv_cache else 0
            if (attention_mask is not None
                and past_len > 0
                and attention_mask.shape[-1] != past_len + input_ids.shape[1]):
                pad = attention_mask.new_ones(attention_mask.size(0), past_len)
                attention_mask = torch.cat([pad, attention_mask], dim=-1)

            if use_cache:
                if self._kv_cache is None:
                    self._kv_cache = DynamicCache()

                if self._kv_cache.get_seq_length() > self.max_cache_size:
                    # Prune oldest half of cache
                    keep_from = self._kv_cache.get_seq_length() // 2
                    self._kv_cache = self._kv_cache.prune(keep_from)
                    self.current_cache_size = self._kv_cache.get_seq_length()

            past_key_values = self._kv_cache

            causal_mask = self._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
            )

            hidden_states = inputs_embeds

            # create position embeddings to be shared across the decoder layers
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

            # decoder layers
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None
        else:
            input_data: dict = input_data.to_here()
            hidden_states = input_data['hidden_states']
            attention_mask = input_data['attention_mask']
            position_ids = input_data['position_ids']
            cache_position = input_data['cache_position']
            
            past_len = self._kv_cache.get_seq_length() if self._kv_cache else 0
            if (attention_mask is not None
                and past_len > 0
                and attention_mask.shape[-1] != past_len + input_ids.shape[1]):
                pad = attention_mask.new_ones(attention_mask.size(0), past_len)
                attention_mask = torch.cat([pad, attention_mask], dim=-1)

            if use_cache:
                if self._kv_cache is None:
                    self._kv_cache = DynamicCache()

                if self._kv_cache.get_seq_length() > self.max_cache_size:
                    # Prune oldest half of cache
                    keep_from = self._kv_cache.get_seq_length() // 2
                    self._kv_cache = self._kv_cache.prune(keep_from)
                    self.current_cache_size = self._kv_cache.get_seq_length()

            past_key_values = self._kv_cache

            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            output_attentions = input_data['output_attentions']
            use_cache = input_data['use_cache']
            flash_attn_kwargs = input_data['flash_attn_kwargs']

            causal_mask = self._update_causal_mask(
                attention_mask, hidden_states, cache_position, past_key_values, output_attentions
            )

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

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

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if use_cache:
                self._kv_cache = past_key_values

        # print(f"kv cache length: {self._kv_cache.get_seq_length()}")
        # print(f"key cache length: {len(self._kv_cache.key_cache)}")
        # print(f"kv cache element size: {self._kv_cache.key_cache[0].shape}")
            
        if self.offset[1] == self.config.num_hidden_layers:
            hidden_states = self.norm(hidden_states)

            # add hidden states from the last decoder layer
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
        
            output = BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values if use_cache else None,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )
            return output if return_dict else tuple(output.values())
        else:

            # assign output as a dict
            output = IntermediateOutput(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=None,
                flash_attn_kwargs=flash_attn_kwargs,
            )
            
            return output
        
    def reset_kv_cache(self):
        """More aggressive cache cleanup"""
        if self._kv_cache is not None:
            # Explicitly delete cache tensors
            if hasattr(self._kv_cache, 'key_cache'):
                for k in self._kv_cache.key_cache:
                    if isinstance(k, torch.Tensor) and k.is_cuda:
                        del k
            if hasattr(self._kv_cache, 'value_cache'):
                for v in self._kv_cache.value_cache:
                    if isinstance(v, torch.Tensor) and v.is_cuda:
                        del v
        self._kv_cache = None
        self.current_cache_size = 0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask
    

