import torch
import torch.nn as nn
import time
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Tuple, Union
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.modeling_llama import *
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from dataclasses import dataclass, fields, replace
from hf.load_config import Config

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


def _move_to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors in nested data structures to the specified device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, tuple):
        return tuple(_move_to_device(x, device) for x in obj)
    elif isinstance(obj, list):
        return [_move_to_device(x, device) for x in obj]
    elif isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    else:
        return obj

def move_to_cpu(instance: IntermediateOutput) -> IntermediateOutput:
    """Move all tensor attributes of the IntermediateOutput instance to CPU."""
    cpu_device = torch.device('cpu')
    new_kwargs = {}
    for field in fields(instance):
        value = getattr(instance, field.name)
        new_value = _move_to_device(value, cpu_device)
        new_kwargs[field.name] = new_value
    return replace(instance, **new_kwargs)

def move_to_cuda(instance: IntermediateOutput, device: Optional[Union[int, str, torch.device]] = None) -> IntermediateOutput:
    """Move all tensor attributes of the IntermediateOutput instance to CUDA."""
    if device is None:
        target_device = torch.device('cuda')
    elif isinstance(device, int):
        target_device = torch.device(f'cuda:{device}')
    elif isinstance(device, str):
        target_device = torch.device(device)
    elif isinstance(device, torch.device):
        target_device = device
    else:
        raise ValueError(f"Invalid device type: {type(device)}. Expected int, str, or torch.device.")

    new_kwargs = {}
    for field in fields(instance):
        value = getattr(instance, field.name)
        new_value = _move_to_device(value, target_device)
        new_kwargs[field.name] = new_value
    return replace(instance, **new_kwargs)

class DistributedModel(nn.Module):
    def __init__(self, config: PretrainedConfig, runtime_config: Config):
        super().__init__()
        self.node_rrefs = []
        for worker in runtime_config.workers:
            worker_name = worker.name
            offset = (int(worker.start), int(worker.end))
            ckpt_path = worker.ckpt_path
            rref = rpc.remote(worker_name, CustomLlamaModel, args=(config, offset, ckpt_path))
            self.node_rrefs.append(rref)

        print("All workers initiated")

    def forward(self, **input_data):
        with torch.no_grad():
            
            start_time = time.time()
            out_rrefs = [RRef(input_data)]
            for rref in self.node_rrefs:
                out_new_rref = rref.remote().forward(out_rrefs[-1])
                out_rrefs.append(out_new_rref)

            print(f"token gen time: {(time.time() - start_time) * 1000} ms ")

            result = out_rrefs[-1].to_here()
            for value in result.values():
                if isinstance(value, torch.Tensor):
                    value = value.cuda()
                    
            return result
    
class CustomLlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, offset: Tuple[int, int], ckpt_path: str):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.offset = offset
        if offset[0] == 0:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx).to("cuda")
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

        if self.offset[0] == 0:
            input_data: dict = input_data.to_here()
            # Tensor values (move to CUDA if present)
            input_ids = get_cuda_tensor(input_data, 'input_ids')
            attention_mask = get_cuda_tensor(input_data, 'attention_mask')
            position_ids = get_cuda_tensor(input_data, 'position_ids')
            past_key_values = get_cuda_tensor(input_data, 'past_key_values')
            inputs_embeds = get_cuda_tensor(input_data, 'inputs_embeds')
            cache_position = get_cuda_tensor(input_data, 'cache_position')
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

            if use_cache and past_key_values is None:
                past_key_values = DynamicCache()

            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )

            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)

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
            # Process tensor attributes (handle renames and CUDA)
            hidden_states = get_cuda_tensor(input_data, 'hidden_states')
            causal_mask = get_cuda_tensor(input_data, 'attention_mask')  # Map attention_mask -> causal_mask
            position_ids = get_cuda_tensor(input_data, 'position_ids')
            past_key_values = get_cuda_tensor(input_data, 'past_key_value')  # Note singular form in input
            cache_position = get_cuda_tensor(input_data, 'cache_position')
            position_embeddings = input_data['position_embeddings']
            pos_embed_cuda = ()
            for elements in position_embeddings:
                pos_embed_cuda += (elements.cuda(),)
            position_embeddings = pos_embed_cuda
            output_attentions = input_data['output_attentions']
            use_cache = input_data['use_cache']
            flash_attn_kwargs = input_data['flash_attn_kwargs']
            

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
            
        if self.offset[1] == self.config.num_hidden_layers:
            hidden_states = self.norm(hidden_states)

            # add hidden states from the last decoder layer
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            hidden_states = hidden_states.cpu()
            past_key_values = past_key_values.cpu() if use_cache else None
            all_hidden_states = all_hidden_states.cpu() if output_hidden_states else None
            all_self_attns = all_self_attns.cpu() if output_attentions else None
        
            # output = BaseModelOutputWithPast(
            #     last_hidden_state=hidden_states,
            #     past_key_values=past_key_values if use_cache else None,
            #     hidden_states=all_hidden_states,
            #     attentions=all_self_attns,
            # )
            output = {
                'last_hidden_state': hidden_states,
                'past_key_values': past_key_values if use_cache else None,
                'hidden_states': all_hidden_states,
                'attentions': all_self_attns,
            }
            # return output if return_dict else output.to_tuple()
            return output if return_dict else tuple(output.values())
        else:
            hidden_states = hidden_states.cpu()
            past_key_values = past_key_values.cpu() if use_cache else None
            causal_mask = causal_mask.cpu() if causal_mask is not None else None
            position_ids = position_ids.cpu() if position_ids is not None else None
            cache_position = cache_position.cpu() if cache_position is not None else None
            pos_embed_cpu = ()
            for elements in position_embeddings:
                pos_embed_cpu += (elements.cpu(),)
            position_embeddings = pos_embed_cpu

            # assign output as a dict
            # output = IntermediateOutput(
            #     hidden_states,
            #     attention_mask=causal_mask,
            #     position_ids=position_ids,
            #     past_key_value=past_key_values,
            #     output_attentions=output_attentions,
            #     use_cache=use_cache,
            #     cache_position=cache_position,
            #     position_embeddings=position_embeddings,
            #     flash_attn_kwargs=flash_attn_kwargs,
            # )
            output = {
                'hidden_states': hidden_states,
                'attention_mask': causal_mask,
                'position_ids': position_ids,
                'past_key_value': None,
                'output_attentions': output_attentions,
                'use_cache': False,
                'cache_position': cache_position,
                'position_embeddings': position_embeddings,
                'flash_attn_kwargs': flash_attn_kwargs,
            }
            
            return output


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

def get_cuda_tensor(data, key):
    """Helper to move tensor to CUDA if it exists and is not None."""
    val = data.get(key)
    return val.cuda() if val is not None else None
    

