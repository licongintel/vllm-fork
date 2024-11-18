import collections
import itertools
import math
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.distributed
import torch.nn as nn

from vllm import envs
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalKwargs, PlaceholderRange
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, cdiv, is_fake_hpu,
                        is_pin_memory_available, make_tensor_with_pad)
from vllm.v1.attention.backends.hpu_attn import HPUAttentionBackendV1, HPUAttentionMetadata
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm_hpu_extension.profiler import (HabanaHighLevelProfiler,
                                         HabanaMemoryProfiler, format_bytes)
from vllm_hpu_extension.ops import batch2block, block2batch
import habana_frameworks.torch as htorch
if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput
from vllm.v1.engine.detokenizer import Detokenizer

logger = init_logger(__name__)

_TYPE_CACHE = {}
# These values are assumed to be zero in several places.
# Use caution when updating them!
_PAD_SLOT_ID = 0
_PAD_BLOCK_ID = 0


@dataclass
class PrefillInputData:

    request_ids: List
    prompt_lens: List
    token_ids: List
    position_ids: List
    attn_metadata: List
    logits_indices: List

    def zipped(self):
        return zip(self.request_ids, self.prompt_lens, self.token_ids,
                   self.position_ids, self.attn_metadata, self.logits_indices)


@dataclass
class DecodeInputData:

    num_decodes: int
    token_ids: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    attn_metadata: HPUAttentionMetadata = None
    logits_indices: Optional[torch.Tensor] = None


def flatten(in_list):
    return list(itertools.chain(*in_list))


def gather_list(input, indices, v):
    return [input[i] if i is not None else v for i in indices]


def _get_padded_batch_size(batch_size: int) -> int:
    # The GMM Pallas kernel requires num_tokens * topk to be a multiple of 16.
    # To meet this requirement in the simplest way, we set the minimal batch
    # size to 8.
    if batch_size <= 8:
        return 8
    else:
        return ((batch_size + 15) // 16) * 16


def _get_padded_prefill_len(x: int) -> int:
    # NOTE(woosuk): The pallas FlashAttention kernel requires the sequence
    # length to be a multiple of 16. We pad the prompt length to the nearest
    # multiple of 16. This is also good for performance.
    if x <= 16:
        return 16
    return 1 << (x - 1).bit_length()


class HpuModelAdapter:

    def __init__(self, model, block_size, dtype, enforce_eager):
        self.model = model
        self.prefill_use_fusedsdpa = os.getenv('VLLM_PROMPT_USE_FUSEDSDPA',
                                               '1').lower() in ['1', 'true'] \
                                                and not is_fake_hpu()
        self.block_size = block_size
        self.dtype = dtype
        if not is_fake_hpu() and not htorch.utils.internal.is_lazy(
        ) and not enforce_eager:
            self.model = torch.compile(self.model,
                                       backend='hpu_backend',
                                       dynamic=False)

    def _set_attn_bias(self, attn_metadata, batch_size, seq_len, device,
                       dtype):
        if (attn_metadata is None or self.prefill_use_fusedsdpa
                or not attn_metadata.is_prompt):
            return attn_metadata

        prefill_metadata = attn_metadata

        seq_lens_t = prefill_metadata.seq_lens_tensor
        context_lens_t = prefill_metadata.context_lens_tensor
        query_lens_t = seq_lens_t - context_lens_t

        block_list = attn_metadata.block_list
        max_context_len = (block_list.size(-1) //
                           batch_size if block_list is not None else 0)
        max_context_len = max_context_len * self.block_size
        past_mask = torch.arange(0,
                                 max_context_len,
                                 dtype=torch.int32,
                                 device=device)
        past_mask = (past_mask.view(1, -1).expand(batch_size, -1).ge(
            context_lens_t.view(-1, 1)).view(batch_size, 1, -1).expand(
                batch_size, seq_len, -1).view(batch_size, 1, seq_len, -1))

        len_mask = (torch.arange(0, seq_len, device=device,
                                 dtype=torch.int32).view(1, seq_len).ge(
                                     query_lens_t.unsqueeze(-1)).view(
                                         batch_size, 1, 1, seq_len))
        causal_mask = torch.triu(torch.ones((batch_size, 1, seq_len, seq_len),
                                            device=device,
                                            dtype=torch.bool),
                                 diagonal=1)
        mask = causal_mask.logical_or(len_mask)
        mask = torch.concat((past_mask, mask), dim=-1)
        attn_bias = (torch.zeros_like(mask, dtype=dtype).masked_fill_(
            mask, -math.inf))
        attn_metadata = prefill_metadata._replace(attn_bias=attn_bias)
        return attn_metadata

    def _set_block_mapping(self, metadata, batch_size, device, dtype):
        mask = torch.arange(0,
                            self.block_size,
                            device=device,
                            dtype=torch.int32).unsqueeze(0)
        mask = mask >= metadata.block_usage.unsqueeze(-1)
        attn_bias = (torch.zeros_like(mask, dtype=dtype).masked_fill_(
            mask, -math.inf))

        if not is_fake_hpu() and htorch.utils.internal.is_lazy():
            block_mapping = torch.nn.functional.one_hot(metadata.block_groups,
                                                        num_classes=batch_size)
        else:
            # Unfortunately one_hot on CPU/torch.compile mode/eager mode
            # doesn't handle out of bounds classes so we need to convert
            # all negative values to 0 (block_mapping) or bs (block_groups)
            block_groups = metadata.block_groups.to(torch.long)
            block_mapping = torch.nn.functional.relu(block_groups)
            block_mapping = torch.nn.functional.one_hot(block_mapping,
                                                        num_classes=batch_size)
            oob_values = block_groups.lt(0)
            block_mapping.masked_fill_(oob_values.unsqueeze(-1), 0)
            block_groups.masked_fill_(oob_values, batch_size)
            metadata = metadata._replace(block_groups=block_groups)
        block_mapping = block_mapping.to(dtype)
        metadata = metadata._replace(block_mapping=block_mapping,
                                     attn_bias=attn_bias)
        return metadata

    def _set_block_scales(self, metadata, device):
        block_mapping = metadata.block_mapping
        ones = torch.ones((block_mapping.size(0), ),
                          device=device,
                          dtype=block_mapping.dtype)
        sums = batch2block(block2batch(ones, block_mapping), block_mapping)
        block_scales = torch.reciprocal(torch.maximum(ones, sums))
        metadata = metadata._replace(block_scales=block_scales)
        return metadata

    def _set_indices_and_offsets(self, metadata, block_size, is_prompt):
        slot_mapping = metadata.slot_mapping.flatten()
        indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
        if is_prompt:
            indices = indices.unflatten(0, (-1, block_size))[:, 0]
            offsets = None
        else:
            offsets = torch.fmod(slot_mapping, block_size)
        metadata = metadata._replace(block_offsets=offsets,
                                     block_indices=indices)
        return metadata

    def _update_metadata(self, attn_metadata, batch_size, seq_len, device,
                         dtype):
        if attn_metadata.is_prompt:
            attn_metadata = self._set_attn_bias(attn_metadata, batch_size,
                                                seq_len, device, dtype)
        else:
            attn_metadata = self._set_block_mapping(attn_metadata, batch_size,
                                                    device, dtype)
            attn_metadata = self._set_block_scales(attn_metadata, device)
        attn_metadata = self._set_indices_and_offsets(attn_metadata,
                                                      self.block_size,
                                                      attn_metadata.is_prompt)
        return attn_metadata

    def forward(self, *args, **kwargs):
        kwargs = kwargs.copy()
        input_ids = kwargs['input_ids']
        kwargs['attn_metadata'] = self._update_metadata(
            kwargs['attn_metadata'], input_ids.size(0), input_ids.size(1),
            input_ids.device, self.dtype)
        hidden_states = self.model(*args, **kwargs)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        return hidden_states

    def compute_logits(self, *args, **kwargs):
        return self.model.compute_logits(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self.model.sample(*args, **kwargs)

    def generate_proposals(self, *args, **kwargs):
        return self.model.generate_proposals(*args, **kwargs)

    # sampler property will be used by spec_decode_worker
    # don't rename
    @property
    def sampler(self):
        return self.model.sampler


def _maybe_wrap_in_hpu_graph(*args, **kwargs):
    return htorch.hpu.wrap_in_hpu_graph(
        HpuModelAdapter(*args, **kwargs), disable_tensor_cache=True
    ) if False and htorch.utils.internal.is_lazy() else HpuModelAdapter(
        *args, **kwargs)


def subtuple(obj: object,
             typename: str,
             to_copy: List[str],
             to_override: Optional[Dict[str, object]] = None):
    if obj is None:
        return None
    if to_override is None:
        to_override = {}
    fields = set(to_copy) | set(to_override.keys())
    if type(obj) is dict:
        values = {key: obj[key] for key in fields if key in obj}
    else:
        values = {f: to_override.get(f, getattr(obj, f)) for f in fields}
    if typename not in _TYPE_CACHE:
        _TYPE_CACHE[typename] = collections.namedtuple(typename,
                                                       ' '.join(fields))
    return _TYPE_CACHE[typename](**values)


def trim_attn_metadata(metadata: HPUAttentionMetadata) -> object:
    # NOTE(kzawora): To anyone working on this in the future:
    # Trimming metadata is required when using HPUGraphs.
    # Attention metadata is going to be hashed by PT bridge, and
    # appropriate HPUGraphs will be matched based on all inputs' hash.

    # Before you put more keys in here, make sure you know their
    # value type and make sure you know how it's going to be hashed.
    # You can find that information in input_hash function
    # in habana_frameworks/torch/hpu/graphs.py. You can also hash
    # it manually with torch.hpu.graphs.input_hash(attention_metadata)

    # If you use primitive types here - they will get hashed based
    # on their value. You *will* get lots of excessive graph captures
    # (and an OOM eventually) if you decide to put something like
    # seq_len int here.
    # If you absolutely need a scalar, put it in a tensor. Tensors
    # get hashed using their metadata, not their values:
    # input_hash(torch.tensor(123)) == input_hash(torch.tensor(321))
    # input_hash(123) != input_hash(321)
    # input_hash("abc") != input_hash("cba")
    attention_metadata = subtuple(metadata, 'TrimmedAttentionMetadata', [
        'attn_bias', 'seq_lens_tensor', 'context_lens_tensor', 'block_list',
        'block_mapping', 'block_usage', 'slot_mapping', 'is_prompt',
        'block_indices', 'block_offsets', 'block_scales', 'block_groups'
    ])
    return attention_metadata


def next_pow2(value: int, base: int):
    res = base
    while value > 1:
        value = (value + 1) // 2
        res *= 2
    return res


def round_up(value: int, k: int):
    return (value + k - 1) // k * k


def pad_list(list, k, v):
    target_len = round_up(len(list), k)
    padding = target_len - len(list)
    return list + [v] * padding


def precompute_indices_and_offsets(block_size, slot_mapping, is_prompt):
    slot_mapping = slot_mapping.flatten()
    indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
    if is_prompt:
        indices = indices.unflatten(0, (-1, block_size))[:, 0]
        offsets = None
    else:
        offsets = torch.fmod(slot_mapping, block_size)
    return indices, offsets


class HPUModelRunner:

    def __init__(
        self,
        vllm_config: VllmConfig,
    ):
        #TODO(kzawora): remove this, this is ugly and only used for diagnostics
        self._ENGINE_ITER = 0
        # TODO: use ModelRunnerBase.__init__(self, vllm_config=vllm_config)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config
        #TODO(kzawora): remove this, this is for debug purposes only
        self._tokenizer = Detokenizer(
            vllm_config.model_config.tokenizer).tokenizer
        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
        if cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                cache_config.cache_dtype]

        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_model_len = model_config.max_model_len
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        self.max_num_tokens = scheduler_config.max_num_batched_tokens

        # Model-related.
        self.num_attn_layers = model_config.get_num_attention_layers(
            parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.head_size = model_config.get_head_size()

        # Lazy initialization
        # self.model: nn.Module  # Set after load_model
        self.kv_caches: List[torch.Tensor] = []

        # Request states.
        self.requests: Dict[str, CachedRequestState] = {}
        # Persistent batch.
        self.input_batch = InputBatch(
            max_num_reqs=self.scheduler_config.max_num_seqs,
            max_model_len=self.max_model_len,
            max_num_blocks_per_req=self.max_num_blocks_per_req,
            device=self.device,
            pin_memory=self.pin_memory,
        )

        self.use_hpu_graph = not self.model_config.enforce_eager
        # TODO(woosuk): Provide an option to tune the max cudagraph batch size.
        self.cudagraph_batch_sizes = [1, 2, 4] + [i for i in range(8, 513, 8)]
        self.max_batch_size = 256  # TODO(kzawora): fix this garbage
        self.input_ids = torch.zeros(
            (self.max_batch_size, self.max_num_tokens),
            dtype=torch.int32,
            device=self.device)
        self.positions = torch.zeros(
            (self.max_batch_size, self.max_num_tokens),
            dtype=torch.int64,
            device=self.device)
        self.prefill_positions = torch.tensor(
            range(self.max_model_len),
            device="cpu",
        ).to(torch.int32).reshape(1, -1)

        self.use_contiguous_pa = os.environ.get('VLLM_CONTIGUOUS_PA',
                                                'true').lower() == 'true'

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        # Remove stopped requests from the cached states.
        # Keep the states of the pre-empted requests.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)

        # Remove the requests from the persistent batch.
        stopped_req_ids = set().union(
            scheduler_output.preempted_req_ids,
            scheduler_output.finished_req_ids,
        )
        removed_req_indices: List[int] = []
        for req_id in stopped_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Update the states of the running requests.
        for req_data in scheduler_output.scheduled_running_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]
            req_index = self.input_batch.req_id_to_index[req_id]

            # Update the num_computed_tokens.
            req_state.num_computed_tokens = req_data.num_computed_tokens
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                req_data.num_computed_tokens)

            # Update the block table.
            num_new_blocks = len(req_data.new_block_ids)
            if num_new_blocks == 0:
                continue
            start_index = len(req_state.block_ids)
            end_index = start_index + num_new_blocks
            req_state.block_ids.extend(req_data.new_block_ids)
            self.input_batch.block_table_cpu[
                req_index, start_index:end_index] = req_data.new_block_ids

        req_ids_to_add: List[str] = []
        # Add new requests to the cached states.
        for req_data in scheduler_output.scheduled_new_reqs:
            req_id = req_data.req_id
            sampling_params = req_data.sampling_params
            if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=req_data.prompt_token_ids,
                prompt=req_data.prompt,
                mm_inputs=req_data.mm_inputs,
                mm_positions=req_data.mm_positions,
                sampling_params=sampling_params,
                generator=generator,
                block_ids=req_data.block_ids,
                num_computed_tokens=req_data.num_computed_tokens,
                output_token_ids=[],
            )
            req_ids_to_add.append(req_id)

        # Update the cached states of the resumed requests.
        for req_data in scheduler_output.scheduled_resumed_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]

            req_state.block_ids = req_data.block_ids
            req_state.num_computed_tokens = req_data.num_computed_tokens
            req_ids_to_add.append(req_id)

        # THIS MOVES ALL THE DECODES TO THE FIRST N IN BATCH.
        # Condense the batched states if there are empty indices.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)

        # ALL THE PREFILLS ARE THE LAST M IN THE BATCH.
        # These are added at the end after the bacth is condensed.
        self.input_batch.num_prefills = len(req_ids_to_add)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            self.input_batch.add_request(req_state, None)

    def _prepare_sampling(self,
                          scheduler_output: "SchedulerOutput",
                          prefill_only: bool = False,
                          decode_only: bool = False,
                          seq_idx: Optional[int] = None) -> SamplingMetadata:
        skip_copy = True
        if (scheduler_output.finished_req_ids
                or scheduler_output.preempted_req_ids):
            skip_copy = False
        if (scheduler_output.scheduled_new_reqs
                or scheduler_output.scheduled_resumed_reqs):
            skip_copy = False
        # Create the sampling metadata.
        sampling_metadata = self.input_batch.make_sampling_metadata(
            skip_copy=skip_copy,
            prefill_only=prefill_only,
            decode_only=decode_only,
            seq_idx=seq_idx)
        return sampling_metadata

    def get_habana_paged_attn_buffers(self, block_tables, slot_mapping):

        last_block_usage = [
            slot[0] % self.block_size + 1 for slot in slot_mapping
        ]
        block_groups = [[i] * len(bt) for i, bt in enumerate(block_tables)]
        block_usage = [[self.block_size] * (len(bt) - 1) + [lbu]
                       for bt, lbu in zip(block_tables, last_block_usage)
                       if bt is not None]

        block_list = flatten(block_tables)
        block_groups = flatten(block_groups)
        block_usage = flatten(block_usage)

        assert len(block_list) == len(block_groups)
        assert len(block_list) == len(block_usage)

        padding_fn = None
        if self.use_contiguous_pa:
            block_bucket_size = max(max(block_list) + 1, len(block_list))
            #block_bucket_size = find_bucket(
            #    block_bucket_size,
            #    self.bucketing_global_state.decode_block_bucket_cfg)
            indices: List[Any]
            indices = [None] * block_bucket_size
            for i, bid in enumerate(block_list):
                indices[bid] = i
            padding_fn = lambda tensor, pad_value: gather_list(
                tensor, indices, pad_value)
        else:
            block_bucket_size = len(block_list)
            #block_bucket_size = find_bucket(
            #    len(block_list),
            #    self.bucketing_global_state.decode_block_bucket_cfg)
            padding_fn = lambda tensor, pad_value: pad_list(
                tensor, block_bucket_size, pad_value)

        block_list = padding_fn(block_list, _PAD_BLOCK_ID)
        block_groups = padding_fn(block_groups, -1)
        block_usage = padding_fn(block_usage, 1)

        block_list = torch.tensor(block_list, dtype=torch.int, device='cpu')
        block_groups = torch.tensor(block_groups,
                                    dtype=torch.int,
                                    device='cpu')
        block_usage = torch.tensor(block_usage,
                                   dtype=self.model_config.dtype,
                                   device='cpu')

        block_list = block_list.to(  # type: ignore
            self.device, non_blocking=True)
        block_groups = block_groups.to(  # type: ignore
            self.device, non_blocking=True)
        block_usage = block_usage.to(  # type: ignore
            self.device, non_blocking=True)
        return block_list, block_groups, block_usage

    def _prepare_prefill_inputs(
        self,
        num_scheduled_tokens: List[int],
    ) -> PrefillInputData:
        # Each prefill run separately with shape [1, padded_prompt_len].
        # So we create lists that will be used in execute_model().

        prefill_request_ids = []
        prefill_prompt_lens = []
        prefill_token_ids = []
        prefill_position_ids = []
        prefill_attn_metadata = []
        prefill_logits_indices = []

        # DECODES are the first num_decodes REQUESTS.
        # PREFILLS are the next num_reqs - num_decodes REQUESTS.
        num_reqs = self.input_batch.num_reqs
        num_decodes = self.input_batch.num_decodes
        for idx in range(num_decodes, num_reqs):
            prefill_request_ids.append(self.input_batch.req_ids[idx])

            # STATIC SHAPE: prefills are padded to the next power of 2.
            prompt_len = num_scheduled_tokens[idx]
            padded_prompt_len = _get_padded_prefill_len(prompt_len)
            prefill_prompt_lens.append(prompt_len)
            assert padded_prompt_len <= self.max_model_len

            # TOKEN_IDS.
            token_ids = torch.from_numpy(self.input_batch.token_ids_cpu[
                idx, :padded_prompt_len].reshape(1, -1))
            prefill_token_ids.append(token_ids.to(self.device))

            # POSITIONS.
            positions = self.prefill_positions[:, :padded_prompt_len]
            prefill_position_ids.append(positions.to(self.device))
            # SLOT_MAPPING.
            # The "slot" is the "physical index" of a token in the KV cache.
            # Look up the block_idx in the block table (logical<>physical map)
            # to compute this.
            block_numbers = self.input_batch.block_table_cpu_tensor[
                idx, positions // self.block_size].reshape(1, -1)
            block_offsets = positions % self.block_size
            slot_mapping = block_numbers * self.block_size + block_offsets
            # Set an out of range value for the padding tokens so that they
            # are ignored when inserting into the KV cache.
            slot_mapping[:, prompt_len:] = _PAD_SLOT_ID
            slot_mapping = slot_mapping.long()

            # ATTN_METADATA.
            prefill_attn_metadata.append(
                HPUAttentionMetadata.make_prefill_metadata(
                    seq_lens_tensor=torch.tensor(prompt_len,
                                                 device=self.device),
                    num_prefills=1,
                    num_prefill_tokens=prompt_len,
                    slot_mapping=slot_mapping.to(self.device),
                ))
            prefill_logits_indices.append(prompt_len - 1)

        return PrefillInputData(request_ids=prefill_request_ids,
                                prompt_lens=prefill_prompt_lens,
                                token_ids=prefill_token_ids,
                                position_ids=prefill_position_ids,
                                attn_metadata=prefill_attn_metadata,
                                logits_indices=prefill_logits_indices)

    def _prepare_decode_inputs(self, num_decodes: int,
                               scheduler_output) -> DecodeInputData:
        # Decodes run as one single padded batch with shape [batch, 1]
        #
        # We need to set _PAD_SLOT_ID for the padding tokens in the
        # slot_mapping, such that the attention KV cache insertion
        # logic knows to ignore those indicies. Otherwise, the
        # padding data can be dummy since we have a causal mask.

        if num_decodes == 0:
            return DecodeInputData(num_decodes=0)

        # PAD FOR STATIC SHAPES.
        padded_batch_size = _get_padded_batch_size(num_decodes)

        # POSITIONS. [batch, 1]
        # We slice at the end, since we use the positions for gathering.
        positions = torch.from_numpy(
            self.input_batch.num_computed_tokens_cpu.reshape(-1, 1))
        index = positions.to(torch.int64)
        positions = positions[:padded_batch_size]

        # TOKEN_IDS. [batch, 1]
        token_ids = torch.gather(
            input=torch.from_numpy(self.input_batch.token_ids_cpu),
            dim=1,
            index=index,
        )[:padded_batch_size]

        # SLOT_MAPPING [batch, 1]
        # The "slot" is the "physical index" of a token in the KV cache.
        # Look up the block_idx in the block table (logical<>physical map)
        # to compute this.
        block_number = torch.gather(
            input=self.input_batch.block_table_cpu_tensor,
            dim=1,
            index=(index // self.block_size))
        block_offsets = index % self.block_size
        slot_mapping = block_number * self.block_size + block_offsets
        # Set an out of range value for the padding tokens so that they
        # are ignored when inserting into the KV cache.
        slot_mapping[num_decodes:] = _PAD_SLOT_ID
        slot_mapping = slot_mapping[:padded_batch_size]

        # BLOCK_TABLE [batch, max_num_blocks_per_req]
        block_table = self.input_batch.block_table_cpu_tensor[:
                                                              padded_batch_size]
        # CONTEXT_LENS [batch_size]
        context_lens = (positions.reshape(-1) + 1)

        block_list, block_groups, block_usage = self.get_habana_paged_attn_buffers(
            block_table, slot_mapping)

        num_scheduled_tokens = []
        max_num_scheduled_tokens = 0
        for req_id in self.input_batch.req_ids[:num_decodes]:
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_scheduled_tokens.append(num_tokens)
            max_num_scheduled_tokens = max(max_num_scheduled_tokens,
                                           num_tokens)
        num_scheduled_tokens = np.array(num_scheduled_tokens, dtype=np.int32)
        query_start_loc = torch.empty((num_decodes + 1, ),
                                      dtype=torch.int32,
                                      device="cpu",
                                      pin_memory=self.pin_memory)
        query_start_loc_np = query_start_loc.numpy()
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1:])
        logits_indices = query_start_loc[1:] - 1
        # CPU<>HPU sync happens here.
        logger.info(f'decode token_ids: {token_ids}')
        logger.info(f'decode positions: {positions}')
        logger.info(f'decode logits_indices: {logits_indices}')
        logger.info(f'decode block_list: {block_list}')
        logger.info(f'decode block_usage: {block_usage}')
        logger.info(f'decode block_groups: {block_groups}')
        logger.info(
            f'decode num_decode_tokens: {torch.sum(context_lens).item()}')
        logger.info(f'decode slot_mapping: {slot_mapping}')
        return DecodeInputData(
            num_decodes=num_decodes,
            token_ids=token_ids.to(self.device),
            position_ids=positions.to(self.device),
            logits_indices=logits_indices.to(self.device),
            attn_metadata=HPUAttentionMetadata.make_decode_metadata(
                block_list=block_list.to(self.device),
                block_usage=block_usage.to(self.device),
                block_groups=block_groups.to(self.device),
                num_decode_tokens=torch.sum(context_lens).item(),
                slot_mapping=slot_mapping.to(self.device),
            ))

    def _prepare_inputs(
        self, scheduler_output: "SchedulerOutput"
    ) -> Tuple[PrefillInputData, Optional[DecodeInputData]]:

        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0

        num_reqs = self.input_batch.num_reqs
        num_decodes = self.input_batch.num_decodes

        # Get the number of scheduled tokens for each request.
        # TODO: The Python loop can be slow. Optimize.
        num_scheduled_tokens = []
        for idx, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_scheduled_tokens.append(num_tokens)

            # NOTE: assert that all the decodes are "decodes".
            if idx < num_decodes:
                assert num_tokens == 1

        return (
            self._prepare_prefill_inputs(num_scheduled_tokens),
            self._prepare_decode_inputs(num_decodes, scheduler_output),
        )

    def _execute_model_generic(self, token_ids, position_ids, attn_metadata,
                               logits_indices):
        # FORWARD.
        trimmed_attn_metadata = trim_attn_metadata(attn_metadata)
        hidden_states = self.model.forward(input_ids=token_ids,
                                           positions=position_ids,
                                           attn_metadata=trimmed_attn_metadata,
                                           kv_caches=self.kv_caches)

        #hidden_states = hidden_states[:num_scheduled_tokens]
        hidden_states = hidden_states[logits_indices]
        logits = self.model.compute_logits(hidden_states, None)
        return logits

    def _execute_model_prefills(self, scheduler_output, prefill_data,
                                prefill_start_idx, sampled_token_ids, logprobs,
                                logprob_token_ids):
        if len(list(prefill_data.zipped())) == 0:
            logger.info(
                f"[ENGINE_ITER {self._ENGINE_ITER}] No prefills detected.")
            return None, None

        for idx, (req_id, prompt_len, token_ids, position_ids, attn_metadata,
                  logits_indices) in enumerate(prefill_data.zipped()):
            logits = self._execute_model_generic(token_ids, position_ids,
                                                 attn_metadata, logits_indices)

            # Sample the next token and get logprobs if needed.
            sampling_metadata = self._prepare_sampling(scheduler_output,
                                                       seq_idx=idx)
            sampler_output = self.model.sample(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )

            # NOTE: HPU<>CPU sync happens here.
            token_id = sampler_output.sampled_token_ids.cpu().item()
            sampled_token_ids[prefill_start_idx + idx] = token_id
            req_state = self.requests[req_id]

            if sampler_output.logprob_token_ids is not None:
                logprob_token_ids[
                    prefill_start_idx +
                    idx] = sampler_output.logprob_token_ids.cpu()
            if sampler_output.logprobs is not None:
                logprobs[prefill_start_idx +
                         idx] = sampler_output.logprobs.cpu()

            # TODO: ASSERT NO PREFIX CACHING.
            assert req_state.num_computed_tokens == 0
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])

            # TODO: ASSERT NO CHUNKED PREFILL.
            assert seq_len == req_state.num_tokens
            assert prompt_len == seq_len

            # UPDATE REQUEST STATE.
            req_idx = self.input_batch.req_id_to_index[req_id]
            self.input_batch.token_ids_cpu[req_idx, seq_len] = token_id
            req_state.output_token_ids.append(token_id)
            detokenized = self._tokenizer.decode(
                token_id) if token_id >= 0 and token_id <= len(
                    self._tokenizer) else 'INVALID!!!'
            logger.info(
                f"[ENGINE_ITER {self._ENGINE_ITER}] Prefill {idx} (req_id:{req_id}) generated token id: {token_id} ({detokenized!r})."
            )
        return sampled_token_ids, logprobs, logprob_token_ids

    def _execute_model_decode(self, scheduler_output, decode_data,
                              sampled_token_ids, logprobs, logprob_token_ids):
        logger.info(
            f"[ENGINE_ITER {self._ENGINE_ITER}] Executing decode on {decode_data.num_decodes} seqs..."
        )
        logits = self._execute_model_generic(decode_data.token_ids,
                                             decode_data.position_ids,
                                             decode_data.attn_metadata,
                                             decode_data.logits_indices)

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self._prepare_sampling(scheduler_output,
                                                   decode_only=True)
        sampler_output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )

        # NOTE: TPU<>CPU sync happens here.
        # We need to call .cpu() first to avoid recompilation.
        token_ids = sampler_output.sampled_token_ids.cpu()[:decode_data.
                                                           num_decodes]
        sampled_token_ids_list = token_ids.tolist()
        sampled_token_ids[:decode_data.num_decodes] = token_ids

        # UPDATE REQUEST STATE.
        for i, req_id in enumerate(
                self.input_batch.req_ids[:decode_data.num_decodes]):
            req_state = self.requests[req_id]

            # TODO: ASSERT NO CHUNKED PREFILL.
            assert scheduler_output.num_scheduled_tokens[req_id] == 1
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            assert seq_len == req_state.num_tokens

            token_id = sampled_token_ids_list[i]
            self.input_batch.token_ids_cpu[i, seq_len] = token_id
            req_state.output_token_ids.append(token_id)
            detokenized = self._tokenizer.decode(
                token_id) if token_id >= 0 and token_id <= len(
                    self._tokenizer) else 'INVALID!!!'
            logger.info(
                f"[ENGINE_ITER {self._ENGINE_ITER}] Decode {i} (req_id:{req_id}) generated token id: {token_id} ({detokenized!r})."
            )
        return sampled_token_ids, None, None

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput:
        self._update_states(scheduler_output)
        logger.info(
            f'[ENGINE_ITER {self._ENGINE_ITER}] Starting engine iteration with {self.input_batch.num_reqs} reqs...'
        )
        prefill_data, decode_data = self._prepare_inputs(scheduler_output)
        num_reqs = self.input_batch.num_reqs
        sampled_token_ids = torch.empty(num_reqs, dtype=torch.int32)
        #FIXME(kzawora): Currently there's no handling of logprobs. Fix that later.
        logprob_token_ids = None
        logprobs = None
        ######################### DECODES #########################
        # Decodes run as one single batch with [padded_batch, 1]
        if decode_data.num_decodes > 0:
            sampled_token_ids, logprob_token_ids, logprobs = self._execute_model_decode(
                scheduler_output, decode_data, sampled_token_ids, logprobs,
                logprob_token_ids)

        ######################### PREFILLS #########################
        # Prefills run separately with shape [1, padded_prefill_len]
        if num_reqs - decode_data.num_decodes > 0:
            logger.info(
                f"[ENGINE_ITER {self._ENGINE_ITER}] Executing prefill on {num_reqs - decode_data.num_decodes} seqs..."
            )
            sampled_token_ids, logprob_token_ids, logprobs = self._execute_model_prefills(
                scheduler_output, prefill_data, decode_data.num_decodes,
                sampled_token_ids, logprobs, logprob_token_ids)

        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids[:num_reqs],
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids_cpu=sampled_token_ids,
            logprob_token_ids_cpu=logprob_token_ids,
            logprobs_cpu=logprobs,
        )
        import pdb
        pdb.set_trace()
        logger.info(
            f'[ENGINE_ITER {self._ENGINE_ITER}] Engine iteration done!')
        if False:
            for i in range(num_reqs):
                req_id = self.input_batch.req_ids[i]
                req_idx = self.input_batch.req_id_to_index[req_id]
                token_ids = self.input_batch.token_ids_cpu[req_idx]
                prompt = self._tokenizer.decode(
                    token_ids[:self.input_batch.
                              num_prompt_tokens_cpu[req_idx]])
                generated = self._tokenizer.decode(token_ids[
                    self.input_batch.num_prompt_tokens_cpu[req_idx]:max(
                        self.input_batch.num_prompt_tokens_cpu[req_idx], self.
                        input_batch.num_computed_tokens_cpu[req_idx])])
                logger.info(
                    f'[ENGINE_ITER {self._ENGINE_ITER}] REQ:{req_id} IDX:{req_idx} generated token: {self._tokenizer.decode(sampled_token_ids[req_idx])!r}, all generated so far: {generated!r}'
                )
        self._ENGINE_ITER += 1
        return model_runner_output

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        with HabanaMemoryProfiler() as m:  # noqa: SIM117
            self.model = get_model(vllm_config=self.vllm_config)
            self.model = _maybe_wrap_in_hpu_graph(
                self.model,
                self.block_size,
                dtype=self.model_config.dtype,
                enforce_eager=self.model_config.enforce_eager)
        self.model_memory_usage = m.consumed_device_memory
        logger.info("Loading model weights took %.4f GB",
                    self.model_memory_usage / float(2**30))

    def _dummy_run(self, batch_size: int, seq_len: int,
                   kv_caches: List[torch.Tensor], is_prompt: bool) -> None:
        """Dummy warmup run for memory usage and graph compilation."""

        input_ids = torch.zeros((batch_size, seq_len),
                                dtype=torch.int32,
                                device=self.device)
        position_ids = torch.zeros((batch_size, seq_len),
                                   dtype=torch.int32,
                                   device=self.device)
        slot_mapping = torch.zeros((batch_size, seq_len),
                                   dtype=torch.int64,
                                   device=self.device)
        block_tables = None if is_prompt else torch.zeros(
            (batch_size, self.max_num_blocks_per_req),
            dtype=torch.int32,
            device=self.device,
        )
        context_lens = None if is_prompt else torch.ones(
            (batch_size, ),
            dtype=torch.int32,
            device=self.device,
        )
        block_indices, block_offsets = precompute_indices_and_offsets(
            self.block_size, slot_mapping, True)
        # TODO(kzawora): form proper hpu attn metadata
        prefix_block_list_tensor = None
        attn_metadata = HPUAttentionMetadata(
            is_prompt=True,
            block_list=prefix_block_list_tensor,
            block_mapping=None,
            block_usage=None,
            block_indices=block_indices,
            block_offsets=block_offsets,
            block_scales=None,
            block_groups=None,
            attn_bias=None,
            seq_lens_tensor=context_lens if is_prompt else None,
            context_lens_tensor=None if is_prompt else context_lens,
            num_prefills=batch_size if is_prompt else 0,
            num_prefill_tokens=batch_size * seq_len if is_prompt else 0,
            num_decode_tokens=0 if is_prompt else batch_size,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=
            None  # FIXME(kzawora): mutli-modality will not work here
        )
        selected_token_indices = torch.arange(0,
                                              seq_len * batch_size,
                                              device=self.device)
        trimmed_attn_metadata = trim_attn_metadata(attn_metadata)
        return  # bypass dummy run for now
        # Dummy run.
        self.model.forward(input_ids=input_ids,
                           positions=position_ids,
                           kv_caches=kv_caches,
                           attn_metadata=trimmed_attn_metadata,
                           selected_token_indices=selected_token_indices)

    @torch.inference_mode()
    def profile_run(self) -> None:
        """Profile to measure peak memory during forward pass."""

        # use an empty tensor instead of `None`` to force Dynamo to pass
        # it by reference, rather by specializing on the value `None`.
        # the `dtype` argument does not matter, and we use `float32` as
        # a placeholder (it has wide hardware support).
        # it is important to create tensors inside the loop, rather than
        # multiplying the list, to avoid Dynamo from treating them as
        # tensor aliasing.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [None] * num_layers

        # Round to multiple of 16.
        seq_len = (self.max_num_tokens + 15) // 16 * 16

        # Run empty forward.
        self._dummy_run(batch_size=1,
                        seq_len=seq_len,
                        kv_caches=kv_caches,
                        is_prompt=True)
        torch.hpu.synchronize()

    @torch.inference_mode()
    def capture_model(self) -> None:
        start_time = time.perf_counter()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]

        with set_forward_context(None):
            # Trigger CUDA graph capture for specific shapes.
            # Capture the large shapes first so that the smaller shapes
            # can reuse the memory pool allocated for the large shapes.
            for num_tokens in reversed(self.cudagraph_batch_sizes):
                self.model(
                    self.input_ids[:num_tokens],
                    self.positions[:num_tokens],
                    kv_caches=self.kv_caches,
                    attn_metadata=None,
                )

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, cuda_graph_size / (1 << 30))

    def initialize_kv_cache(self, num_blocks: int) -> None:
        assert len(self.kv_caches) == 0
        kv_cache_shape = HPUAttentionBackendV1.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        self.kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        dtype = self.dtype
        if self.device != 'hpu' and not is_fake_hpu() \
          and self.dtype == torch.float8_e4m3fn:
            dtype = torch.uint8
        for _ in range(self.num_attn_layers):
            key_cache = torch.zeros(kv_cache_shape,
                                    dtype=dtype,
                                    device=self.device)
            value_cache = torch.zeros(kv_cache_shape,
                                      dtype=dtype,
                                      device=self.device)
            kv_layer = (key_cache, value_cache)
            self.kv_caches.append(kv_layer)
        htorch.hpu.synchronize()

    def _get_padded_batch_size(self, batch_size: int) -> Optional[int]:
        # TODO: Optimize this?
        for size in self.cudagraph_batch_sizes:
            if batch_size <= size:
                return size
        return None


@dataclass
class CachedRequestState:

    req_id: str
    prompt_token_ids: List[int]
    prompt: Optional[str]
    mm_inputs: List[MultiModalKwargs]
    mm_positions: List["PlaceholderRange"]
    sampling_params: SamplingParams
    generator: Optional[torch.Generator]

    block_ids: List[int]
    num_computed_tokens: int
    output_token_ids: List[int]

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)


class InputBatch:

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_blocks_per_req: int,
        device: torch.device,
        pin_memory: bool,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.device = device
        self.pin_memory = pin_memory

        self.req_ids: List[Optional[str]] = [None] * max_num_reqs
        self.req_id_to_index: Dict[str, int] = {}

        self.token_ids_cpu = np.empty((max_num_reqs, max_model_len),
                                      dtype=np.int32)
        self.num_computed_tokens_cpu = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_output_tokens_cpu = np.empty(max_num_reqs, dtype=np.int32)
        self.num_prompt_tokens_cpu = np.empty(max_num_reqs, dtype=np.int32)

        # Attention-related.
        self.block_table = torch.zeros((max_num_reqs, max_num_blocks_per_req),
                                       device=self.device,
                                       dtype=torch.int32)
        self.block_table_cpu_tensor = torch.zeros(
            (max_num_reqs, max_num_blocks_per_req),
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self.block_table_cpu = self.block_table_cpu_tensor.numpy()

        # Sampling-related.
        self.temperature = torch.empty((max_num_reqs, ),
                                       dtype=torch.float32,
                                       device=device)
        self.temperature_cpu_tensor = torch.empty((max_num_reqs, ),
                                                  dtype=torch.float32,
                                                  device="cpu",
                                                  pin_memory=pin_memory)
        self.temperature_cpu = self.temperature_cpu_tensor.numpy()
        self.greedy_reqs: Set[str] = set()
        self.random_reqs: Set[str] = set()

        self.top_p = torch.empty((max_num_reqs, ),
                                 dtype=torch.float32,
                                 device=device)
        self.top_p_cpu_tensor = torch.empty((max_num_reqs, ),
                                            dtype=torch.float32,
                                            device="cpu",
                                            pin_memory=pin_memory)
        self.top_p_cpu = self.top_p_cpu_tensor.numpy()
        self.top_p_reqs: Set[str] = set()

        self.top_k = torch.empty((max_num_reqs, ),
                                 dtype=torch.int32,
                                 device=device)
        self.top_k_cpu_tensor = torch.empty((max_num_reqs, ),
                                            dtype=torch.int32,
                                            device="cpu",
                                            pin_memory=pin_memory)
        self.top_k_cpu = self.top_k_cpu_tensor.numpy()
        self.top_k_reqs: Set[str] = set()

        # req_index -> generator
        self.generators: Dict[int, torch.Generator] = {}

        self.num_logprobs: Dict[str, int] = {}
        self.prompt_logprob_reqs: Set[str] = set()

        self.num_prefills = 0

    def add_request(
        self,
        request: "CachedRequestState",
        req_index: Optional[int] = None,
    ) -> None:
        if req_index is None:
            req_index = self.num_reqs
        assert req_index < self.max_num_reqs

        req_id = request.req_id
        self.req_ids[req_index] = req_id
        self.req_id_to_index[req_id] = req_index

        # Copy the prompt token ids and output token ids.
        num_prompt_tokens = len(request.prompt_token_ids)
        self.token_ids_cpu[
            req_index, :num_prompt_tokens] = request.prompt_token_ids
        start_idx = num_prompt_tokens
        end_idx = start_idx + len(request.output_token_ids)
        self.token_ids_cpu[req_index,
                           start_idx:end_idx] = request.output_token_ids

        self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens
        #self.num_output_tokens_cpu[req_index] = request.num_output_tokens
        self.num_prompt_tokens_cpu[req_index] = len(request.prompt_token_ids)
        num_blocks = len(request.block_ids)
        self.block_table_cpu[req_index, :num_blocks] = request.block_ids

        sampling_params = request.sampling_params
        self.temperature_cpu[req_index] = sampling_params.temperature
        if sampling_params.sampling_type == SamplingType.GREEDY:
            self.greedy_reqs.add(req_id)
        else:
            self.random_reqs.add(req_id)

        self.top_p_cpu[req_index] = sampling_params.top_p
        if sampling_params.top_p < 1:
            self.top_p_reqs.add(req_id)
        self.top_k_cpu[req_index] = sampling_params.top_k
        if sampling_params.top_k > 0:
            self.top_k_reqs.add(req_id)

        self.generators[req_index] = request.generator

        num_logprobs = sampling_params.logprobs
        if num_logprobs is not None and num_logprobs > 0:
            self.num_logprobs[req_id] = num_logprobs
        if sampling_params.prompt_logprobs:
            self.prompt_logprob_reqs.add(req_id)

    def remove_request(self, req_id: str) -> Optional[int]:
        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return None
        self.req_ids[req_index] = None

        self.greedy_reqs.discard(req_id)
        self.random_reqs.discard(req_id)
        self.top_p_reqs.discard(req_id)
        self.top_k_reqs.discard(req_id)
        self.generators.pop(req_index, None)
        self.num_logprobs.pop(req_id, None)
        self.prompt_logprob_reqs.discard(req_id)
        return req_index

    def clear(self) -> None:
        self.req_ids = [None] * self.max_num_reqs
        self.req_id_to_index.clear()
        self.greedy_reqs.clear()
        self.random_reqs.clear()
        self.top_p_reqs.clear()
        self.top_k_reqs.clear()
        self.generators.clear()
        self.num_logprobs.clear()
        self.prompt_logprob_reqs.clear()

    def condense(self, empty_req_indices: List[int]) -> None:
        if self.num_reqs == 0:
            # The batched states are empty.
            return

        # NOTE(woosuk): This function assumes that the empty_req_indices
        # is sorted in descending order.
        last_req_index = self.num_reqs + len(empty_req_indices) - 1
        while empty_req_indices:
            # Find the largest non-empty index.
            while last_req_index in empty_req_indices:
                last_req_index -= 1

            # Find the smallest empty index.
            empty_index = empty_req_indices.pop()
            if empty_index >= last_req_index:
                break

            # Swap the states.
            req_id = self.req_ids[last_req_index]
            self.req_ids[empty_index] = req_id
            self.req_ids[last_req_index] = None
            self.req_id_to_index[req_id] = empty_index

            # TODO(woosuk): Optimize the copy of token_ids_cpu and
            # block_table_cpu.
            self.token_ids_cpu[empty_index] = self.token_ids_cpu[
                last_req_index]
            self.num_computed_tokens_cpu[
                empty_index] = self.num_computed_tokens_cpu[last_req_index]
            self.block_table_cpu[empty_index] = self.block_table_cpu[
                last_req_index]
            self.temperature_cpu[empty_index] = self.temperature_cpu[
                last_req_index]
            self.top_p_cpu[empty_index] = self.top_p_cpu[last_req_index]
            self.top_k_cpu[empty_index] = self.top_k_cpu[last_req_index]
            generator = self.generators.pop(last_req_index, None)
            if generator is not None:
                self.generators[empty_index] = generator

            # Decrement last_req_index since it is now empty.
            last_req_index -= 1

    def make_sampling_metadata(
            self,
            skip_copy: bool = False,
            prefill_only: bool = False,
            decode_only: bool = False,
            seq_idx: Optional[int] = None) -> SamplingMetadata:
        start_seq = 0
        end_seq = self.num_reqs
        if prefill_only:
            start_seq = self.num_decodes
            assert not (
                decode_only or seq_idx
            ), "make_sampling_metadata can be either prefill_only, decode_only, seq_idx, or neither."
        elif decode_only:
            end_seq = self.num_decodes
            assert not (
                prefill_only or seq_idx
            ), "make_sampling_metadata can be either prefill_only, decode_only, seq_idx, or neither."
        elif seq_idx is not None:
            start_seq = seq_idx
            end_seq = seq_idx + 1
            assert not (
                prefill_only or decode_only
            ), "make_sampling_metadata can be either prefill_only, decode_only, seq_idx, or neither."

        if not skip_copy:
            self.temperature[start_seq:end_seq].copy_(
                self.temperature_cpu_tensor[start_seq:end_seq],
                non_blocking=True)
            self.top_p[start_seq:end_seq].copy_(
                self.top_p_cpu_tensor[start_seq:end_seq], non_blocking=True)
            self.top_k[start_seq:end_seq].copy_(
                self.top_k_cpu_tensor[start_seq:end_seq], non_blocking=True)
        return SamplingMetadata(
            temperature=self.temperature[start_seq:end_seq],
            all_greedy=self.all_greedy,
            all_random=self.all_random,
            top_p=self.top_p[start_seq:end_seq],
            top_k=self.top_k[start_seq:end_seq],
            no_top_p=self.no_top_p,
            no_top_k=self.no_top_k,
            generators=self.generators,
            max_num_logprobs=self.max_num_logprobs,
        )

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    @property
    def num_decodes(self) -> int:
        return self.num_reqs - self.num_prefills

    @property
    def all_greedy(self) -> bool:
        return len(self.random_reqs) == 0

    @property
    def all_random(self) -> bool:
        return len(self.greedy_reqs) == 0

        #    @property
        #    def all_prefill(self) -> bool:
        return all(
            output_tokens == 0
            for output_tokens in self.num_output_tokens_cpu[:self.num_reqs])


#    @property
#    def all_decode(self) -> bool:
#        return all(output_tokens > 0 for output_tokens in self.num_output_tokens_cpu[:self.num_reqs])

    @property
    def mixed_batch(self) -> bool:
        return not self.all_prefill and not self.all_decode

    @property
    def no_top_p(self) -> bool:
        return len(self.top_p_reqs) == 0

    @property
    def no_top_k(self) -> bool:
        return len(self.top_k_reqs) == 0

    @property
    def max_num_logprobs(self) -> int:
        return max(self.num_logprobs.values()) if self.num_logprobs else 0

    @property
    def no_logprob(self) -> bool:
        return len(self.num_logprobs) == 0

    @property
    def no_prompt_logprob(self) -> bool:
        return len(self.prompt_logprob_reqs) == 0
