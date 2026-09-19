import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384   # Maximum number of tokens that can be processed in one batch (prompt + generation tokens together)
    max_num_seqs: int = 512   # Maximum number of concurrent sequences (requests) the engine will handle
    max_model_len: int = 4096  # Maximum sequence length the engine will allow (prompt + generated tokens)
    gpu_memory_utilization: float = 0.9  
    tensor_parallel_size: int = 1   # How many GPUs to split the model across (tensor parallelism)
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1   # End-of-sequence token ID (usually loaded from the tokenizer later)
    kvcache_block_size: int = 256  # Size of each block in the paged KV cache (in terms of number of token count)
    num_kvcache_blocks: int = -1   # Total number of KV-cache blocks available (–1 means “calculate automatically”) 

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)


# total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
# model_memory = memory currently occupied by weights + activations + overhead
# available_memory = (total_gpu_memory - model_memory) * gpu_memory_utilization
# Example: if gpu_memory_utilization = 0.9, only 90% of the free memory is allowed to be used for the KV cache.
# bytes_per_block = 2 * num_layers * num_kv_heads * head_dim * kvcache_block_size * bytes_per_element
# num_kvcache_blocks = floor(available_memory / bytes_per_block)
