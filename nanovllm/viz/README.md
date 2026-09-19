# Paged Attention Scope

A step-debugger for nano-vllm's scheduler and KV block manager.

```bash
python -m nanovllm.viz          # opens http://127.0.0.1:8008/
```

Enter a prompt, pick a model, hit **Trace request**. The engine runs to
completion, every step is recorded, and the UI replays it — play/pause, step
forward and back, scrub, and slow motion down to 0.1×. A decode step takes
milliseconds; replay is the only way to actually watch one.

## What you are looking at

| Panel | Shows |
|---|---|
| **Token tape** | Prompt tokens laid into fixed-size logical blocks, each mapped to a physical block. Chips turn from *pending* → *computing* → *cached*; generated tokens append one per step. |
| **KV cache pool** | Every physical block. Filled = held, hatched = `ref_count > 1`, amber ring = allocated this step. |
| **Block table** | `seq.block_table` — logical `L0 → #12`, the page table made literal. |
| **Batch this step** | Tokens scheduled against `max_num_batched_tokens`; hatched when chunked. |
| **Why banner** | The scheduler's reasoning: chunked prefill, prefix cache hit, admission blocked, new block appended. |
| **Event log** | Raw tracer events for the step. |

## Models

The dropdown lists the mock engine plus any model found under `~/huggingface`,
`~/models`, `$NANOVLLM_MODELS` or the HF hub cache. Unavailable models stay in
the list with the reason attached (missing dependency, no CUDA, unsupported
architecture) rather than silently disappearing.

**Mock engine** runs the genuine `Scheduler` and `BlockManager` with a stand-in
forward pass. Paging and scheduling are decided entirely by those two classes,
so every block, page-table entry and admission decision is real — only the
matmuls are fake. It needs no GPU and no weights, and it is the only mode that
permits a block size small enough to watch.

**Real models** (Qwen3, as in `example.py`) need torch+CUDA, transformers,
triton and flash-attn.

Three things behave differently on a GPU:

- `block_size` is pinned to **256** — the paged attention kernel requires it.
  A short request therefore occupies one block and never pages. Generate a few
  hundred tokens, or paste a long prompt, to see `may_append()` add pages.
- `allocate_kv_cache()` sizes the pool to fill VRAM, so it typically yields
  hundreds or thousands of blocks. That is neither legible as a grid nor
  reachable by one request, so the visualizer **caps the pool** to the *pool
  blocks* value and the block manager hands out only that subset. The caption
  reports both numbers. Raise the cap to see more; the real ceiling is what the
  GPU allocated.
- The `ModelRunner` is **built once per model path and reused** across traces —
  weights and NCCL init cost seconds. Traces are serialized behind a lock, since
  they share one CUDA stream.

## Tracing from your own code

Tracing is off by default and costs one `is None` check per call site.

```python
from nanovllm.viz.tracer import tracing

with tracing() as tr:
    llm.generate(prompts, sampling_params)
    events = tr.drain()
```

Hook sites live in `engine/block_manager.py` (`block_alloc`, `block_free`,
`block_reuse`, `block_append`, `block_hashed`, `prefix_hit`) and
`engine/scheduler.py` (`seq_scheduled`, `admission_blocked`, `preempt`,
`token_emitted`, `seq_finished`).

## Known limits

- One request per trace. Continuous batching, prefix sharing across requests
  and preemption need several — that's the multi-request view.
- A lone sequence that outgrows the pool preempts itself forever and trips
  `assert scheduled_seqs` in the scheduler. The server pre-checks peak block
  demand and explains it instead of crashing.
