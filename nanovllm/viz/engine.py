"""Runs one (or more) requests and records a frame per engine step.

Mirrors LLMEngine.step() rather than calling it, so the scheduled sequences and
the prefill/decode decision are visible to the recorder.
"""

import atexit
import itertools
from dataclasses import dataclass
from threading import Lock
from time import perf_counter

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams
from nanovllm.viz.mock_runner import MockModelRunner
from nanovllm.viz.models import MOCK_ID, resolve
from nanovllm.viz.tokenizer import load_tokenizer, piece_text
from nanovllm.viz.tracer import tracing


# The paged attention kernel requires 256-token blocks. The mock engine has no
# kernel, so it can use a size small enough to actually watch.
REAL_BLOCK_SIZE = 256
MAX_STEPS = 2000

# Loading weights and initializing NCCL costs seconds, so a real ModelRunner is
# built once per model path and reused. CUDA work must not overlap, so every
# trace holds the lock.
_RUNNERS: dict[str, object] = {}
_LOCK = Lock()


def _real_runner(model_path: str, max_num_batched_tokens: int, max_num_seqs: int, eos: int):
    """Build (or reuse) a ModelRunner. Returns (runner, blocks the GPU allocated)."""
    if model_path in _RUNNERS:
        runner = _RUNNERS[model_path]
        return runner, runner.config.num_kvcache_blocks

    from nanovllm.config import Config
    from nanovllm.engine.model_runner import ModelRunner

    cfg = Config(
        model_path,
        max_num_batched_tokens=max(max_num_batched_tokens, REAL_BLOCK_SIZE),
        max_num_seqs=max_num_seqs,
        kvcache_block_size=REAL_BLOCK_SIZE,
        enforce_eager=True,   # skip cudagraph capture; tracing is not a benchmark
    )
    cfg.eos = eos
    runner = ModelRunner(cfg, 0, [])
    _RUNNERS[model_path] = runner
    atexit.register(lambda: runner.call("exit"))
    return runner, cfg.num_kvcache_blocks


@dataclass
class VizConfig:
    """Only the fields Scheduler reads — avoids requiring a real model dir."""
    max_num_seqs: int
    max_num_batched_tokens: int
    eos: int
    kvcache_block_size: int
    num_kvcache_blocks: int


def _note_from(event: dict) -> dict | None:
    k = event["kind"]
    if k == "prefix_hit":
        return {"k": "hit", "seq": event["seq_id"], "blocks": event["blocks"]}
    if k == "preempt":
        return {"k": "evict", "seq": event["seq_id"], "blocks": event["blocks_freed"],
                "lost": event["tokens_lost"]}
    if k == "admission_blocked" and event["reason"] == "no_blocks":
        return {"k": "blocks", "seq": event["seq_id"], "need": event["need"], "free": event["free"]}
    if k == "admission_blocked" and event["reason"] == "token_budget":
        return {"k": "budget", "seq": event["seq_id"], "want": event.get("want"), "left": event.get("left")}
    if k == "seq_scheduled" and event.get("chunked"):
        return {"k": "chunk", "seq": event["seq_id"], "took": event["tokens"], "of": event["of"]}
    if k == "block_alloc" and event.get("evicted_cache"):
        return {"k": "evict_cache", "block": event["block_id"]}
    if k == "seq_finished":
        return {"k": "done", "seq": event["seq_id"], "reason": event["reason"]}
    return None


def _run_trace(
    model_id: str,
    prompt: str,
    *,
    max_tokens: int = 24,
    temperature: float = 0.6,
    block_size: int = 8,
    num_blocks: int = 48,
    max_num_batched_tokens: int = 32,
    max_num_seqs: int = 4,
    apply_chat_template: bool = True,
) -> dict:
    entry = resolve(model_id) if model_id != MOCK_ID else None
    is_mock = model_id == MOCK_ID
    if entry is not None and not entry.available:
        raise RuntimeError(f"{entry.label} cannot run here: {entry.reason}")

    model_path = entry.path if entry else None
    tokenizer, tok_kind = load_tokenizer(model_path)

    if is_mock:
        block_size = max(1, int(block_size))
    else:
        block_size = REAL_BLOCK_SIZE

    text = prompt
    if apply_chat_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt
    prompt_ids = tokenizer.encode(text)
    if not prompt_ids:
        raise ValueError("prompt is empty after tokenization")

    # A lone sequence that outgrows the pool preempts itself forever, and the
    # scheduler's `assert scheduled_seqs` fires. Catch it here with an
    # explanation the user can act on.
    eos = getattr(tokenizer, "eos_token_id", 0) or 0
    Sequence.block_size = block_size

    pool_allocated = num_blocks
    if is_mock:
        runner = MockModelRunner(tokenizer, eos)
    else:
        runner, pool_allocated = _real_runner(model_path, max_num_batched_tokens, max_num_seqs, eos)
        # The GPU sizes the KV cache to fill VRAM - often thousands of blocks,
        # which is neither legible nor reachable by a short request. The block
        # manager may hand out any subset, so show (and use) only the first N.
        num_blocks = min(num_blocks, pool_allocated)

    peak_blocks = -(-(len(prompt_ids) + max_tokens) // block_size)
    if peak_blocks > num_blocks:
        extra = "" if is_mock else f" The GPU allocated {pool_allocated} blocks in total, so you can raise it that far."
        raise ValueError(
            f"This request peaks at {peak_blocks} blocks "
            f"({len(prompt_ids)} prompt + {max_tokens} generated tokens / {block_size} per block) "
            f"but the visible pool holds {num_blocks}. Raise 'pool blocks' to at least {peak_blocks}, "
            f"or lower 'max tokens'.{extra} "
            f"(A lone sequence cannot be preempted - that needs a second request to evict.)"
        )

    config = VizConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        eos=eos,
        kvcache_block_size=block_size,
        num_kvcache_blocks=num_blocks,
    )
    scheduler = Scheduler(config)

    # Must come after the runner: warmup_model() builds throwaway Sequences that
    # would otherwise consume seq_id 0.
    Sequence.counter = itertools.count()
    seq = Sequence(prompt_ids, SamplingParams(temperature=temperature, max_tokens=max_tokens))
    scheduler.add(seq)
    seqs = [seq]

    token_texts = [piece_text(tokenizer, t) for t in prompt_ids]
    frames = []
    preemptions = 0
    reused_tokens = 0
    prompt_total = len(prompt_ids)

    with tracing() as tr:
        for step in range(MAX_STEPS):
            if scheduler.is_finished():
                break
            tr.step = step
            t0 = perf_counter()
            batch, is_prefill = scheduler.schedule()
            snapshot = [
                {"id": s.seq_id, "tokens": s.num_scheduled_tokens, "chunked": False}
                for s in batch
            ]
            token_ids = runner.call("run", batch, is_prefill)
            scheduler.postprocess(batch, token_ids, is_prefill)
            latency = (perf_counter() - t0) * 1000

            events = tr.drain()
            notes = [n for n in (_note_from(e) for e in events) if n]
            chunked = {e["seq_id"] for e in events if e["kind"] == "seq_scheduled" and e.get("chunked")}
            for row in snapshot:
                row["chunked"] = row["id"] in chunked
            preemptions += sum(1 for e in events if e["kind"] == "preempt")
            reused_tokens += sum(e["tokens"] for e in events if e["kind"] == "prefix_hit")

            emitted = None
            for e in events:
                if e["kind"] == "token_emitted":
                    token_texts.append(piece_text(tokenizer, e["token_id"]))
                    emitted = {"id": e["token_id"], "text": token_texts[-1], "index": e["index"]}

            bm = scheduler.block_manager
            owners: dict[int, list[int]] = {}
            for s in seqs:
                if s.status == SequenceStatus.FINISHED:
                    continue
                for bid in s.block_table:
                    owners.setdefault(bid, [])
                    if s.seq_id not in owners[bid]:
                        owners[bid].append(s.seq_id)

            blocks = [
                {
                    "id": b.block_id,
                    "ref": b.ref_count,
                    "owners": owners.get(b.block_id, []),
                    "hashed": b.hash != -1,
                    "free": b.block_id not in bm.used_block_ids,
                    "tokens": [piece_text(tokenizer, t) for t in b.token_ids[:6]],
                }
                for b in bm.blocks
            ]

            in_batch = {s.seq_id for s in batch}
            evicted = {e["seq_id"] for e in events if e["kind"] == "preempt"}
            row_state = {}
            for s in seqs:
                if s.seq_id in evicted:
                    row_state[s.seq_id] = "evict"
                elif s.status == SequenceStatus.FINISHED:
                    row_state[s.seq_id] = "done"
                elif s.seq_id in in_batch:
                    row_state[s.seq_id] = "prefill" if is_prefill else "decode"
                else:
                    row_state[s.seq_id] = "waiting"

            frames.append({
                "step": len(frames),
                "phase": "prefill" if is_prefill else "decode",
                "batch": snapshot,
                "budget": max_num_batched_tokens,
                "blocks": blocks,
                "blockTables": {s.seq_id: list(s.block_table) for s in seqs},
                "seqMeta": {s.seq_id: {
                    "numTokens": s.num_tokens,
                    "numPrompt": s.num_prompt_tokens,
                    "numCached": s.num_cached_tokens,
                    "numScheduled": snapshot[0]["tokens"] if snapshot and snapshot[0]["id"] == s.seq_id else 0,
                    "status": s.status.name.lower(),
                    "maxTokens": s.max_tokens,
                } for s in seqs},
                "rowState": row_state,
                "used": len(bm.used_block_ids),
                "total": num_blocks,
                "running": len(scheduler.running),
                "waiting": len(scheduler.waiting),
                "preemptions": preemptions,
                "reuse": reused_tokens / prompt_total if prompt_total else 0.0,
                "notes": notes,
                "emitted": emitted,
                "latencyMs": round(latency, 3),
                "events": events,
            })

    completion = seq.completion_token_ids
    try:
        completion_text = tokenizer.decode(completion)
    except Exception:
        completion_text = "".join(token_texts[len(prompt_ids):])

    return {
        "frames": frames,
        "cfg": {
            "blockSize": block_size,
            "numBlocks": num_blocks,
            "poolAllocated": pool_allocated,
            "maxNumBatchedTokens": max_num_batched_tokens,
            "maxNumSeqs": max_num_seqs,
        },
        "seqs": [{"id": s.seq_id, "numPrompt": s.num_prompt_tokens} for s in seqs],
        "tokenTexts": token_texts,
        "promptText": text,
        "completionText": completion_text,
        "model": {
            "id": model_id,
            "label": "Mock engine" if is_mock else entry.label,
            "kind": "mock" if is_mock else "local",
            "tokenizer": tok_kind,
        },
    }


def run_trace(*args, **kwargs) -> dict:
    """Serialize traces: a cached ModelRunner and its CUDA stream are shared."""
    with _LOCK:
        return _run_trace(*args, **kwargs)
