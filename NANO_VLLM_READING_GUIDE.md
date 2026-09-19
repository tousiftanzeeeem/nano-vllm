# 📖 Nano-vLLM Reading Guide

> A carefully curated reading order for [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)  
> ~1,200 lines of clean, educational code that teaches you how modern LLM inference engines really work.

<p align="center">
  <img src="https://img.shields.io/badge/Lines-≈1%2C200-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Difficulty-Beginner%20→%20Advanced-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Focus-PagedAttention%20%7C%20Continuous%20Batching-orange?style=for-the-badge" />
</p>

---

## 🗺️ Mental Model

```text
User Request
     │
     ▼
┌─────────────────────┐
│   LLM / LLMEngine   │  ← Orchestration
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Scheduler + Sequence│  ← Who runs this step?
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│    BlockManager     │  ← Which physical KV blocks?
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│    ModelRunner      │  ← Prepare tensors + CUDA Graph
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Qwen3 + Attention  │  ← Actual compute + store KV
└─────────────────────┘
```

---

## 📚 Recommended Reading Order

### 1️⃣ Entry Point & High-level API

| File | Purpose |
|------|---------|
| [`example.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/example.py) | How a user actually calls the library |
| [`nanovllm/llm.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/llm.py) | Thin public wrapper around the engine |
| [`nanovllm/config.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/config.py) | All configuration knobs (`kvcache_block_size`, `max_num_seqs`…) |
| [`nanovllm/sampling_params.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/sampling_params.py) | Temperature, max tokens, etc. |

### 2️⃣ Core Data Structure

| File | Purpose |
|------|---------|
| [`nanovllm/engine/sequence.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/sequence.py) | One request = prompt + generated tokens + `block_table` |

### 3️⃣ Memory Management (PagedAttention) ⭐

| File | Purpose |
|------|---------|
| [`nanovllm/engine/block_manager.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/block_manager.py) | **Most important file**. Allocates / frees / shares KV blocks + prefix caching |

### 4️⃣ Scheduling (Continuous Batching)

| File | Purpose |
|------|---------|
| [`nanovllm/engine/scheduler.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/scheduler.py) | Prefill vs Decode decisions + preemption |

### 5️⃣ Orchestration

| File | Purpose |
|------|---------|
| [`nanovllm/engine/llm_engine.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/llm_engine.py) | The main loop: `add_request → schedule → run → postprocess` |

### 6️⃣ Execution Layer

| File | Purpose |
|------|---------|
| [`nanovllm/engine/model_runner.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/model_runner.py) | Tensor preparation, CUDA Graphs, Tensor Parallelism |

### 7️⃣ Model & Layers

| File | Purpose |
|------|---------|
| [`nanovllm/models/qwen3.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/models/qwen3.py) | The transformer model |
| [`nanovllm/layers/attention.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/layers/attention.py) | FlashAttention + KV cache store/load |
| [`nanovllm/layers/linear.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/layers/linear.py) | Tensor-parallel linear layers |
| [`nanovllm/layers/rotary_embedding.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/layers/rotary_embedding.py) | RoPE |
| [`nanovllm/layers/layernorm.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/layers/layernorm.py) | RMSNorm |
| [`nanovllm/layers/sampler.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/layers/sampler.py) | Token sampling |
| Other layers | Activation & embedding helpers |

### 8️⃣ Utilities

| File | Purpose |
|------|---------|
| [`nanovllm/utils/context.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/utils/context.py) | Global context (`slot_mapping`, `block_tables`…) |
| [`nanovllm/utils/loader.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/utils/loader.py) | Weight loading |

---

## 💡 Pro Tips

- After reading **`sequence.py` + `block_manager.py` + `scheduler.py`** you already understand ~70% of what makes vLLM special.
- Read with the mental model above open — it helps a lot.
- Prefer reading the files in the exact order listed. Jumping around makes the design harder to see.

---

## 🎯 Learning Goals

By the end of this reading path you will understand:

- [x] How PagedAttention works in practice  
- [x] How Continuous Batching is scheduled  
- [x] How prefix caching shares KV blocks  
- [x] How CUDA Graphs accelerate decode  
- [x] How Tensor Parallelism is implemented cleanly  

---

<p align="center">
  <b>Happy reading!</b><br>
  Made with ❤️ for people who want to truly understand LLM inference engines.
</p>
```

The file has been created at:

**`/home/workdir/artifacts/NANO_VLLM_READING_GUIDE.md`**

You can download it from there. Would you like me to also make any small tweaks (different title, more sections, badges, etc.)?  
```render_file```file_path is /home/workdir/artifacts/NANO_VLLM_READING_GUIDE.md鈥