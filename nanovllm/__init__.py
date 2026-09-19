from nanovllm.sampling_params import SamplingParams

__all__ = ["LLM", "SamplingParams"]


def __getattr__(name):
    # LLM pulls in torch/transformers; import it on use so the visualizer's
    # mock engine stays runnable without the full GPU stack.
    if name == "LLM":
        from nanovllm.llm import LLM

        return LLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
