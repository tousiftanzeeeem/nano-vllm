"""Tokenizer resolution for the visualizer.

Prefers the model's real tokenizer. Falls back to a word-piece-ish splitter so
the token tape still shows recognizable text when `transformers` is absent.
"""

import re


class FallbackTokenizer:
    """Deterministic regex tokenizer with a vocabulary grown on demand.

    Not a real BPE — it exists so the paging visualization has honest token
    boundaries to lay into blocks when the real tokenizer cannot be loaded.
    """

    PATTERN = re.compile(r"\s*\w+|\s*[^\w\s]|\s+")

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.inv: dict[int, str] = {}
        self.eos_token_id = 0
        self._add("<|eos|>")

    def _add(self, piece: str) -> int:
        if piece not in self.vocab:
            tid = len(self.vocab)
            self.vocab[piece] = tid
            self.inv[tid] = piece
        return self.vocab[piece]

    def encode(self, text: str) -> list[int]:
        pieces = [p for p in self.PATTERN.findall(text) if p]
        # long words split further, mimicking sub-word behaviour
        out = []
        for p in pieces:
            stripped = p.strip()
            if len(stripped) > 7:
                head = p[: len(p) - len(stripped) + 4]
                out.append(self._add(head))
                for i in range(4, len(stripped), 4):
                    out.append(self._add(stripped[i : i + 4]))
            else:
                out.append(self._add(p))
        return out

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.inv.get(t, "") for t in token_ids)

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [self.inv.get(t, "?") for t in token_ids]


def load_tokenizer(model_path: str | None):
    """Return (tokenizer, name). Falls back when transformers/model is missing."""
    if model_path:
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(model_path, use_fast=True), "real"
        except Exception:
            pass
    return FallbackTokenizer(), "fallback"


def piece_text(tokenizer, token_id: int) -> str:
    """Human-readable text for one token, with whitespace made visible.

    Memoized per tokenizer: a real tokenizer's decode() is slow enough that
    tracing a few hundred steps would otherwise spend seconds here.
    """
    cache = getattr(tokenizer, "_viz_piece_cache", None)
    if cache is None:
        cache = {}
        try:
            tokenizer._viz_piece_cache = cache
        except AttributeError:
            cache = None
    if cache is not None and token_id in cache:
        return cache[token_id]
    try:
        if isinstance(tokenizer, FallbackTokenizer):
            raw = tokenizer.inv.get(token_id, "?")
        else:
            raw = tokenizer.decode([token_id])
    except Exception:
        raw = "?"
    out = "·" if raw == "" else raw.replace("\n", "↵").replace(" ", "·")
    if cache is not None:
        cache[token_id] = out
    return out
