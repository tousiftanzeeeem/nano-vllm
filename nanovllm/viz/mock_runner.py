"""A ModelRunner stand-in that needs no GPU, no weights and no attention kernel.

Paging and scheduling are decided entirely by Scheduler + BlockManager; the
model only turns tokens into tokens. Swapping it out lets the whole
visualization run on any machine — which is the point of the tool.
"""

import time


CANNED = (
    "Paged attention stores the KV cache in fixed-size blocks instead of one "
    "contiguous buffer per request. A block table maps each logical block of a "
    "sequence to a physical block in the pool, exactly like a page table maps "
    "virtual pages to physical frames. Because blocks are uniform, any free "
    "block fits any request, so the pool never fragments and sequences sharing "
    "a prefix can point at the same physical block with a reference count."
)


class MockModelRunner:
    """Returns tokens from a canned answer so the token tape reads sensibly."""

    def __init__(self, tokenizer, eos: int, step_delay: float = 0.0):
        self.tokenizer = tokenizer
        self.eos = eos
        self.step_delay = step_delay
        self.cursor: dict[int, int] = {}
        self.script = tokenizer.encode(CANNED)

    def call(self, method: str, *args):
        if method == "run":
            return self.run(*args)
        return None

    def run(self, seqs, is_prefill: bool) -> list[int]:
        if self.step_delay:
            time.sleep(self.step_delay)
        out = []
        for seq in seqs:
            i = self.cursor.get(seq.seq_id, 0)
            if i >= len(self.script):
                out.append(self.eos)
            else:
                out.append(self.script[i])
                self.cursor[seq.seq_id] = i + 1
        return out

    def exit(self):
        pass
