"""Arabic / English language detection, splitting, and chunking."""
from __future__ import annotations

import re

ARABIC_RE = re.compile(
    r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]"
)


def arabic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    arabic = sum(1 for c in letters if ARABIC_RE.match(c))
    return arabic / len(letters)


def is_arabic_line(line: str, threshold: float = 0.3) -> bool:
    return arabic_ratio(line) >= threshold


def split_by_language(text: str) -> tuple[str, str]:
    """Classify each non-empty line as Arabic or English and return
    (english_text, arabic_text)."""
    english: list[str] = []
    arabic: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        (arabic if is_arabic_line(line) else english).append(line)
    return "\n".join(english), "\n".join(arabic)


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?۔])\s+")


def chunk_text(
    text: str, target_chars: int = 600, overlap: int = 80
) -> list[str]:
    """Chunk into ~target_chars pieces. Split on paragraph boundaries, fall
    back to sentences for oversized paragraphs, and carry a small tail of
    each chunk into the next so cross-boundary context is preserved."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    def flush():
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    for para in paragraphs:
        if len(para) > target_chars:
            flush()
            buffer = ""
            for sent in _SENTENCE_SPLIT.split(para):
                sent = sent.strip()
                if not sent:
                    continue
                if len(buffer) + len(sent) + 1 <= target_chars:
                    buffer = f"{buffer} {sent}" if buffer else sent
                else:
                    if buffer:
                        chunks.append(buffer)
                    buffer = sent
            if buffer:
                chunks.append(buffer)
        elif len(current) + len(para) + 1 <= target_chars:
            current = f"{current}\n{para}" if current else para
        else:
            flush()
            current = para
    flush()

    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-overlap:]
        out.append(f"{tail}\n{chunks[i]}")
    return out
