"""
BERT WordPiece tokenizer (uncased), pure Python.

Reproduces the Hugging Face `tokenizers` pipeline that ships with
`nomic-embed-text-v1.5` (`tokenizer.json`): BertNormalizer (clean text,
pad CJK ideographs, strip accents, lowercase), BertPreTokenizer (split on
whitespace and punctuation), WordPiece (greedy longest match, `##`
continuation, `[UNK]` for words over 100 characters), then
`[CLS] ... [SEP]` with truncation.

Kept in the repo instead of depending on `tokenizers` so the embedding
pipeline has no hidden version drift and no extra transitive packages.
Exactness is proved by `tests/unit/test_wordpiece.py` against token IDs
produced by `tokenizers` from the same pinned `tokenizer.json`.
"""

from __future__ import annotations

import string
import unicodedata
from pathlib import Path

_ASCII_PUNCT = frozenset(string.punctuation)
_CONTROL_CATEGORIES = frozenset({"Cc", "Cf", "Co"})

# Added tokens of the pinned tokenizer.json (`normalized: false`): matched in
# the raw text before normalization, exactly like `tokenizers` does.
SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")

# CJK Unified Ideographs blocks, as in BERT's `_is_chinese_char`.
_CJK_RANGES = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0xF900, 0xFAFF),
    (0x2F800, 0x2FA1F),
)


def _is_cjk(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _is_whitespace(ch: str) -> bool:
    return ch in " \t\n\r" or ch.isspace()


def _is_control(ch: str) -> bool:
    # `tokenizers` drops control (Cc), format (Cf) and private-use (Co) only;
    # unassigned code points (Cn) stay in the word (and make it `[UNK]`).
    if ch in "\t\n\r":
        return False
    return unicodedata.category(ch) in _CONTROL_CATEGORIES


def _is_punctuation(ch: str) -> bool:
    return ch in _ASCII_PUNCT or unicodedata.category(ch).startswith("P")


class WordPieceTokenizer:
    """Uncased BERT WordPiece tokenizer loaded from a `vocab.txt` file."""

    def __init__(
        self,
        vocab: dict[str, int],
        unk_token: str = "[UNK]",  # noqa: S107 (tokenizer token, not a secret)
        cls_token: str = "[CLS]",  # noqa: S107
        sep_token: str = "[SEP]",  # noqa: S107
        max_input_chars_per_word: int = 100,
    ):
        self.vocab = vocab
        self.unk_id = vocab[unk_token]
        self.cls_id = vocab[cls_token]
        self.sep_id = vocab[sep_token]
        self.max_input_chars_per_word = max_input_chars_per_word

    @classmethod
    def from_file(cls, path: str | Path) -> WordPieceTokenizer:
        vocab: dict[str, int] = {}
        with open(path, encoding="utf-8") as fh:
            for index, line in enumerate(fh):
                token = line.rstrip("\n")
                if token not in vocab:
                    vocab[token] = index
        return cls(vocab)

    # --- normalizer -------------------------------------------------------

    @staticmethod
    def normalize(text: str) -> str:
        out: list[str] = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or _is_control(ch):
                continue
            if _is_whitespace(ch):
                out.append(" ")
            elif _is_cjk(cp):
                out.append(f" {ch} ")
            else:
                out.append(ch)
        text = unicodedata.normalize("NFD", "".join(out))
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        return text.lower()

    # --- pre-tokenizer ----------------------------------------------------

    @staticmethod
    def pre_tokenize(text: str) -> list[str]:
        words: list[str] = []
        current: list[str] = []
        for ch in text:
            if _is_whitespace(ch):
                if current:
                    words.append("".join(current))
                    current = []
            elif _is_punctuation(ch):
                if current:
                    words.append("".join(current))
                    current = []
                words.append(ch)
            else:
                current.append(ch)
        if current:
            words.append("".join(current))
        return words

    # --- model ------------------------------------------------------------

    def _word_to_ids(self, word: str) -> list[int]:
        if len(word) > self.max_input_chars_per_word:
            return [self.unk_id]
        ids: list[int] = []
        start = 0
        while start < len(word):
            end = len(word)
            found = None
            while start < end:
                piece = word[start:end]
                if start > 0:
                    piece = "##" + piece
                found = self.vocab.get(piece)
                if found is not None:
                    break
                end -= 1
            if found is None:
                return [self.unk_id]
            ids.append(found)
            start = end
        return ids

    def _split_special(self, text: str) -> list[tuple[str, int | None]]:
        """Split raw text on special tokens (matched exactly, before normalizing)."""
        parts: list[tuple[str, int | None]] = []
        i = start = 0
        while i < len(text):
            if text[i] == "[":
                match = next((t for t in SPECIAL_TOKENS if text.startswith(t, i)), None)
                if match is not None and match in self.vocab:
                    if start < i:
                        parts.append((text[start:i], None))
                    parts.append((match, self.vocab[match]))
                    i += len(match)
                    start = i
                    continue
            i += 1
        if start < len(text):
            parts.append((text[start:], None))
        return parts

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        """Token IDs with `[CLS]`/`[SEP]`, truncated to `max_length` in total."""
        ids: list[int] = []
        for segment, special_id in self._split_special(text):
            if special_id is not None:
                ids.append(special_id)
                continue
            for word in self.pre_tokenize(self.normalize(segment)):
                ids.extend(self._word_to_ids(word))
        if max_length is not None:
            ids = ids[: max(0, max_length - 2)]
        return [self.cls_id, *ids, self.sep_id]
