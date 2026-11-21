"""WordPiece tokenizer parity with Hugging Face `tokenizers` (#134).

The fixture holds token IDs produced by `tokenizers==0.23.3` from the pinned
`tokenizer.json` of nomic-embed-text-v1.5 (revision e9b6763) for 322 inputs:
mixed case, ALL CAPS (including Greek final sigma), accents, CJK, emoji,
control and private-use characters, special-token literals, a 3000-word text
and truncation. Any difference fails: the token IDs feed every stored vector.

`known_differences` lists rare code points where `tokenizers` (own, older
Unicode tables) and Python's `unicodedata` disagree. They are documented,
not gated; the Unicode version is stored in the collection registry.

`vocab.txt` (Apache-2.0, nomic-ai/nomic-embed-text-v1.5) is committed so this
test runs in CI without downloading the model.
"""

import hashlib
import json
from pathlib import Path

import pytest

from src.services.embedding.model_store import VOCAB_SHA256
from src.services.embedding.wordpiece import WordPieceTokenizer

FIXTURES = Path(__file__).parent.parent / "fixtures" / "embedding"


@pytest.fixture(scope="module")
def golden():
    return json.loads((FIXTURES / "wordpiece_golden.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tokenizer():
    return WordPieceTokenizer.from_file(FIXTURES / "vocab.txt")


def test_fixture_vocab_is_the_pinned_vocab(golden):
    actual = hashlib.sha256((FIXTURES / "vocab.txt").read_bytes()).hexdigest()
    assert actual == VOCAB_SHA256
    assert golden["vocab_sha256"] == VOCAB_SHA256


def test_token_ids_match_reference_exactly(golden, tokenizer):
    mismatches = [
        (case["text"][:60], case.get("max_length"))
        for case in golden["cases"]
        if tokenizer.encode(case["text"], case.get("max_length")) != case["ids"]
    ]
    assert len(golden["cases"]) >= 300
    assert mismatches == []


def test_uncased_input_gives_same_ids(tokenizer):
    # The model is uncased: case must not change the tokens (the #134 bug).
    assert tokenizer.encode("REPLACE SUMP PUMP") == tokenizer.encode(
        "replace sump pump"
    )
    assert tokenizer.encode("REPLACE SUMP PUMP") != tokenizer.encode(
        "CREATE A SPREADSHEET"
    )


def test_final_sigma_lowercased_per_character(tokenizer):
    # str.lower() would turn a word-final Σ into ς; the reference gives σ.
    assert tokenizer.normalize("ΟΔΟΣ") == "οδοσ"
    assert tokenizer.encode("ΣΊΣΥΦΟΣ") == tokenizer.encode("σισυφοσ")


def test_known_unicode_differences_are_documented(golden):
    known = golden["known_differences"]
    assert len(known) >= 5
    texts = {c["text"] for c in golden["cases"]}
    for item in known:
        assert item["codepoint"].startswith("U+") and item["reference_ids"]
        assert item["text"] not in texts  # never part of the gated set


def test_truncation_keeps_special_tokens(tokenizer):
    ids = tokenizer.encode("word " * 50, max_length=8)
    assert len(ids) == 8
    assert ids[0] == tokenizer.cls_id and ids[-1] == tokenizer.sep_id


def test_small_vocab_wordpiece_rules():
    vocab = {
        t: i
        for i, t in enumerate(
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "un", "##aff", "##able", "!"]
        )
    }
    tok = WordPieceTokenizer(vocab)
    assert tok.encode("Unaffable!") == [2, 4, 5, 6, 7, 3]
    assert tok.encode("unknown") == [2, 1, 3]  # no full match -> one [UNK]
