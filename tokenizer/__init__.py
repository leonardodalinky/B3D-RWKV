"""RWKV-world tokenizer (v20230424).

Default usage:

    from tokenizer import RWKVTokenizer
    tok = RWKVTokenizer()             # auto-loads bundled vocab, uses trie (fast)
    ids = tok.encode("hello world")   # -> list[int]
    text = tok.decode(ids)            # -> str

The two underlying classes are also exposed for advanced use:

    from tokenizer import TRIE_TOKENIZER, RWKV_TOKENIZER
"""
from pathlib import Path

from .rwkv_tokenizer import RWKV_TOKENIZER, TRIE_TOKENIZER

_DEFAULT_VOCAB = Path(__file__).parent / "rwkv_vocab_v20230424.txt"


def RWKVTokenizer(vocab_path: str | Path = _DEFAULT_VOCAB, fast: bool = True):
    """Build the RWKV-world tokenizer with the bundled vocab.

    Args:
        vocab_path: vocab .txt file. Defaults to the bundled ``rwkv_vocab_v20230424.txt``.
        fast: if True (default) returns the trie-based ``TRIE_TOKENIZER``.
              Set False to get the slower naive ``RWKV_TOKENIZER`` reference.
    """
    cls = TRIE_TOKENIZER if fast else RWKV_TOKENIZER
    return cls(str(vocab_path))


__all__ = ["RWKVTokenizer", "RWKV_TOKENIZER", "TRIE_TOKENIZER"]
