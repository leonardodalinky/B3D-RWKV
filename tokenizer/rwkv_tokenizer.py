"""RWKV-world tokenizer (v20230424 vocab).

Lifted from third-party/json2binidx_tool/tools/rwkv_tokenizer.py, which is in turn
based on https://github.com/BlinkDL/ChatRWKV/blob/main/tokenizer/rwkv_tokenizer.py
and https://github.com/TkskKurumi/ChatRWKV-TRIE-Tokenizer.

Two interchangeable implementations:
- ``RWKV_TOKENIZER``: naive table lookup, slow reference implementation.
- ``TRIE_TOKENIZER``:  trie-based, materially faster. Used by the json2binidx data
  preprocessing pipeline. Prefer this one.

Both derive ``vocab_size`` from the vocab file (``max(id) + 1``); for the bundled
``rwkv_vocab_v20230424.txt`` this is **65530** (real ids 1..65529 plus implicit
EOS at id 0). The trailing 6 slots (ids 65530..65535) are unused — embeddings are
padded to 65536 for power-of-two friendliness. In DiffuRWKV we reuse id 65535 as
the diffusion MASK token (see CLAUDE.md).

Note: the upstream tokenizer hardcodes ``vocab_size = 65525``, which is **stale** —
the vocab file has been extended to 65529 entries since then. We derive it from
the data so the value is always correct.
"""


class RWKV_TOKENIZER:
    """Naive table-lookup tokenizer (slow reference)."""

    def __init__(self, file_name):
        self.idx2token = {}
        sorted_tokens = []  # must be already sorted in the vocab file
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for l in lines:
            idx = int(l[: l.index(" ")])
            x = eval(l[l.index(" ") : l.rindex(" ")])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(" ") :])
            sorted_tokens += [x]
            self.idx2token[idx] = x

        self.token2idx = {v: int(k) for k, v in self.idx2token.items()}
        # max id in the vocab file + 1 (account for the implicit EOS at id 0).
        self.vocab_size = max(self.idx2token.keys()) + 1

        # precompute some tables for fast matching
        self.table = [[[] for _ in range(256)] for _ in range(256)]
        self.good = [set() for _ in range(256)]
        self.wlen = [0 for _ in range(256)]

        for i in reversed(range(len(sorted_tokens))):  # reverse order: match longer tokens first
            s = sorted_tokens[i]
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src: bytes):
        src_len = len(src)
        tokens = []
        i = 0
        while i < src_len:
            s = src[i : i + 1]
            if i < src_len - 1:
                s1 = int(src[i + 1])
                s0 = int(src[i])
                if s1 in self.good[s0]:
                    sss = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except StopIteration:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)
        return tokens

    def decodeBytes(self, tokens):
        return b"".join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode("utf-8")

    def token_to_id(self, token):
        return self.token2idx[token]

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.idx2token

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode("utf-8")
            except UnicodeDecodeError:
                pass
            print(f"{repr(s)}{i}", end=" ")
        print()


class TRIE:
    __slots__ = ("ch", "to", "values", "front")

    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None] * 256
        self.values = set()
        self.front = front

    def __repr__(self):
        fr = self
        ret = []
        while fr is not None:
            if fr.ch is not None:
                ret.append(fr.ch)
            fr = fr.front
        return "<TRIE %s %s>" % (ret[::-1], self.values)

    def add(self, key: bytes, idx: int = 0, val=None):
        if idx == len(key):
            if val is None:
                val = key
            self.values.add(val)
            return self
        ch = key[idx]
        if self.to[ch] is None:
            self.to[ch] = TRIE(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx + 1, val=val)

    def find_longest(self, key: bytes, idx: int = 0):
        u = self
        ch = key[idx]
        ret = None
        while u.to[ch] is not None:
            u = u.to[ch]
            idx += 1
            if u.values:
                ret = idx, u, u.values
            if idx == len(key):
                break
            ch = key[idx]
        return ret


class TRIE_TOKENIZER:
    """Trie-based tokenizer; materially faster than ``RWKV_TOKENIZER``. Use this."""

    def __init__(self, file_name):
        self.idx2token = {}
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for l in lines:
            idx = int(l[: l.index(" ")])
            x = eval(l[l.index(" ") : l.rindex(" ")])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(" ") :])
            self.idx2token[idx] = x

        self.token2idx = {v: int(k) for k, v in self.idx2token.items()}
        # max id in the vocab file + 1 (account for the implicit EOS at id 0).
        self.vocab_size = max(self.idx2token.keys()) + 1

        self.root = TRIE()
        for t, i in self.token2idx.items():
            self.root.add(t, val=(t, i))

    def encodeBytes(self, src: bytes):
        idx = 0
        tokens = []
        while idx < len(src):
            _idx = idx
            idx, _, values = self.root.find_longest(src, idx)
            assert idx != _idx
            _, token = next(iter(values))
            tokens.append(token)
        return tokens

    def decodeBytes(self, tokens):
        return b"".join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode("utf-8")

    def token_to_id(self, token):
        return self.token2idx[token]

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.idx2token

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode("utf-8")
            except UnicodeDecodeError:
                pass
            print(f"{repr(s)}{i}", end=" ")
        print()
