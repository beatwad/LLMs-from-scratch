"""My implementation of BPE tokenizer + GPT-4 compatable BPE tokenizer"""

from typing import List, Tuple

import regex as re
import tiktoken
import yaml
from tqdm.auto import tqdm

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
MERGES_PATH = "data/merges.yaml"
VOCAB_PATH = "data/vocab.yaml"


class RegexTokenizer:
    """Basic tokenizer which uses GPT4 string split pattern"""

    def __init__(self):
        self._load_merges_vocab(MERGES_PATH, VOCAB_PATH)
        self._pat = re.compile(GPT4_SPLIT_PATTERN)

    def train(self, text: str, vocab_size: int, verbose=False) -> None:
        tokens = []
        words = self._prepocess(text)
        for word in words:
            tokens.extend(list(word.encode("utf-8")))
        num_merges = vocab_size - 256
        if verbose:
            iter = tqdm(range(num_merges))
        else:
            iter = range(num_merges)
        for i in iter:
            stats = self._get_stats(tokens)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            tokens = self._merge(tokens, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(
                    f"{pair} -> {idx} ({self.vocab[idx].decode('utf-8', errors='replace')} has {stats[pair]} of occurencies)"
                )
        self._save_merges_vocab(MERGES_PATH, VOCAB_PATH)

    def encode(self, text: str) -> List[int]:
        tokens = []
        words = self._prepocess(text)
        for word in words:
            text_bytes = list(word.encode("utf-8"))
            tokens.extend(self._encode_chunk(text_bytes))
        return tokens

    def decode(self, tokens: List[int]) -> str:
        tokens = b"".join(self.vocab[t] for t in tokens)
        text = tokens.decode("utf-8", errors="replace")
        return text

    def _prepocess(self, text: str) -> List[str]:
        words = self._pat.findall(text)
        return words

    def _encode_chunk(self, text_bytes: List[str]) -> List[int]:
        while len(text_bytes) >= 2:
            stats = self._get_stats(text_bytes)
            # find the pair in merges with the minimal order number
            # so order of merges creation is preserved
            pair = min(stats, key=lambda x: self.merges.get(x, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            text_bytes = self._merge(text_bytes, pair, idx)
        return text_bytes

    def _get_stats(self, tokens: List[int]) -> List[int]:
        stats = {}
        for pair in zip(tokens, tokens[1:]):
            stats[pair] = stats.get(pair, 0) + 1
        return stats

    def _merge(self, ids: List[int], pair: Tuple[int], idx: int) -> List[int]:
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def _save_merges_vocab(self, merges_path: str, vocab_path: str) -> None:
        with open(merges_path, "w") as f:
            inverse_merges = {v: k for k, v in self.merges.items()}
            yaml.safe_dump(inverse_merges, f)
        with open(vocab_path, "w") as f:
            yaml.safe_dump(self.vocab, f)

    def _load_merges_vocab(self, merges_path: str, vocab_path: str) -> None:
        try:
            with open(merges_path, "r") as f:
                inverse_merges = yaml.safe_load(f)
                self.merges = {tuple(v): k for k, v in inverse_merges.items()}
        except FileNotFoundError:
            self.merges = {}
        try:
            with open(vocab_path, "r") as f:
                self.vocab = yaml.safe_load(f)
        except FileNotFoundError:
            self.vocab = {idx: bytes([idx]) for idx in range(256)}


class GPT4Tokenizer(RegexTokenizer):
    """Reproduction of GPT-4 tokenizer"""

    def __init__(self):
        enc = tiktoken.get_encoding("cl100k_base")
        mergeable_ranks = enc._mergeable_ranks
        self.merges = self.recover_merges(mergeable_ranks)
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for pair, idx in self.merges.items():
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
        # OpenAI for unknown reason shuffle bytes, so these two dicts are used for forward and inverse shuffling
        self.byte_shuffle = {i: enc._mergeable_ranks[bytes([i])] for i in range(256)}
        self.inverse_byte_shuffle = {enc._mergeable_ranks[bytes([i])]: i for i in range(256)}
        self._pat = re.compile(GPT4_SPLIT_PATTERN)

    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError

    def encode(self, text: str) -> List[int]:
        tokens = []
        words = self._prepocess(text)
        for word in words:
            text_bytes = list(word.encode("utf-8"))
            text_bytes = bytes(self.byte_shuffle[b] for b in text_bytes)
            tokens.extend(self._encode_chunk(text_bytes))
        return tokens

    def decode(self, tokens: List[int]) -> str:
        tokens = b"".join(self.vocab[t] for t in tokens)
        tokens = bytes([self.inverse_byte_shuffle[t] for t in tokens])
        text = tokens.decode("utf-8", errors="replace")
        return text

    @staticmethod
    def bpe(mergeable_ranks, token, max_rank):
        """Helper function used in get_gpt4_merges() to reconstruct the merges forest"""
        parts = [bytes([b]) for b in token]
        while True:
            min_idx = None
            min_rank = None
            for i, pair in enumerate(zip(parts, parts[1:])):
                rank = mergeable_ranks.get(pair[0] + pair[1])
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_idx = i
                    min_rank = rank
            if min_rank is None or (max_rank is not None and min_rank >= max_rank):
                break
            assert min_idx is not None
            parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]
        return parts

    def recover_merges(self, mergeable_ranks):
        """
        The `merges` are already the byte sequences in their merged state.
        so we have to recover the original pairings. We can do this by doing
        a small BPE training run on all the tokens, in their order.
        also see https://github.com/openai/tiktoken/issues/60
        also see https://github.com/karpathy/minbpe/issues/11#issuecomment-1950805306
        """
        merges = {}
        for token, rank in mergeable_ranks.items():
            if len(token) == 1:
                continue  # skip raw bytes
            pair = tuple(self.bpe(mergeable_ranks, token, max_rank=rank))
            assert len(pair) == 2
            # recover the integer ranks of the pair
            ix0 = mergeable_ranks[pair[0]]
            ix1 = mergeable_ranks[pair[1]]
            merges[(ix0, ix1)] = rank
        return merges


if __name__ == "__main__":
    vocab_size = 500
    with open("taylorswift.txt") as f:
        text = f.read()

    regex_tokenizer = RegexTokenizer()
    enc = tiktoken.get_encoding("cl100k_base")  # this is the GPT-4 tokenizer
    gpt4_tokenizer = GPT4Tokenizer()

    if not regex_tokenizer.merges:
        regex_tokenizer.train(text, vocab_size, verbose=False)

    text = "hello world!!!? (안녕하세요!) lol123 😉"
    ids1 = regex_tokenizer.encode(text)
    assert text == regex_tokenizer.decode(ids1)

    ids2 = enc.encode(text)
    ids3 = gpt4_tokenizer.encode(text)

    assert ids2 == ids3
    assert text == gpt4_tokenizer.decode(ids2)
