import os
from typing import BinaryIO
import multiprocessing
import collections
import regex as re
import pickle
from typing import Iterable

END_OF_TEXT_STR = "<|endoftext|>"
PRETOKEN_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRETOKEN_RE = re.compile(PRETOKEN_PAT)

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab: dict[int, bytes] = vocab

        self.bytes_to_token: dict[bytes, int] = {}
        for t, b in self.vocab.items():
            self.bytes_to_token[b] = t

        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens = special_tokens if special_tokens is not None else []

        if special_tokens is not None:
            self.special_tokens_re = re.compile(f"({'|'.join(map(re.escape, special_tokens))})")

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if len(self.special_tokens) > 0:
            splits = self.special_tokens_re.splititer(text)
        else:
            splits = [text]
        print(f"Splits: {splits}")

        res = []
        for match in PRETOKEN_RE.finditer(text):
            s = match.group()
            if s in self.special_tokens:
                res.append(self.bytes_to_token[s.encode("utf-8")])
            else:
                res.extend(self._tokenize_pretoken(s))

        return res
    
    def _tokenize_pretoken(self, pretoken: str) -> list[int]:
        print(f"Tokenizing pretoken: {pretoken}")
        current_token = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
        print(f"Current token: {current_token}")
        for m1, m2 in self.merges:
            if (m1, m2) in get_byte_pairs(current_token):
                print(f"Merging {m1} and {m2}")
                current_token = merge_pair(current_token, (m1, m2), m1 + m2)
                print(f"Current token: {current_token}")

        print(f"Final token: {current_token}")
        res = []
        for b in current_token:
            res.append(self.bytes_to_token[b])
        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        pass

    def decode(self, tokens: list[int]) -> str:
        res = []
        for token in tokens:
            res.append(self.vocab[token].decode("utf-8"))
        return "".join(res)



def tokenize(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    tokens = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        tokens[len(tokens)] = token.encode("utf-8")

    pretoken_cts = pretokenize(input_path, special_tokens)
    merges = []

    while len(tokens) < vocab_size:
        bp_cts = collections.defaultdict(int)
        for pt, ct in pretoken_cts.items():
            for i in range(len(pt) - 1):
                bp_cts[(pt[i], pt[i + 1])] += ct

        if not bp_cts:
            break

        best_pair = max(bp_cts.items(), key=lambda x: (x[1], x[0]))[0]

        new_token_bytes = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        tokens[len(tokens)] = new_token_bytes

        next_pretoken_cts = collections.defaultdict(int)

        for pt, ct in pretoken_cts.items():
            new_pt = merge_pair(pt, best_pair, new_token_bytes)
            next_pretoken_cts[new_pt] += ct

        pretoken_cts = next_pretoken_cts

    return tokens, merges


def merge_pair(sequence: tuple[bytes, ...], pair_to_merge: tuple[bytes, bytes], new_token_bytes: bytes) -> tuple[bytes, ...]:
    new_sequence = []
    i = 0
    while i < len(sequence):
        if i < len(sequence) - 1 and (sequence[i], sequence[i + 1]) == pair_to_merge:
            new_sequence.append(new_token_bytes)
            i += 2
        else:
            new_sequence.append(sequence[i])
            i += 1
    return tuple(new_sequence)


def pretokenize(input_path: str, special_tokens: list[str]):
    pretoken_cts = collections.defaultdict(int)
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, multiprocessing.cpu_count(), END_OF_TEXT_STR.encode("utf-8"))

    num_processes = len(boundaries) - 1

    special_tokens_re = re.compile("|".join(map(re.escape, special_tokens)))
    with multiprocessing.Pool(processes=num_processes) as pool:
        args = [
            (input_path, boundaries[i], boundaries[i + 1] - boundaries[i], special_tokens_re)
            for i in range(num_processes)
        ]
        cts = pool.starmap(pretokenize_chunk, args)

    pretoken_cts = collections.defaultdict(int)

    for d in cts:
        for pretoken, ct in d.items():
            pretoken_cts[pretoken] += ct

    return pretoken_cts


def pretokenize_chunk(input_path: str, offset: int, size: int, special_tokens_re: re.Pattern):
    with open(input_path, "rb") as f:
        f.seek(offset)
        pieces = special_tokens_re.split(f.read(size).decode("utf-8", "ignore"))
    pretoken_cts = collections.defaultdict(int)

    for piece in pieces:
        for match in PRETOKEN_RE.finditer(piece):
            s = match.group()
            pretoken_cts[tuple(bytes([b]) for b in s.encode("utf-8"))] += 1

    return pretoken_cts


def get_byte_pairs(b: tuple[bytes]) -> list[tuple[bytes, bytes]]:
    return [(b[i], b[i + 1]) for i in range(len(b) - 1)]


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))

if __name__ == "__main__":
    vocab, merges = tokenize("./data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    
    print(f"Max length token: {max(vocab.values(), key=len).decode('utf-8')}")