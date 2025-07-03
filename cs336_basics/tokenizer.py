import os
from typing import BinaryIO
import multiprocessing
import collections
import regex as re

END_OF_TEXT_STR = "<|endoftext|>"
PRETOKEN_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# TODO: iterate over file and regex OTF


class Tokenizer:
    @classmethod
    def tokenize(
        cls, input_path: str, vocab_size: int, special_tokens: list[str]
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        tokens = {b: bytes([b]) for b in range(256)}
        for token in special_tokens:
            tokens[len(tokens)] = token.encode("utf-8")

        special_pattern = "|".join(map(re.escape, special_tokens))
        master_pattern = re.compile(f"({special_pattern})|({PRETOKEN_PAT})")

        pretoken_cts = pretokenize(input_path, master_pattern).items()
        bp_cts = collections.defaultdict(int)

        merges = []
        while len(tokens) < vocab_size:
            for pt, ct in pretoken_cts:
                for bp in get_byte_pairs(pt):
                    bp_cts[bp] += ct

            best_pair = sorted(bp_cts.items(), key=lambda x: (x[1], x[0]), reverse=True)[0][0]
            new_token_bytes = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            tokens[len(tokens)] = new_token_bytes

            bp_cts = collections.defaultdict(int)
            pretoken_cts = [(merge_pair(pt, best_pair, new_token_bytes), ct) for pt, ct in pretoken_cts]
        return tokens, merges


def merge_pair(sequence: tuple[bytes, ...], pair_to_merge: tuple[bytes, bytes], new_token: bytes) -> tuple[bytes, ...]:
    new_sequence = []
    i = 0
    while i < len(sequence):
        if i < len(sequence) - 1 and (sequence[i], sequence[i + 1]) == pair_to_merge:
            new_sequence.append(new_token)
            i += 2
        else:
            new_sequence.append(sequence[i])
            i += 1
    return tuple(new_sequence)


def pretokenize(input_path: str, pattern: str):
    pretoken_cts = collections.defaultdict(int)
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, multiprocessing.cpu_count(), END_OF_TEXT_STR.encode("utf-8"))

    num_processes = len(boundaries) - 1

    with multiprocessing.Pool(processes=num_processes) as pool:
        args = [(input_path, boundaries[i], boundaries[i + 1] - boundaries[i], pattern) for i in range(num_processes)]
        cts = pool.starmap(pretokenize_chunk, args)

    pretoken_cts = collections.defaultdict(int)

    for d in cts:
        for pretoken, ct in d.items():
            pretoken_cts[pretoken] += ct

    return pretoken_cts


def pretokenize_chunk(input_path: str, offset: int, size: int, pattern: str):
    with open(input_path, "r") as f:
        f.seek(offset)
        s = f.read(max(0, size - len(END_OF_TEXT_STR)))

    pretoken_cts = collections.defaultdict(int)

    for match in re.finditer(pattern, s):
        pretoken_cts[tuple(bytes([b]) for b in match.group().encode("utf-8"))] += 1

    return pretoken_cts


def get_byte_pairs(b: tuple[bytes]) -> list[tuple[bytes, bytes]]:
    return [(b[i], b[i + 1]) for i in range(len(b) - 1)]


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

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
