import os
from typing import BinaryIO
import multiprocessing
import collections
import regex as re

END_OF_TEXT_STR = "<|endoftext|>"
PRETOKEN_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# TODO: iterate over file and regex OTF


def tokenize(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    tokens = {b: bytes(b) for b in range(255)}
    for token in special_tokens:
        tokens[token.encode("utf-8")] = len(tokens)

    pretoken_cts = pretokenize(input_path)

    while len(tokens) < vocab_size:
        bp_cts = collections.defaultdict(int)
        for pretoken, ct in pretoken_cts.items():
            for bp in get_byte_pairs(pretoken):
                bp_cts[bp] += ct

    # TODO:


def encode(t: tuple[bytes], token: bytes) -> tuple[bytes]:
    n = len(t)
    l = r = 0
    cur = b""
    res = b""
    while r < min(len(token) - 1, n):
        cur += t[r]
        r += 1
    while r < n:
        cur += t[r]
        r += 1
        if cur == token:
            res += token
            cur = b""
        else:
            res += cur[0]
            cur = cur[1:]
    res += cur
    return tuple(b for b in res)


def pretokenize(input_path: str):
    pretoken_cts = collections.defaultdict(int)
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            multiprocessing.cpu_count(),
            END_OF_TEXT_STR.encode("utf-8"),
        )

    num_processes = len(boundaries) - 1

    with multiprocessing.Pool(processes=num_processes) as pool:
        args = [(input_path, boundaries[i], boundaries[i + 1] - boundaries[i])
                for i in range(num_processes)]
        cts = pool.starmap(pretokenize_chunk, args)

    pretoken_cts = collections.defaultdict(int)

    for d in cts:
        for pretoken, ct in d.items():
            pretoken_cts[pretoken] += ct

    return pretoken_cts


def pretokenize_chunk(input_path: str, offset: int, size: int):
    with open(input_path, "r") as f:
        f.seek(offset)
        pref = f.read(len(END_OF_TEXT_STR))
        if pref != END_OF_TEXT_STR:
            s = pref
        else:
            s = ""
        s += f.read(max(0, size - len(END_OF_TEXT_STR)))

    pretoken_cts = collections.defaultdict(int)

    for match in re.finditer(PRETOKEN_PAT, s):
        pretoken_cts[tuple(b for b in match.group().encode("utf-8"))] += 1

    return pretoken_cts


def get_byte_pairs(b: bytes) -> list[tuple[bytes, bytes, int]]:
    return [(b[i], b[i + 1]) for i in range(len(b) - 1)]


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token,
                      bytes), "Must represent special token as a bytestring"

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
