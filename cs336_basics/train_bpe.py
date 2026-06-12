from cs336_basics.pretokenization_example import find_chunk_boundaries
import os
from collections import Counter,defaultdict
import multiprocessing

def run_train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    num_processes = os.cpu_count() or 1

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            num_processes,
            special_tokens[0].encode("utf-8"),
        )

    tasks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tasks.append((input_path, start, end, special_tokens))

    with multiprocessing.Pool(processes=num_processes) as pool:
        counters = pool.map(_pretokenize_chunk, tasks)

    pretoken_counts = Counter()
    for c in counters:
        pretoken_counts.update(c)


    return train_bpe_fast_from_pretoken_counts(
        pretoken_counts=pretoken_counts,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def get_pretoken_counts(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    pretoken_counts = Counter()

    # 1. 先按 special tokens 切开，避免 <|endoftext|> 参与 BPE merge
    if special_tokens:
        special_tokens = sorted(special_tokens, key=len, reverse=True)
        pattern = "|".join(re.escape(tok) for tok in special_tokens)
        documents = re.split(pattern, text)
    else:
        documents = [text]

    # 2. 对每个 document 单独做 pre-tokenization
    for doc in documents:
        for match in re.finditer(PAT, doc):
            pretoken = match.group(0)

            # 3. 把 pre-token 转成 byte tuple
            byte_tuple = tuple(bytes([b]) for b in pretoken.encode("utf-8"))

            # 4. 统计频率
            pretoken_counts[byte_tuple] += 1

    return pretoken_counts

from collections import Counter

def get_pair_counts(pretoken_counts) -> Counter[tuple[bytes, bytes]]:
    pair_counts = Counter()

    for token_tuple, count in pretoken_counts.items():
        for i in range(len(token_tuple) - 1):
            pair = (token_tuple[i], token_tuple[i + 1])
            pair_counts[pair] += count

    return pair_counts

def get_best_pair(
    pair_counts: Counter[tuple[bytes, bytes]]
) -> tuple[bytes, bytes] | None:
    if not pair_counts:
        return None

    best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]

    return best_pair

def merge_pair_in_token(token_tuple, pair_to_merge: tuple[bytes, bytes],):
    new_token=pair_to_merge[0] + pair_to_merge[1]
    result=[]
    i=0
    while i <len(token_tuple):
        if(
            i<len(token_tuple)-1
            and token_tuple[i]==pair_to_merge[0]
            and token_tuple[i+1]==pair_to_merge[1]
        ):
            result.append(new_token)
            i+=2
        else:
            result.append(token_tuple[i])
            i+=1
    return tuple(result)

def merge_pair_in_pretokens(
    pretoken_counts:Counter[tuple[bytes, ...]],
    pair_to_merge:tuple[bytes,bytes],
)->Counter[tuple[bytes,...]]:
    result=Counter()
    for token_tuple,count in pretoken_counts.items():
        new_token_tuple=merge_pair_in_token(token_tuple,pair_to_merge)
        result[new_token_tuple]+=count
    return result

def main():
    # 1. 构造一个 toy corpus
    toy_text = """
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
hello<|endoftext|>world
"""

    toy_path = "toy_bpe_test.txt"

    # 2. 写入测试文件
    with open(toy_path, "w", encoding="utf-8") as f:
        f.write(toy_text)

    # 3. 训练 BPE
    vocab, merges = run_train_bpe(
        input_path=toy_path,
        vocab_size=270,
        special_tokens=["<|endoftext|>"],
    )
    
def train_bpe_fast_from_pretoken_counts(
    pretoken_counts: Counter[tuple[bytes, ...]],
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 1. 初始化 vocab
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    merges: list[tuple[bytes, bytes]] = []

    # 2. 把 Counter 转成 unique pre-token list + frequency list
    pretokens: list[list[bytes]] = []
    freqs: list[int] = []

    for token_tuple, count in pretoken_counts.items():
        pretokens.append(list(token_tuple))
        freqs.append(count)

    # 3. 初始化 pair_counts 和 pair_to_indices
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_indices: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    for idx, token in enumerate(pretokens):
        freq = freqs[idx]

        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            pair_counts[pair] += freq
            pair_to_indices[pair].add(idx)

    # 4. BPE merge loop
    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        # 注意：保持和测试一致的 tie-breaking
        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]

        merges.append(best_pair)

        a, b = best_pair
        new_token = a + b
        vocab[len(vocab)] = new_token

        # 只更新包含 best_pair 的 pre-token
        affected_indices = list(pair_to_indices.get(best_pair, set()))

        for idx in affected_indices:
            token = pretokens[idx]
            freq = freqs[idx]

            # 保险：如果这个 token 里已经没有 best_pair，跳过
            has_best_pair = False
            for i in range(len(token) - 1):
                if token[i] == a and token[i + 1] == b:
                    has_best_pair = True
                    break

            if not has_best_pair:
                continue

            # 4.1 删除这个 token 原来的 pair 贡献
            for i in range(len(token) - 1):
                old_pair = (token[i], token[i + 1])

                pair_counts[old_pair] -= freq
                if pair_counts[old_pair] <= 0:
                    del pair_counts[old_pair]

                if old_pair in pair_to_indices:
                    pair_to_indices[old_pair].discard(idx)
                    if not pair_to_indices[old_pair]:
                        del pair_to_indices[old_pair]

            # 4.2 对这个 token 应用 merge
            merged_token: list[bytes] = []
            i = 0

            while i < len(token):
                if (
                    i < len(token) - 1
                    and token[i] == a
                    and token[i + 1] == b
                ):
                    merged_token.append(new_token)
                    i += 2
                else:
                    merged_token.append(token[i])
                    i += 1

            pretokens[idx] = merged_token

            # 4.3 加回 merge 之后的新 pair 贡献
            token = pretokens[idx]

            for i in range(len(token) - 1):
                new_pair = (token[i], token[i + 1])

                pair_counts[new_pair] += freq
                pair_to_indices[new_pair].add(idx)

    return vocab, merges

def _pretokenize_chunk(args):
    input_path, start, end, special_tokens = args

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)

    text = chunk_bytes.decode("utf-8", errors="ignore")
    return get_pretoken_counts(text, special_tokens)


if __name__ == "__main__":
    main()