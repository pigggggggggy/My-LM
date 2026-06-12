from __future__ import annotations

from collections.abc import Iterable, Iterator
import pickle
import json
import ast
import regex as re


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges = list(merges)

        self.bytes_to_id: dict[bytes, int] = {
            token_bytes: token_id for token_id, token_bytes in self.vocab.items()
        }

        self.special_tokens = special_tokens or []

        next_id = max(self.vocab.keys(), default=-1) + 1

        # 如果 special token 不在 vocab 里，就追加进去
        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")

            if token_bytes not in self.bytes_to_id:
                self.vocab[next_id] = token_bytes
                self.bytes_to_id[token_bytes] = next_id
                next_id += 1

        self.special_token_to_id = {
            special_token: self.bytes_to_id[special_token.encode("utf-8")]
            for special_token in self.special_tokens
        }

        # pair -> merge priority
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }

        # 长的 special token 优先匹配，避免短 token 抢先匹配
        self.special_tokens_sorted = sorted(
            self.special_tokens,
            key=len,
            reverse=True,
        )

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        vocab = cls._load_vocab(vocab_filepath)
        merges = cls._load_merges(merges_filepath)
        return cls(vocab, merges, special_tokens)

    @staticmethod
    def _load_vocab(path: str) -> dict[int, bytes]:
        # 推荐你自己训练完之后用 pickle 保存 vocab
        with open(path, "rb") as f:
            data = f.read()

        try:
            vocab = pickle.loads(data)
            return vocab
        except Exception:
            pass

        # 兼容 JSON: {"0": [116], "1": [104], ...}
        obj = json.loads(data.decode("utf-8"))

        vocab: dict[int, bytes] = {}
        for key, value in obj.items():
            token_id = int(key)

            if isinstance(value, list):
                token_bytes = bytes(value)
            elif isinstance(value, str):
                token_bytes = value.encode("latin-1")
            else:
                raise TypeError(f"Unsupported vocab value type: {type(value)}")

            vocab[token_id] = token_bytes

        return vocab

    @staticmethod
    def _load_merges(path: str) -> list[tuple[bytes, bytes]]:
        # 推荐你自己训练完之后用 pickle 保存 merges
        with open(path, "rb") as f:
            data = f.read()

        try:
            merges = pickle.loads(data)
            return merges
        except Exception:
            pass

        # 兼容 JSON: [[[116], [104]], [[116, 104], [101]], ...]
        try:
            obj = json.loads(data.decode("utf-8"))

            merges: list[tuple[bytes, bytes]] = []
            for left, right in obj:
                left_b = bytes(left) if isinstance(left, list) else left.encode("latin-1")
                right_b = bytes(right) if isinstance(right, list) else right.encode("latin-1")
                merges.append((left_b, right_b))

            return merges
        except Exception:
            pass

        # 兼容每行一个 Python repr: (b't', b'h')
        merges = []
        for line in data.decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            left, right = ast.literal_eval(line)
            merges.append((left, right))

        return merges

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []

        for piece in self._split_by_special_tokens(text):
            if piece == "":
                continue

            if piece in self.special_token_to_id:
                ids.append(self.special_token_to_id[piece])
                continue

            for match in re.finditer(PAT, piece):
                pretoken = match.group(0)
                token_bytes = pretoken.encode("utf-8")
                ids.extend(self._encode_pretoken(token_bytes))

        return ids

    def _split_by_special_tokens(self, text: str) -> list[str]:
        if not self.special_tokens_sorted:
            return [text]

        pattern = "(" + "|".join(
            re.escape(token) for token in self.special_tokens_sorted
        ) + ")"

        return re.split(pattern, text)

    def _encode_pretoken(self, token_bytes: bytes) -> list[int]:
        # 初始状态：每个 UTF-8 byte 是一个 token
        parts: tuple[bytes, ...] = tuple(bytes([b]) for b in token_bytes)

        while len(parts) >= 2:
            best_pair = None
            best_rank = float("inf")

            # 找当前 pre-token 中 rank 最靠前的 merge pair
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                rank = self.merge_ranks.get(pair)

                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            # 把 best_pair 的所有非重叠出现都合并
            merged_parts: list[bytes] = []
            i = 0

            while i < len(parts):
                if (
                    i < len(parts) - 1
                    and parts[i] == best_pair[0]
                    and parts[i + 1] == best_pair[1]
                ):
                    merged_parts.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    merged_parts.append(parts[i])
                    i += 1

            parts = tuple(merged_parts)

        return [self.bytes_to_id[token] for token in parts]

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        懒加载编码，用于大文件。

        核心思路：
        - 不直接对每个 chunk 单独 encode，因为 token 可能跨 chunk 边界。
        - 每次保留 buffer 末尾那个可能还没结束的 pre-token。
        - 只 encode 安全的前缀。
        """
        buffer = ""

        for chunk in iterable:
            if chunk == "":
                continue

            buffer += chunk

            safe_end = self._safe_prefix_end(buffer)

            if safe_end > 0:
                prefix = buffer[:safe_end]

                for token_id in self.encode(prefix):
                    yield token_id

                buffer = buffer[safe_end:]

        if buffer:
            for token_id in self.encode(buffer):
                yield token_id

    def _safe_prefix_end(self, text: str) -> int:
        """
        返回 text 中可以安全 encode 的前缀结束位置。

        末尾最后一个 regex pre-token 可能会被下一个 chunk 扩展，
        所以要保留它。
        """
        matches = list(re.finditer(PAT, text))

        if not matches:
            return 0

        safe_end = len(text)

        last_match = matches[-1]

        if last_match.end() == len(text):
            safe_end = min(safe_end, last_match.start())

        # 还要防止 special token 被切断
        special_suffix_start = self._special_suffix_start(text)
        safe_end = min(safe_end, special_suffix_start)

        return safe_end

    def _special_suffix_start(self, text: str) -> int:
        """
        如果 text 的末尾是某个 special token 的前缀，
        则返回这个前缀的开始位置，避免提前 encode。
        """
        best_keep = 0

        for token in self.special_tokens_sorted:
            max_len = min(len(token), len(text))

            for length in range(max_len, 0, -1):
                if text.endswith(token[:length]):
                    best_keep = max(best_keep, length)
                    break

        if best_keep == 0:
            return len(text)

        return len(text) - best_keep