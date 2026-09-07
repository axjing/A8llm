import copy
import os
from collections.abc import Iterable
from typing import Any, cast

import torch
from tokenizers import Regex, decoders, pre_tokenizers
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>",  # user messages
    "<|user_end|>",
    "<|assistant_start|>",  # assistant messages
    "<|assistant_end|>",
    "<|python_start|>",  # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>",  # python REPL outputs back to assistant
    "<|output_end|>",
]


# 注意：此分割模式与GPT-4不同，我们使用\p{N}{1,2}而非\p{N}{1,3}
# 我这样做是因为对于较小的词汇量，不想在数字上“浪费”太多标记。
# 我已证实，对于32K的词汇量，2是最佳选择。1稍差一些，3则更差。

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class HF2Tokenizer:
    def __init__(self, tokenizer: HFTokenizer) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path: str) -> "HF2Tokenizer":
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir: str) -> "HF2Tokenizer":
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(
        cls, text_iterator: Iterable[str], vocab_size: int
    ) -> "HF2Tokenizer":
        # 从文本迭代器进行训练 ，配置HuggingFace分词器
        tokenizer = HFTokenizer(BPE(byte_fallback=True, unk_token="", fuse_unk=False))
        # Normalizer: None
        tokenizer.normalizer = None
        # 预分词器：GPT-4风格
        # GPT-4用于在BPE之前将文本分割成组的正则表达式模式
        # 注意：该模式已从\p{N}{1,3}改为\p{N}{1,2}，因为我怀疑这对
        # 非常小的模型和更小的词汇量有害，因为这在标记空间上有点浪费。
        # TODO 需要进一步有验证这一点
        gpt4_split_regex = Regex(
            SPLIT_PATTERN
        )  # huggingface demands that you wrap it in Regex!!
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    pattern=gpt4_split_regex, behavior="isolated", invert=False
                ),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        # Decoder: ByteLevel (it pairs together with the ByteLevel pre-tokenizer)
        tokenizer.decoder = decoders.ByteLevel()
        # Post-processor: None
        tokenizer.post_processor = None
        # Trainer: BPE
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0,  # no minimum frequency
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        # Kick off the training
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab(self) -> dict[str, int]:
        return cast(dict[str, int], self.tokenizer.get_vocab())

    def get_special_tokens(self) -> list[str]:
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values() if w.special]
        return special_tokens

    def id_to_token(self, id: int) -> str | None:
        return cast(str | None, self.tokenizer.id_to_token(id))

    def _encode_one(
        self,
        text: str,
        prepend: str | int | None = None,
        append: str | int | None = None,
        num_threads: int | None = None,
    ) -> list[int]:
        # 对单个字符串进行编码
        # 前缀/后缀既可以是特殊标记的字符串，也可以直接是标记ID。
        # 线程数会被忽略（仅由nanochat分词器用于并行编码）
        assert isinstance(text, str)
        ids: list[int] = []
        if prepend is not None:
            prepend_id = (
                prepend if isinstance(prepend, int) else self.encode_special(prepend)
            )
            assert prepend_id is not None, f"Unknown special token: {prepend}"
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = (
                append if isinstance(append, int) else self.encode_special(append)
            )
            assert append_id is not None, f"Unknown special token: {append}"
            ids.append(append_id)
        return ids

    def encode_special(self, text: str) -> int | None:
        # encode a single special token via exact match
        return cast(int | None, self.tokenizer.token_to_id(text))

    def get_bos_token_id(self) -> int:
        # Different HuggingFace models use different BOS tokens and there is little consistency
        # 1) attempt to find a <|bos|> token
        bos = self.encode_special("<|bos|>")
        # 2) if that fails, attempt to find a <|endoftext|> token (e.g. GPT-2 models)
        if bos is None:
            bos = self.encode_special("<|endoftext|>")
        # 3) if these fail, it's better to crash than to silently return None
        assert bos is not None, "Failed to find BOS token in tokenizer"
        return bos

    def encode(
        self,
        text: str | list[str],
        prepend: str | int | None = None,
        append: str | int | None = None,
        num_threads: int | None = None,
    ) -> list[int] | list[list[int]]:
        if isinstance(text, str):
            return self._encode_one(
                text, prepend=prepend, append=append, num_threads=num_threads
            )
        else:
            return [
                self._encode_one(
                    t, prepend=prepend, append=append, num_threads=num_threads
                )
                for t in text
            ]

    def __call__(
        self, text: str | list[str], **kwargs: Any
    ) -> list[int] | list[list[int]]:
        # **kwargs forwarded to encode(); Any required for dynamic passthrough
        return self.encode(text, **kwargs)

    def decode(self, ids: list[int]) -> str:
        return cast(str, self.tokenizer.decode(ids, skip_special_tokens=False))

    def save(self, tokenizer_dir: str) -> None:
        # save the tokenizer to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")


# -----------------------------------------------------------------------------
# Tokenizer based on rustbpe + tiktoken combo
import pickle

import rustbpe
import tiktoken


class RustBPETokenizer:
    """Light wrapper around tiktoken (for efficient inference) but train with rustbpe"""

    def __init__(self, enc: tiktoken.Encoding, bos_token: str) -> None:
        self.enc = enc
        self._special_cache: dict[str, int] = {}
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(
        cls, text_iterator: Iterable[str], vocab_size: int
    ) -> "RustBPETokenizer":
        # 1) train using rustbpe
        tokenizer = rustbpe.Tokenizer()
        # the special tokens are inserted later in __init__, we don't train them here
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, (
            f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        )
        tokenizer.train_from_iterator(
            text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN
        )
        # 2) construct the associated tiktoken encoding for inference
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {
            name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)
        }
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,  # dict[bytes, int] (token bytes -> merge priority rank)
            special_tokens=special_tokens,  # dict[str, int] (special token name -> token id)
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir: str) -> "RustBPETokenizer":
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name: str) -> "RustBPETokenizer":
        # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktoken calls the special document delimiter token "<|endoftext|>"
        # yes this is confusing because this token is almost always PREPENDED to the beginning of the document
        # it most often is used to signal the start of a new sequence to the LLM during inference etc.
        # so in nanoChat we always use "<|bos|>" short for "beginning of sequence", but historically it is often called "<|endoftext|>".
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self) -> int:
        return self.enc.n_vocab

    def get_special_tokens(self) -> set[str]:
        return self.enc.special_tokens_set

    def id_to_token(self, id: int) -> str:
        return self.enc.decode([id])

    def encode_special(self, text: str) -> int:
        if text not in self._special_cache:
            self._special_cache[text] = self.enc.encode_single_token(text)
        return self._special_cache[text]

    def get_bos_token_id(self) -> int:
        return self.bos_token_id

    def encode(
        self,
        text: str | list[str],
        prepend: str | int | None = None,
        append: str | int | None = None,
        num_threads: int = 8,
    ) -> list[int] | list[list[int]]:
        # text can be either a string or a list of strings
        ids: list[int] | list[list[int]]

        if prepend is not None:
            prepend_id = (
                prepend if isinstance(prepend, int) else self.encode_special(prepend)
            )
        if append is not None:
            append_id = (
                append if isinstance(append, int) else self.encode_special(append)
            )

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)  # TODO: slightly inefficient here? :( hmm
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    assert isinstance(ids_row, list)
                    ids_row.insert(0, prepend_id)  # TODO: same
            if append is not None:
                for ids_row in ids:
                    assert isinstance(ids_row, list)
                    ids_row.append(append_id)
        else:
            raise TypeError(f"Invalid input type: {type(text)}")

        return ids

    def __call__(self, *args: Any, **kwargs: Any) -> list[int] | list[list[int]]:
        # passthrough to encode(); Any required for dynamic dispatch
        return self.encode(*args, **kwargs)

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)

    def save(self, tokenizer_dir: str) -> None:
        # save the encoding object to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

    def render_conversation(
        self, conversation: dict[str, Any], max_tokens: int = 2048
    ) -> tuple[list[int], list[int]]:
        """
        Tokenize a single Chat conversation (which we call a "doc" or "document" here).
        Returns:
        - ids: list[int] is a list of token ids of this rendered conversation
        - mask: list[int] of same length, mask = 1 for tokens that the Assistant is expected to train on.
        """
        # ids, masks that we will return and a helper function to help build them up.
        ids: list[int] = []
        mask: list[int] = []

        def add_tokens(token_ids: int | list[int], mask_val: int) -> None:
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # sometimes the first message is a system message...
        # => just merge it with the second (user) message
        if conversation["messages"][0]["role"] == "system":
            # some conversation surgery is necessary here for now...
            conversation = copy.deepcopy(conversation)  # avoid mutating the original
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", (
                "System message must be followed by a user message"
            )
            messages[1]["content"] = (
                messages[0]["content"] + "\n\n" + messages[1]["content"]
            )
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        # fetch all the special tokens we need
        bos = self.get_bos_token_id()
        user_start, user_end = (
            self.encode_special("<|user_start|>"),
            self.encode_special("<|user_end|>"),
        )
        assistant_start, assistant_end = (
            self.encode_special("<|assistant_start|>"),
            self.encode_special("<|assistant_end|>"),
        )
        python_start, python_end = (
            self.encode_special("<|python_start|>"),
            self.encode_special("<|python_end|>"),
        )
        output_start, output_end = (
            self.encode_special("<|output_start|>"),
            self.encode_special("<|output_end|>"),
        )

        # now we can tokenize the conversation
        add_tokens(bos, 0)
        for i, message in enumerate(messages):
            # some sanity checking here around assumptions, to prevent footguns
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, (
                f"Message {i} is from {message['role']} but should be from {must_be_from}"
            )

            # content can be either a simple string or a list of parts (e.g. containing tool calls)
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), (
                    "User messages are simply expected to be strings"
                )
                value_ids = cast(list[int], self.encode(content))
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    # simple string => simply add the tokens
                    value_ids = cast(list[int], self.encode(content))
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = cast(list[int], self.encode(part["text"]))
                        if part["type"] == "text":
                            # string part => simply add the tokens
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            # python tool call => add the tokens inside <|python_start|> and <|python_end|>
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # python output => add the tokens inside <|output_start|> and <|output_end|>
                            # none of these tokens are supervised because the tokens come from Python at test time
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        # truncate to max_tokens tokens MAX (helps prevent OOMs)
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def visualize_tokenization(
        self, ids: list[int], mask: list[int], with_token_id: bool = False
    ) -> str:
        """Small helper function useful in debugging: visualize the tokenization of render_conversation"""
        RED = "\033[91m"
        GREEN = "\033[92m"
        RESET = "\033[0m"
        GRAY = "\033[90m"
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return "|".join(tokens)

    def render_for_completion(self, conversation: dict[str, Any]) -> list[int]:
        """
        Used during Reinforcement Learning. In that setting, we want to
        render the conversation priming the Assistant for a completion.
        Unlike the Chat SFT case, we don't need to return the mask.
        """
        # We have some surgery to do: we need to pop the last message (of the Assistant)
        conversation = copy.deepcopy(conversation)  # avoid mutating the original
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", (
            "Last message must be from the Assistant"
        )
        messages.pop()  # remove the last message (of the Assistant) inplace

        # Now tokenize the conversation
        ids, _ = self.render_conversation(conversation)

        # Finally, to prime the Assistant for a completion, append the Assistant start token
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids


# -----------------------------------------------------------------------------
# nanochat-specific convenience functions


def get_tokenizer() -> RustBPETokenizer:
    from xlm.common.file_os import get_base_dir

    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    # return HuggingFaceTokenizer.from_directory(tokenizer_dir)
    return RustBPETokenizer.from_directory(tokenizer_dir)


def get_token_bytes(device: str = "cpu") -> torch.Tensor:
    from xlm.common.file_os import get_base_dir

    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), (
        f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    )
    with open(token_bytes_path, "rb") as f:
        return cast(torch.Tensor, torch.load(f, map_location=device))
