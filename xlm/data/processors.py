from typing import Any

from torchvision import transforms
from transformers import AutoTokenizer

from xlm.data.custom_transforms import DynamicResize, GlobalAndSplitImages

TOKENIZERS_CACHE: dict[str, Any] = {}


def get_tokenizer(
    name: str,
    extra_special_tokens: list[str] | dict[str, str] | None = None,
    chat_template: str | None = None,
) -> Any:
    # Any required: transformers AutoTokenizer is a dynamic factory whose concrete
    # type is not statically known and injects custom special-token attributes
    # (image_token_id, r1c1, ...) at runtime.
    if name not in TOKENIZERS_CACHE:
        tokenizer_init_kwargs: dict[str, Any] = {"use_fast": True}
        if extra_special_tokens is not None:
            tokenizer_init_kwargs["extra_special_tokens"] = extra_special_tokens
        if chat_template is not None:
            tokenizer_init_kwargs["chat_template"] = chat_template
        tokenizer = AutoTokenizer.from_pretrained(
            name,
            **tokenizer_init_kwargs,
        )
        tokenizer.pad_token = tokenizer.eos_token
        TOKENIZERS_CACHE[name] = tokenizer
    return TOKENIZERS_CACHE[name]


def get_image_processor(
    max_img_size: int,
    splitted_image_size: int,
    resize_to_max_side_len: bool = False,
) -> Any:
    # Any required: torchvision ships no type stubs in this environment; returns
    # a transforms.Compose of DynamicResize -> ToTensor -> GlobalAndSplitImages.
    return transforms.Compose(
        [
            DynamicResize(splitted_image_size, max_img_size, resize_to_max_side_len),
            transforms.ToTensor(),
            GlobalAndSplitImages(splitted_image_size),
        ]
    )


def get_image_string(
    tokenizer: Any,
    splitted_image_counts: list[tuple[int, int]],
    mp_image_token_length: int,
) -> str:
    image_string = ""
    # splitted_image_counts is a list of tuples (n_h, n_w)
    for idx, (n_h, n_w) in enumerate(splitted_image_counts):
        if len(splitted_image_counts) > 1:
            image_string += f"<image: {idx}>"
        if hasattr(tokenizer, "global_image_token"):
            image_string += tokenizer.global_image_token
            image_string += tokenizer.image_token * mp_image_token_length
            if (
                n_h == 1 and n_w == 1
            ):  # If there is only one patch, treat it as the global image
                continue
        for i in range(n_h):
            for j in range(n_w):
                image_string += getattr(tokenizer, f"r{i + 1}c{j + 1}")
                image_string += tokenizer.image_token * mp_image_token_length
    return image_string
