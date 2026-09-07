import logging
from collections.abc import Iterator
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from xlm.data.processors import get_image_string

logger = logging.getLogger(__name__)


class DatasetBase(Dataset[Any]):
    def __init__(
        self,
        dataset: Any,
        tokenizer: Any,
        image_processor: Any,
        mp_image_token_length: int,
        relevance_min_rating: int = 1,
        image_correspondence_min_rating: int = 1,
        visual_dependency_min_rating: int = 1,
        formatting_min_rating: int = 1,
    ) -> None:
        # Any required: `dataset` is a HuggingFace datasets object (no stubs),
        # `tokenizer` is a transformers tokenizer with runtime-injected custom
        # special-token attributes, and `image_processor` is a torchvision transform.
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.relevance_min_rating = relevance_min_rating
        self.image_correspondence_min_rating = image_correspondence_min_rating
        self.visual_dependency_min_rating = visual_dependency_min_rating
        self.formatting_min_rating = formatting_min_rating
        self.prefix_len = self._get_prefix_len()

    def __len__(self) -> int:
        return len(self.dataset)

    def _get_prefix_len(self) -> int:
        random_string_5_letters = "xzyvd"
        random_string_chat_templated = self.tokenizer.apply_chat_template(
            [
                {
                    "role": "assistant",
                    "content": random_string_5_letters,
                }
            ],
            tokenize=False,
            add_special_tokens=False,
        )
        random_string_location = random_string_chat_templated.find(
            random_string_5_letters
        )
        return len(
            self.tokenizer.encode(random_string_chat_templated[:random_string_location])
        )

    def _get_messages(
        self, item: dict[str, Any], splitted_image_counts: list[tuple[int, int]]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for index, text in enumerate(item["texts"]):
            try:
                if (
                    item.get("relevance_ratings") is not None
                    and item["relevance_ratings"][index] is not None
                    and item["relevance_ratings"][index] < self.relevance_min_rating
                ):
                    continue
                if (
                    item.get("image_correspondence_ratings") is not None
                    and item["image_correspondence_ratings"][index] is not None
                    and item["image_correspondence_ratings"][index]
                    < self.image_correspondence_min_rating
                ):
                    continue
                if (
                    item.get("visual_dependency_ratings") is not None
                    and item["visual_dependency_ratings"][index] is not None
                    and item["visual_dependency_ratings"][index]
                    < self.visual_dependency_min_rating
                ):
                    continue
                if (
                    item.get("formatting_ratings") is not None
                    and item["formatting_ratings"][index] is not None
                    and item["formatting_ratings"][index] < self.formatting_min_rating
                ):
                    continue
            except (KeyError, TypeError, IndexError) as e:
                logger.warning(
                    "Error processing item: %s, index: %s: %s", item, index, e
                )

            messages.append({"role": "user", "content": text["user"]})
            messages.append({"role": "assistant", "content": text["assistant"]})

        if len(messages) == 0:
            return messages

        # Safety check to ensure no image tokens are persent in the text before adding them.
        for msg in messages:
            if self.tokenizer.image_token in msg["content"]:
                logger.warning(
                    "Found and removed an image token in the %s text before adding the image string.",
                    msg["role"],
                )
                msg["content"] = msg["content"].replace(self.tokenizer.image_token, "")

        if len(splitted_image_counts) > 0:
            image_string = get_image_string(
                self.tokenizer, splitted_image_counts, self.mp_image_token_length
            )
            messages[0]["content"] = image_string + messages[0]["content"]

        return messages

    def _process_images(
        self, images: list[Image.Image]
    ) -> tuple[list[torch.Tensor], list[tuple[int, int]]]:
        processed_images: list[torch.Tensor] = []
        splitted_image_counts: list[tuple[int, int]] = []
        for image in images:
            if isinstance(image, Image.Image):
                if image.mode != "RGB":
                    image = image.convert("RGB")
                processed_image, splitted_image_count = self.image_processor(image)
                if (
                    not hasattr(self.tokenizer, "global_image_token")
                    and splitted_image_count[0] * splitted_image_count[1]
                    == len(processed_image) - 1
                ):
                    # If the tokenizer doesn't have a global image token, but the processor generated it, remove it
                    processed_image = processed_image[1:]
                processed_images.append(processed_image)
                splitted_image_counts.append(splitted_image_count)
            else:
                raise TypeError(f"Error processing image: {image}")
        return processed_images, splitted_image_counts

    def _prepare_inputs_and_loss_mask(
        self, messages: list[dict[str, str]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        conv_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            return_dict=True,
        )
        total_tokens = len(conv_ids["input_ids"])
        mask = [0] * total_tokens

        # Locate each assistant turn and flip its mask to 1.
        # Use incremental prefix encoding to get accurate per-message token offsets
        # in the full conversation. Encoding [msg] in isolation can produce a
        # different segment length (e.g. an extra BOS token) than the same
        # message embedded in the full chat, which caused the cursor to drift
        # and the loss-mask to miss every assistant token (=> all labels -100 => nan loss).
        offsets: list[int] = [0]
        for i in range(1, len(messages) + 1):
            partial_ids = self.tokenizer.apply_chat_template(
                messages[:i], tokenize=True, add_special_tokens=False
            )
            offsets.append(len(partial_ids))

        # Safety: if for some reason incremental encoding doesn't match full
        # encoding (should not happen), fall back to the full length.
        if offsets[-1] != total_tokens:
            offsets[-1] = total_tokens

        for i, msg in enumerate(messages):
            seg_start = offsets[i]
            seg_end = offsets[i + 1]

            if msg["role"] == "assistant":
                # prefix_len skips the assistant-role template prefix so the
                # model is only trained on the actual assistant reply text.
                start = min(seg_start + self.prefix_len, total_tokens)
                end = min(seg_end, total_tokens)
                if start < end:
                    mask[start:end] = [1] * (end - start)

        return (
            torch.tensor(conv_ids["input_ids"]),
            torch.tensor(mask).to(torch.bool),
            torch.tensor(conv_ids["attention_mask"]),
        )


class VQADataset(DatasetBase):  # Visual Question Answering Dataset
    def iter_for_worker(self) -> Iterator[dict[str, Any] | None]:
        # with iterable datasets, each worker gets different shards
        for data in self.dataset:
            yield self._process_data(data)

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        item = self.dataset[idx]
        return self._process_data(item)

    def _process_data(self, item: dict[str, Any]) -> dict[str, Any] | None:
        # Handle images (should be a list)
        if item["images"] is None:
            images_data = []
        else:
            images_data = item["images"]
            if not isinstance(images_data, list):
                images_data = [images_data]

        processed_images: list[torch.Tensor] = []
        splitted_image_counts: list[tuple[int, int]] = []
        if images_data:  # Only process if there are images
            processed_images, splitted_image_counts = self._process_images(images_data)

        messages = self._get_messages(item, splitted_image_counts)

        if len(messages) == 0:
            return None

        input_ids, mask, attention_mask = self._prepare_inputs_and_loss_mask(messages)
        labels = self._get_labels(input_ids, mask)

        return {
            "images": processed_images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _get_labels(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        labels = input_ids.clone().masked_fill(~mask, -100)
        labels = labels.roll(-1)  # Shift labels for causal LM
        labels[-1] = -100  # Last token has no target

        return labels


class CollatorBase:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

        self.data_field: dict[str, list[Any]] = {
            "input_ids": [],
            "labels": [],
            "attention_mask": [],
            "images": [],
        }

    def _pad_batch(self, batch: dict[str, list[Any]], max_length: int) -> None:
        batch["input_ids"] = [
            torch.nn.functional.pad(
                ids, (max_length - len(ids), 0), value=self.tokenizer.pad_token_id
            )
            for ids in batch["input_ids"]
        ]
        batch["labels"] = [
            torch.nn.functional.pad(
                labels, (max_length - len(labels), 0), value=self.tokenizer.pad_token_id
            )
            for labels in batch["labels"]
        ]
        batch["attention_mask"] = [
            torch.nn.functional.pad(
                attention_mask, (max_length - len(attention_mask), 0), value=0
            )
            for attention_mask in batch["attention_mask"]
        ]

    def prepare_batch(
        self, batch: list[dict[str, Any] | None], max_lenght: int | None = None
    ) -> dict[str, Any]:
        # 1. Hadndle empty
        if not batch:
            return self.data_field

        # 2. Drop None rows
        samples = [s for s in batch if s is not None]
        if not samples:
            return self.data_field

        # 3. batch is a list of dicts, each containing 'input_ids', 'attention_mask', 'labels', 'images'
        # let's convert it to a dict of lists of tensors
        batched: dict[str, list[Any]] = {
            k: [item[k] for item in samples] for k in samples[0]
        }

        if max_lenght is not None:
            batched = self._discard_samples_that_are_too_long(batched, max_lenght)

        if len(batched["input_ids"]) == 0:
            return batched

        # 4. Pad samples to max_length
        if max_lenght is not None:
            max_len = max_lenght
        else:
            max_len = max(map(len, batched["input_ids"]))

        self._pad_batch(batched, max_len)

        return {
            "input_ids": torch.stack(batched["input_ids"]),
            "attention_mask": torch.stack(batched["attention_mask"]),
            "images": batched["images"],
            "labels": torch.stack(batched["labels"]),
        }

    def _discard_samples_that_are_too_long(
        self, batch: dict[str, list[Any]], max_length: int
    ) -> dict[str, list[Any]]:
        filtered = [
            (ids, label, attn_mask, image)
            for ids, label, attn_mask, image in zip(
                batch["input_ids"],
                batch["labels"],
                batch["attention_mask"],
                batch["images"],
            )
            if len(ids) <= max_length
        ]

        if not filtered:
            return self.data_field

        batch_token_ids, batch_labels, batch_attention_mask, batch_images = zip(
            *filtered
        )

        return {
            "input_ids": list(batch_token_ids),
            "labels": list(batch_labels),
            "attention_mask": list(batch_attention_mask),
            "images": list(batch_images),
        }


class VQACollator(CollatorBase):
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.max_length = max_length
        super().__init__(tokenizer)

    def _pad_batch(self, batch: dict[str, list[Any]], max_length: int) -> None:
        # 重新改写，将标签的填充值设为 -100，这样损失函数会自动忽略该值。
        batch["input_ids"] = [
            torch.nn.functional.pad(
                ids, (max_length - len(ids), 0), value=self.tokenizer.pad_token_id
            )
            for ids in batch["input_ids"]
        ]
        batch["labels"] = [
            torch.nn.functional.pad(labels, (max_length - len(labels), 0), value=-100)
            for labels in batch["labels"]
        ]
        batch["attention_mask"] = [
            torch.nn.functional.pad(
                attention_mask, (max_length - len(attention_mask), 0), value=0
            )
            for attention_mask in batch["attention_mask"]
        ]

    def __call__(self, batch: list[dict[str, Any] | None]) -> dict[str, Any]:
        prepared = self.prepare_batch(batch, max_lenght=self.max_length)
        return prepared