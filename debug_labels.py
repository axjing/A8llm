import sys
sys.path.insert(0, '.')

import torch
from xlm.data.processors import get_tokenizer
from xlm.data.vqa_datasets import DatasetBase
from xlm.models.config import VLMConfig


def debug_mask_issue():
    vlm_cfg = VLMConfig()
    tokenizer = get_tokenizer(
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        vlm_cfg.vlm_extra_tokens,
        vlm_cfg.lm_chat_template,
    )

    class MockDataset:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {
                "images": None,
                "texts": [
                    {
                        "user": "Hello, what is in this picture?",
                        "assistant": "The picture shows a beautiful cat sitting on a windowsill.",
                    }
                ],
            }

    class TestDataset(DatasetBase):
        def test_prepare(self):
            splitted_image_counts = []
            messages = self._get_messages(self.dataset[0], splitted_image_counts)
            print("=== Messages ===")
            for m in messages:
                print(f"  role={m['role']}: {m['content'][:60]}...")

            input_ids, mask, attn_mask = self._prepare_inputs_and_loss_mask(messages)
            mask_sum = mask.sum().item()
            total_tokens = len(input_ids)
            print(f"\n=== _prepare_inputs_and_loss_mask result ===")
            print(f"  Total tokens: {total_tokens}")
            print(f"  Masked (assistant) tokens to train on: {mask_sum}")
            print(f"  Ratio: {mask_sum/total_tokens:.2%}")

            labels = self._get_labels(input_ids, mask)
            valid_labels = (labels != -100).sum().item()
            print(f"  Valid labels (!= -100): {valid_labels}")

            if mask_sum == 0:
                print("\n❌ FAIL: mask_sum == 0 → loss WILL be nan")
                return False
            else:
                # Decode the assistant tokens we're training on
                valid_positions = mask.bool()
                train_tokens = input_ids[valid_positions]
                decoded = tokenizer.decode(train_tokens)
                print(f"\n  Decoded training content (assistant reply, minus template prefix):")
                print(f"  '{decoded.strip()}'")
                print("\n✅ PASS: mask correctly identifies assistant tokens, loss will be finite.")
                return True

    test_ds = TestDataset.__new__(TestDataset)
    test_ds.dataset = MockDataset()
    test_ds.tokenizer = tokenizer
    test_ds.mp_image_token_length = 16
    test_ds.relevance_min_rating = 1
    test_ds.image_correspondence_min_rating = 1
    test_ds.visual_dependency_min_rating = 1
    test_ds.formatting_min_rating = 1
    test_ds.prefix_len = test_ds._get_prefix_len()
    print(f"prefix_len (assistant-template prefix token count) = {test_ds.prefix_len}")
    return test_ds.test_prepare()


if __name__ == "__main__":
    ok = debug_mask_issue()
    sys.exit(0 if ok else 1)