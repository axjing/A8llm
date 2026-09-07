import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TrainConfig:
    lr_mp: float = 0.00512
    lr_vision_backbone: float = 5e-5  # 0.0005 #
    lr_language_backbone: float = 5e-5  # 0
    val_size: int = 50000
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    eval_in_epochs: bool = True
    eval_interval: int = 500
    stats_log_interval: int = 100
    max_training_steps: int = 40000
    max_images_per_example: int = 4
    max_images_per_knapsack: int = 18
    # Should be close to n_positions to filter overly long samples
    max_sample_length: int = 4096
    compile: bool = False
    resume_from_vlm_checkpoint: bool = False  # Indicate if the training should be resumed from a checkpoint of the whole VLM or you want to start from scratch
    train_dataset_path: str = "HuggingFaceM4/FineVision"  # '/nasdata/anxiangjing/data/cache_dir/huggingface/full/'# 'HuggingFaceM4/FineVision' # 'HuggingFaceM4/FineVision_concat_shuffled_2' # 'HuggingFaceM4/FineVision'
    train_dataset_name: tuple[str, ...] = (
        "sharegpt4v(coco)",
    )  # ('allava_laion',) # 'allava_laion','allava_vflan', 'cambrian(filtered)_processed', 'LLaVA_Instruct_150K', 'mmevol', 'sharegpt4o', 'sharegpt4v(coco)', 'sharegpt4v(knowledge)', 'sharegpt4v(llava)','sharegpt4v(sam)', vision_flan(filtered)', 'lvis_instruct4v',("default", ) #
    stream_dataset: bool = False
    relevance_min_rating: int = 1
    image_correspondence_min_rating: int = 1
    visual_dependency_min_rating: int = 1
    formatting_min_rating: int = 1
    wandb_entity: str = "HuggingFace"  # Indicate the entity to log to in wandb
    log_wandb: bool = True
    use_lmms_eval: bool = True  # Use lmms-eval for evaluation
    lmms_eval_tasks: str = "mmstar,mmmu_val,ocrbench,textvqa_val,docvqa_val,scienceqa,mme,infovqa_val,chartqa"  # Pass additional task as one string, seperated by commas without spaces (e.g. 'mmstar,mmmu,ocrbench')
    lmms_eval_limit: float | None = None
    lmms_eval_batch_size: int = 2

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        """Create a TrainConfig from a dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, file_path: str) -> "TrainConfig":
        """Load a TrainConfig from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, file_path: str | None = None, indent: int = 4) -> str:
        """
        配置转JSON：可返回字符串 / 直接写入文件
        :param file_path: 保存的文件路径，为None时仅返回字符串
        :param indent: 格式化缩进
        :return: file_path为None 返回JSON字符串；写入文件则返回None
        """

        data_dict = asdict(self)
        json_str = json.dumps(data_dict, ensure_ascii=False, indent=indent)

        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data_dict, f, ensure_ascii=False, indent=indent)
            print(f">>> Json saved: {file_path}")

        return json_str
