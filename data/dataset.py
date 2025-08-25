from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets, DownloadConfig
from transformers import AutoTokenizer, AutoProcessor, DataCollatorForSeq2Seq
import random
import torch
import os

base_path = os.path.dirname(os.path.abspath(__file__))
qwen_path = os.path.join(base_path, "..", "Qwen3-0.6B")
clip_path = os.path.join(base_path, "..", "clip-vit-base-patch16")


class VLMDataset(Dataset):
    def __init__(
        self, qwen_path, clip_path, config=None, split_type="train", val_split_ratio=0.1
    ):
        super().__init__()
        print(f"加载 lmms-lab/COCO-Caption2017 并准备 '{split_type}' 数据集...")
        coco_dataset = load_dataset(
            "lmms-lab/COCO-Caption2017",
            split="val",
            cache_dir=base_path,
            download_config=DownloadConfig(resume_download=True),
        )

        print("加载 Flickr30k 数据集...")
        flickr_dataset = load_dataset(
            "lmms-lab/flickr30k",
            split="test",
            cache_dir=base_path,
            download_config=DownloadConfig(resume_download=True),
        )

        def standardize_coco(dataset):
            return {'captions': dataset['answer']}
        
        def standardize_flickr(dataset):
            return {'captions': dataset['caption']}
        
        coco_standarized=coco_dataset.map(standardize_coco,remove_columns=['question','answer'])
        flickr_standardized = flickr_dataset.map(standardize_flickr, remove_columns=['caption'])
        
        merged_dataset=concatenate_datasets([coco_standarized,flickr_standardized])
        
        # Split the dataset into training and validation sets
        split_dataset = merged_dataset.train_test_split(
            test_size=val_split_ratio, seed=42
        )

        if split_type == "train":
            self.dataset = split_dataset["train"]
            print(f"训练集大小: {len(self.dataset)}")
        else:
            self.dataset = split_dataset["test"]
            print(f"验证集大小: {len(self.dataset)}")

        self.tokenizer = AutoTokenizer.from_pretrained(qwen_path)
        self.processor = AutoProcessor.from_pretrained(clip_path, use_fast=True)
        self.image_pad_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.config = config

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]

        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")

        image_processed = self.processor(images=image, return_tensors="pt")
        pixel_values = image_processed["pixel_values"].squeeze(0)

        captions_list = item["captions"]
        selected_caption = None

        if isinstance(captions_list, list) and captions_list:
            valid_captions = [
                c for c in captions_list if isinstance(c, str) and c.strip()
            ]
            if valid_captions:
                selected_caption = random.choice(valid_captions)

        if not selected_caption:
            selected_caption = "No valid caption for this image."

        user_prompt = "	Please carefully observe the image and come up with a caption for the image."
        if "<image>" not in user_prompt:
            user_prompt += "\n<image>"

        chat_history = [
            {
                "role": "user",
                "content": user_prompt.replace(
                    "<image>", "<|image_pad|>" * self.config.image_pad_num
                ),
            },
            {"role": "assistant", "content": selected_caption},
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=False,
        )

        tokenizer_output = self.tokenizer(
            formatted_prompt,
            truncation=True,
            max_length=1024,
        )

        input_ids = tokenizer_output["input_ids"]
        attention_mask = tokenizer_output["attention_mask"]

        labels = [-100] * len(input_ids)

        assistant_turn_idx = -1
        for i, turn in enumerate(chat_history):
            if turn["role"] == "assistant":
                assistant_turn_idx = i
                break

        if assistant_turn_idx != -1:
            prompt_history = chat_history[:assistant_turn_idx]
            formatted_prompt_only = self.tokenizer.apply_chat_template(
                prompt_history,
                tokenize=False,
                add_generation_prompt=True,
            )

            prompt_ids = self.tokenizer(
                formatted_prompt_only,
            )["input_ids"]

            prompt_len = len(prompt_ids)
            labels[prompt_len:] = input_ids[prompt_len:]
            for i in range(len(input_ids)):
                if attention_mask[i] == 0:
                    labels[i] = -100
                if input_ids[i] == self.image_pad_token_id:
                    labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }


class VLMDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        pixel_values = torch.stack(
            [feature.pop("pixel_values") for feature in features]
        )

        batch = super().__call__(features, return_tensors)

        batch["pixel_values"] = pixel_values
        return batch

if __name__=="__main__":
    dataset=VLMDataset(qwen_path,clip_path)
    print(random.choice(dataset))