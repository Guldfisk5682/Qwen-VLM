import os
from torch import _pin_memory
from transformers import Trainer, TrainingArguments, AutoTokenizer

from data.dataset import COCOCaptionDataset, VLMDataCollator
from model import VLM, VLMConfig

base_path = os.path.dirname(os.path.abspath(__file__))
qwen_path = os.path.join(base_path, "Qwen3-0.6B")
clip_path = os.path.join(base_path, "clip-vit-base-patch16")
output_dir = os.path.join(base_path, "output")

if __name__ == "__main__":
    config = VLMConfig(qwen_path=qwen_path, clip_path=clip_path)
    model = VLM(config).cuda()
    tokenizer = AutoTokenizer.from_pretrained(qwen_path)
    
    # model.qwen.gradient_checkpointing_enable(use_cache=False)
    # model.clip.gradient_checkpointing_enable(use_cache=False)

    print("数据集构建...")
    train_dataset = COCOCaptionDataset(qwen_path, clip_path, config, split_type="train")
    eval_dataset = COCOCaptionDataset(qwen_path, clip_path, config, split_type="val")

    data_collator = VLMDataCollator(
        tokenizer=tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
    )

    print("配置参数...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.03,
        warmup_ratio=0.05,
        optim="adamw_8bit",  # adamw_8bit备用
        lr_scheduler_type="cosine",
        fp16=True,
        logging_steps=40,
        logging_dir=os.path.join(output_dir, "logs"),
        save_strategy="steps",
        eval_strategy="steps",
        save_total_limit=4,
        save_steps=200,
        eval_steps=40,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        dataloader_num_workers=0,
    )

    print("实例化Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("开始训练...")
    #trainer.train(resume_from_checkpoint=True)
    trainer.train()
    trainer.save_model(output_dir)
