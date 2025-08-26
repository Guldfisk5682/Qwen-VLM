import torch
from data.dataset import LoRADataset, VLMDataCollator
from transformers import Trainer, TrainingArguments, AutoTokenizer
from safetensors.torch import load_file as load_safetensors
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import os

from model import VLM, VLMConfig


base_path = os.path.dirname(os.path.abspath(__file__))
state_path = os.path.join(base_path, "output", "final_best_model")
qwen_path = os.path.join(base_path, "Qwen3-0.6B")
clip_path = os.path.join(base_path, "clip-vit-base-patch16")
output_dir = os.path.join(base_path, "output")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    
    print("加载模型")

    config = VLMConfig(qwen_path=qwen_path, clip_path=clip_path)
    model = VLM(config).to(device)

    checkpoint_file = os.path.join(state_path, "model.safetensors")
    if os.path.exists(checkpoint_file):
        print(f"从 {checkpoint_file} 加载权重...")
        state_dict = load_safetensors(checkpoint_file)
    else:
        raise FileNotFoundError(f"在 {state_path} 中找不到权重文件!")

    model.load_state_dict(state_dict, strict=False)


    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model.qwen = get_peft_model(model.qwen, lora_config)

    # params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Trainable params: {params/1e6}M")
    
    TRAINING_TYPE = torch.bfloat16
    config = VLMConfig(
        qwen_path=qwen_path, clip_path=clip_path, torch_dtype=TRAINING_TYPE
    )
    model = VLM(config).cuda()
    tokenizer = AutoTokenizer.from_pretrained(qwen_path)

    # model.qwen.gradient_checkpointing_enable(use_cache=False)
    # model.clip.gradient_checkpointing_enable(use_cache=False)

    print("数据集构建...")
    train_dataset = LoRADataset(qwen_path, clip_path, config, split_type="train")
    eval_dataset = LoRADataset(qwen_path, clip_path, config, split_type="val")

    data_collator = VLMDataCollator(
        tokenizer=tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
    )

    print("配置参数...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=1,
        weight_decay=0.03,
        warmup_ratio=0.05,
        optim="adamw_torch",  # adamw_8bit备用
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        logging_steps=40,
        logging_dir=os.path.join(output_dir, "logs"),
        save_strategy="steps",
        eval_strategy="steps",
        save_total_limit=2,
        save_steps=40,
        eval_steps=40,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        dataloader_num_workers=6,
        dataloader_pin_memory=True,
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
    # trainer.train(resume_from_checkpoint=True)
    trainer.train()

    print("训练完成，正在保存最佳模型...")
    trainer.save_model(os.path.join(output_dir, "lora_best_model"))
