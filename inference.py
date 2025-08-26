import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor
from safetensors.torch import load_file as load_safetensors
import torch.nn.functional as F
import os

from model import VLM, VLMConfig


base_path = os.path.dirname(os.path.abspath(__file__))
state_path = os.path.join(base_path, "output", "best_model")
qwen_path = os.path.join(base_path, "Qwen3-0.6B")
clip_path = os.path.join(base_path, "clip-vit-base-patch16")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
model.eval()

tokenizer = AutoTokenizer.from_pretrained(qwen_path)
processor = AutoProcessor.from_pretrained(clip_path)

print("模型加载完成")


def generate_response(model, tokenizer, device):
    while True:
        usr_input = input("输入图片路径：")
        if usr_input.strip() == "":
            print("无图片输入")
        image_path = os.path.join(base_path, usr_input)
        if not os.path.exists(image_path):
            print("图片路径不存在或格式错误")
            continue
        prompt = input("输入问题：")
        if prompt.strip() == "":
            print("问题不能为空")
            continue
        elif prompt.lower() == "exit":
            print("退出程序")
            break
        
        if not usr_input.strip() == "":
            image = Image.open(image_path).convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(
                device
            )

        if "<image>" not in prompt:
            prompt += "\n<image>"
        template_prompt = prompt.replace(
            "<image>", "<|image_pad|>" * model.image_pad_num
        )

        conversation = [{"role": "user", "content": template_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        print("模型生成回答")

        # 和forward中一个步骤
        text_embeds = model.qwen.get_input_embeddings()(input_ids)
        if not usr_input.strip() == "":
            image_embeds = model.clip.vision_model(pixel_values).last_hidden_state[:, 1:, :]
            b, s, d = image_embeds.shape
            image_embeds = image_embeds.view(b, -1, 4 * d)
            image_features = model.dense2(F.silu(model.dense1(image_embeds)))
            text_embeds = text_embeds.to(image_features.dtype)
            inputs_embeds = model.merge_text_and_image(
                input_ids, text_embeds, image_features
            )
        else:
            inputs_embeds = text_embeds

        # text_prompt = tokenizer.apply_chat_template(
        #     [{"role": "user", "content": "你好，请做个自我介绍"}],
        #     tokenize=False,
        #     add_generation_prompt=True,
        # )
        # text_input_ids = tokenizer(text_prompt, return_tensors="pt").to(device)

        eos_token_id = tokenizer.eos_token_id
        if not isinstance(eos_token_id, list):
            eos_token_id = [eos_token_id]

        print(f"--- [DEBUG] Input shape: {input_ids.shape}")

        with torch.no_grad():
            generation_output = model.qwen.generate(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.9,
                top_p=0.9,
                eos_token_id=eos_token_id,
            )

        print(f"--- [DEBUG] Output shape: {generation_output.shape}")

        # with torch.no_grad():
        #     generation_output = model.qwen.generate(
        #     **text_input_ids,
        #     max_new_tokens=50,
        #     eos_token_id=tokenizer.eos_token_id
        # )

        response_ids = generation_output[:, input_ids.shape[1]:]
        
        # print("--- [DEBUG] Raw Generated Token IDs ---")
        # print(response_ids)
        # print("-----------------------------------")

        print("--- [DEBUG] Raw Decoded Output ---")
        response_raw = tokenizer.decode(response_ids[0], skip_special_tokens=False)
        print(repr(response_raw)) # 使用 repr 确保能看到完整的字符串
        print("---------------------------------")


        response_clean = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        print("回答：", response_clean)


if __name__ == "__main__":
    generate_response(model, tokenizer, device)
