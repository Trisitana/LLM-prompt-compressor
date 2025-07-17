import json
import os
import time
import torch
import numpy as np
import torch.nn as nn
from peft import LoraConfig
from L3LoraQwen import L3LoraQwen
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer


class TextDataset(Dataset):
    def __init__(self, filepath, start_line=0):
        """
        Collect texts.

        Args:
            filepath (str): Path for lines of texts.
            start_line (int): The line start to be loaded.
        """
        with open(filepath, 'r', encoding='utf-8') as file:
            self.lines = file.readlines()[start_line:]
       
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx].strip()

def group_by_token_length(text_dataset, tokenizer, min_lengths_group):
    groups = {length: [] for length in min_lengths_group}
    for idx, line in enumerate(text_dataset.lines):
        tokens = tokenizer(line, add_special_tokens=False).input_ids
        token_count = len(tokens)
        # 修改分组逻辑，允许重叠
        for length in min_lengths_group:
            if token_count >= length:
                groups[length].append(idx)
    return groups

if __name__ == "__main__":
    device = torch.device("cuda")

    base_url = "/seu_nvme/home/jiayuheng/213210050/llama3/Meta-Llama-3-8B-Instruct_20250227105222"

    # whether to clear the previous predictions
    clear = True
    # path for the original texts, lines of texts
    text_file_path = os.path.join(base_url, "split_test_unique.txt")
    # huggingface path for the Qwen model
    qwen_path = os.path.join(base_url, "Qwen2.5-7B-Instruct_20250426235044/Qwen2.5-7B-Instruct")
    # cache path to save or load the Qwen model
    cache_dir = "<to be filled>"
    # huggingface token to use the Qwen model
    use_auth_token = "<to be filled>"

    base_lora_path = "../pretraining/lora_param_500"
    # "bos" for regeneration
    prompt = "bos"
    # "regeneration" or "qa"
    mode = "regeneration"
    # max number of text tokens in encoder input (truncation or padding)
    context_len = 512
    # max number of new tokens to be generated
    max_new_tokens = 96  # 默认值，后续会根据max_length调整

    # this file only supports batch size 1
    batch_size = 1
    # start line of the benchmark file
    # set clear=False and start_line for continued prediction
    start_line = 0
    dataset = TextDataset(text_file_path, start_line=start_line)
    # number of texts to be processed
    num_texts = 500

    # LoRA configurations
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Qwen specific attention layers
    )

    # 定义循环的target和max_length
    target = "qwen_param_512to1_3_epoch"

    min_lengths = [192, 288, 384]

    if "to8" in target:
        num_mem = 8
    elif "to4" in target:
        num_mem = 4
    elif "to2" in target:
        num_mem = 2
    elif "to16" in target:
        num_mem = 16
    elif "to1" in target:
        num_mem = 1
    else:
        raise ValueError(f"Unknown target format: {target}")

    lora_path_regen = os.path.join(base_lora_path, target, "lora_params.pth")
    lora_path_qa = ""

    if mode == "regeneration":
        lora_path = lora_path_regen
    elif mode == "qa":
        lora_path = lora_path_qa
    else:
        print("""Please specify the mode: "regeneration" or "qa".""")

    model = L3LoraQwen(
        qwen_path=qwen_path,
        cache_dir=cache_dir,
        use_auth_token=use_auth_token,
        max_length=max_new_tokens,
        lora_path=lora_path,
        lora_config=lora_config,
        num_mem=num_mem,
        device=device
    )
    model = model.to(device)

    groups = group_by_token_length(dataset, model.tokenizer, min_lengths)

    for min_length in min_lengths:
        # 更新max_new_tokens
        max_new_tokens = min_length

        model.max_length = min_length

        # 设置输出文件路径，添加max_length后缀
        output_file_path = os.path.join("regenerate", target, 'min_'+min_length.__str__(), "output.txt")
        target_file_path = os.path.join("regenerate", target, 'min_'+min_length.__str__(), "target.txt")
        status_file_path = os.path.join("regenerate", target, 'min_'+min_length.__str__(), "status.txt")
        # 添加压缩时间和预测时间的文件路径
        compress_time_path = os.path.join("regenerate", target, 'min_'+min_length.__str__(), "compress_time.txt")
        predict_time_path = os.path.join("regenerate", target, 'min_'+min_length.__str__(), "predict_time.txt")

        output_dir = os.path.dirname(output_file_path)
        target_dir = os.path.dirname(target_file_path)
        status_dir = os.path.dirname(status_file_path)
        compress_time_dir = os.path.dirname(compress_time_path)
        predict_time_dir = os.path.dirname(predict_time_path)

        # 如果目录不存在，则创建目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        if not os.path.exists(status_dir):
            os.makedirs(status_dir)
        if not os.path.exists(compress_time_dir):
            os.makedirs(compress_time_dir)
        if not os.path.exists(predict_time_dir):
            os.makedirs(predict_time_dir)

        # 是否清除之前的输出文件
        if clear:
            with open(output_file_path, 'w') as file:
                pass
            with open(target_file_path, 'w') as file:
                pass
            with open(status_file_path, 'w') as file:
                pass
            with open(compress_time_path, 'w') as file:
                pass
            with open(predict_time_path, 'w') as file:
                pass

        # 数据记录编号
        i = 0
        # 自动停止的数据记录数量
        n = 0

        group_indices = groups[min_length]
        subset_dataset = Subset(dataset, group_indices)
        group_data_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False)

        for batch_texts in group_data_loader:
            print(f"Processed {i} batches for target {target} with min_length {min_length}.")
            # 初始化上下文token
            back_tokens = torch.full(
                (context_len,),
                model.tokenizer.eos_token_id,
                dtype=torch.long
            )

            start_time = time.time()

            # 获取文本token
            text_tokens = model.tokenizer(
                batch_texts,
                truncation=True,
                max_length=min_length,
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids[0]

            # 存储上下文token
            back_tokens[0:text_tokens.shape[0]] = text_tokens

            # 目标文本（原始上下文）
            target_text = model.tokenizer.decode(
                text_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            # 记录目标文本
            with open(target_file_path, 'a', encoding='utf-8') as file:
                target_text = target_text.replace("\n", " ").strip()
                file.write(target_text + '\n')

            # 压缩上下文
            compress_start = time.time()
            past_key_values = model.compress(
                text=batch_texts,
                text_tokens=back_tokens.unsqueeze(0),
                output_path=None
            )
            compress_end = time.time()
            # 记录压缩时间
            compress_time = compress_end - compress_start
            print(f"Compressed in {compress_time:.2f} seconds.")
            with open(compress_time_path, 'a', encoding='utf-8') as file:
                file.write(str(compress_time) + '\n')

            # 生成文本
            predict_start = time.time()
            predicted_text, end, generated_token_length = model.predict(
                past_key_values=past_key_values,
                max_new_tokens=max_new_tokens,
                prompt=prompt
            )
            predict_end = time.time()
            # 记录预测时间
            predict_time = predict_end - predict_start
            print(f"Predicted in {predict_time:.2f} seconds.")
            with open(predict_time_path, 'a', encoding='utf-8') as file:
                file.write(str(predict_time) + '\n')

            # 是否自动停止
            if end == True:
                n += 1

            # 记录生成文本
            with open(output_file_path, 'a', encoding='utf-8') as file:
                predicted_text = predicted_text.replace("\n", " ").strip()
                file.write(predicted_text + '\n')

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(
                f"{n}/{i + 1} Processed in {elapsed_time:.2f} seconds for target {target} with min_length {min_length}.")

            i += 1

            # 达到最大处理数量时停止
            if i == num_texts:
                break

        # 记录自动停止的数量
        with open(status_file_path, "a") as file:
            file.write(f"{n}\n") 