import json
import os
import time
import torch
import numpy as np
import torch.nn as nn
from peft import LoraConfig
from L3LoraL3 import L3LoraL3
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, LlamaForCausalLM, TrainingArguments, Trainer


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

def group_by_token_length(text_dataset, tokenizer,max_lengths_group):

        groups = {length: [] for length in max_lengths_group}
        for idx, line in enumerate(text_dataset.lines):
            tokens = tokenizer(line, add_special_tokens=False).input_ids
            token_count = len(tokens)
            for length in max_lengths_group[::-1]:  # 倒序遍历
              if token_count >= length:
                groups[length].append(idx)
                break
        return groups


if __name__ == "__main__":
    device = torch.device("cuda")

    base_url = "/seu_nvme/home/jiayuheng/213210050/llama3/Meta-Llama-3-8B-Instruct_20250227105222"

    # whether to clear the previous predictions
    clear = True
    # path for the original texts, lines of texts
    text_file_path = os.path.join(base_url, "test_data.txt")
    # huggingface path for the LLM
    llama_path = os.path.join(base_url, "Meta-Llama-3-8B-Instruct")
    # cache path to save or load the LLM
    cache_dir = "<to be filled>"
    # huggingface token to use the LLaMA model
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
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # number of texts to be processed
    num_texts = 500

    # LoRA configurations
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 定义循环的target和max_length
    target = "lora_param_512to1_15_epoch"

    max_lengths = [96, 192, 288, 384, 480]

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

    model = L3LoraL3(
        llama_path=llama_path,
        cache_dir=cache_dir,
        use_auth_token=use_auth_token,
        max_length=max_new_tokens,
        lora_path=lora_path,
        lora_config=lora_config,
        num_mem=num_mem,
        device=device
    )
    model = model.to(device)

    groups = group_by_token_length(dataset,model.tokenizer,max_lengths)

    for max_length in max_lengths:
            # 根据target中的to{x}设置num_mem
            # 更新max_new_tokens
            max_new_tokens = max_length

            model.max_length = max_length

            # 设置输出文件路径，添加max_length后缀
            output_file_path = os.path.join("regenerate", target, max_length.__str__(),"output.txt")
            target_file_path = os.path.join("regenerate", target, max_length.__str__(),"target.txt")
            status_file_path = os.path.join("regenerate", target, max_length.__str__(),"status.txt")

            output_dir = os.path.dirname(output_file_path)
            target_dir = os.path.dirname(target_file_path)
            status_dir = os.path.dirname(status_file_path)

            # 如果目录不存在，则创建目录
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            if not os.path.exists(status_dir):
                os.makedirs(status_dir)

            # 是否清除之前的输出文件
            if clear:
                with open(output_file_path, 'w') as file:
                    pass

                with open(target_file_path, 'w') as file:
                    pass

                with open(status_file_path, 'w') as file:
                    pass
            # 数据记录编号
            i = 0
            # 自动停止的数据记录数量
            n = 0

            group_indices = groups[max_length]

            subset_dataset = Subset(dataset, group_indices)
            group_data_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False)

            for batch_texts in group_data_loader:
                print(f"Processed {i} batches for target {target} with max_length {max_length}.")
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
                    max_length=max_length,
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
                past_key_values = model.compress(
                    text=batch_texts,
                    text_tokens=back_tokens.unsqueeze(0),
                    output_path=None
                )

                # 生成文本
                predicted_text, end, generated_token_length = model.predict(
                    past_key_values=past_key_values,
                    max_new_tokens=max_new_tokens,
                    prompt=prompt
                )

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
                    f"{n}/{i + 1} Processed in {elapsed_time:.2f} seconds for target {target} with max_length {max_length}.")

                i += 1

                # 达到最大处理数量时停止
                if i == num_texts:
                    break

            # 记录自动停止的数量
            with open(status_file_path, "a") as file:
                file.write(f"{n}\n")