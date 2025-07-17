import json
import time
import torch
import numpy as np
import torch.nn as nn
from peft import LoraConfig
from L3LoraL3 import L3LoraL3
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import pipeline
import os

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

    # whether to clear the previous predictions  
    base_url = "/seu_nvme/home/jiayuheng/213210050/llama3/Meta-Llama-3-8B-Instruct_20250227105222"

    # whether to clear the previous predictions
    clear = True
    # path for the original texts, lines of texts
    text_file_path = os.path.join(base_url, "split_test_unique.txt")

    # root path to save results
    root = "regenerate/all_with_system_qwen_step"
    if not os.path.exists(root):
        os.makedirs(root)
    # path for generated answers
    
    # this file only supports batch size 1
    batch_size = 1
    # start line of the benchmark file
    # set clear=False and start_line for continued prediction
    start_line = 0
    dataset = TextDataset(text_file_path, start_line=start_line)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # number of texts to be processed
    num_texts = 500

    # huggingface path for the LLM
    base_url = "/seu_nvme/home/jiayuheng/213210050/llama3/Meta-Llama-3-8B-Instruct_20250227105222"
    qwen_path = os.path.join(base_url, "Qwen2.5-7B-Instruct_20250426235044/Qwen2.5-7B-Instruct")

    # Create pipeline
    pipeline = pipeline(
        "text-generation",
        model=qwen_path,
        torch_dtype=torch.bfloat16,
        device=device
    )
    min_lengths = [192, 288, 384]

    groups = group_by_token_length(dataset, pipeline.tokenizer, min_lengths)


    for min_length in min_lengths:

        output_file_path = os.path.join(root, 'min_'+min_length.__str__(), "output.txt")
        target_file_path = os.path.join(root, 'min_'+min_length.__str__(), "target.txt")
        status_file_path = os.path.join(root, 'min_'+min_length.__str__(), "status.txt")
        predict_time_path = os.path.join(root, 'min_'+min_length.__str__(), "predict_time.txt")

        output_dir = os.path.dirname(output_file_path)
        target_dir = os.path.dirname(target_file_path)
        status_dir = os.path.dirname(status_file_path)
        predict_time_dir = os.path.dirname(predict_time_path)

        # 如果目录不存在，则创建目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        if not os.path.exists(status_dir):
            os.makedirs(status_dir)
        if not os.path.exists(predict_time_dir):
            os.makedirs(predict_time_dir)

        if clear:
            with open(output_file_path, 'w') as file:
                pass
            with open(target_file_path, 'w') as file:
                pass
            with open(status_file_path, 'w') as file:
                pass
            with open(predict_time_path, 'w') as file:
                pass

        # number of the data record
        i = 0

        group_indices = groups[min_length]
        subset_dataset = Subset(dataset, group_indices)
        group_data_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False)

        for batch_texts in group_data_loader:
            
            print(f"Processed {i} batches.")

            # Format messages for pipeline
            # Tokenize and truncate context to min_length
            text_tokens = pipeline.tokenizer(
                batch_texts,
                truncation=True,
                max_length=min_length,
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids[0]

            # Decode truncated tokens back to text
            context = pipeline.tokenizer.decode(
                text_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            # 记录目标文本
            with open(target_file_path, 'a', encoding='utf-8') as file:
                target_text = context.replace("\n", " ").strip()
                file.write(target_text + '\n')

            messages = [
                {"role": "system", "content": f"请你打印我给你的context"},
                {"role": "user", "content": f"context: {context}"}
            ]

            prompt = pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize prompt and move to device
            input_ids = pipeline.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            
            terminators = [
                pipeline.tokenizer.eos_token_id,
                151643
            ]
            
            predict_start = time.time()
            
            # Initialize empty list for generated tokens
            generated_tokens = []
            
            # Initialize past key values as None
            past_key_values = None
            
            # Generate tokens one by one
            for _ in range(min_length):
                with torch.no_grad():
                    # Forward pass with past key values
                    outputs = pipeline.model(
                        input_ids if past_key_values is None else input_ids[:, -1:],
                        past_key_values=past_key_values,
                    )
                
                # Get next token probabilities
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)
                
                # Break if we hit a terminator
                if next_token.item() in terminators:
                    break
                    
                # Append token and update input_ids
                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
                
                # Update past key values
                past_key_values = outputs.past_key_values
            
            # Decode generated tokens
            generated_text = pipeline.tokenizer.decode(generated_tokens)
            
            predict_end = time.time()
            predict_time = predict_end - predict_start
            
            generated_token_length = len(generated_tokens)
      
            print(f"predict in {predict_time:.2f} seconds.")

            with open(predict_time_path, 'a', encoding='utf-8') as file:
                file.write(str(predict_time) + '\n')

            # record the generated answer
            with open(output_file_path, 'a', encoding='utf-8') as file:
                generated_text = generated_text.replace("\n", " ").strip()
                file.write(generated_text + '\n')

            i += 1

            # stop if reach the maximum number of processed data records
            if i == num_texts:
                break
