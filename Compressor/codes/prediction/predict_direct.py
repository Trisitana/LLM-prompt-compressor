import json
import time
import torch
import numpy as np
import torch.nn as nn
from peft import LoraConfig
from L3LoraL3 import L3LoraL3
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM, TrainingArguments, Trainer, AutoModelForCausalLM
import os


class TextDataset(Dataset):
    def __init__(self, filepath, start_line=0):
        """
        Load the QA dataset.

        Args:
            filepath (str): Path for extractive QA pairs (jsonl, lines of json).
            start_line (int): The line to start to be loaded.
        """
        self.data = []
        with open(filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()[start_line:]
            for line in lines:
                json_data = json.loads(line.strip())
                self.data.append(json_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    device = torch.device("cuda")

    # whether to clear the previous predictions
    clear = True
    # path for extractive QA pairs
    text_file_path = os.path.join("../finetuning/datasets/test.jsonl")

    # root path to save results
    root = "raw"
    if not os.path.exists(root):
        os.makedirs(root)
    # path for generated answers
    output_file_path = root + "/gen_results"
    # path for target answers
    target_file_path = root + "/tar_results"
    # path for the questions in the extractive QA pairs
    question_file_path = root + "/question"
    # path for the contexts in the extractive QA pairs
    context_file_path = root + "/context"
    # path for the time for generating the answer
    predict_time_path = root + "/predict_time"
    # path for the length of the generated answer
    generated_token_length_path = root + "/generated_token_length"

    # max number of text tokens in encoder input (truncation or padding)
    context_len = 512
    # max number of context tokens to be compressed
    # max number of new tokens to be generated
    max_new_tokens = 46

    # this file only supports batch size 1
    batch_size = 1
    # start line of the benchmark file
    # set clear=False and start_line for continued prediction
    start_line = 0
    dataset = TextDataset(text_file_path, start_line=start_line)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # number of texts to be processed
    num_texts = 5000

    # huggingface path for the LLM
    base_url = "/seu_nvme/home/jiayuheng/213210050/llama3/Meta-Llama-3-8B-Instruct_20250227105222"

    llama_path = os.path.join(base_url, "Meta-Llama-3-8B-Instruct")
    # cache path to save or load the LLM
    cache_dir = "<to be filled>"
    # huggingface token to use the LLaMA model
    use_auth_token = "<to be filled>"

    # load the compression model: llama-3-lora + llama-3
    model = AutoModelForCausalLM.from_pretrained(
        llama_path,
        cache_dir=cache_dir,
        use_auth_token=use_auth_token,
        torch_dtype=torch.bfloat16,
    )
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(llama_path, use_auth_token="<to be filled>")

    # whether to clear the output file before prediction
    if clear == True:
        with open(output_file_path, 'w') as file:
            pass

        with open(target_file_path, 'w') as file:
            pass

        with open(question_file_path, 'w') as file:
            pass

        with open(context_file_path, 'w') as file:
            pass

        with open(predict_time_path, 'w') as file:
            pass

        with open(generated_token_length_path, 'w') as file:
            pass

    # number of the data record
    i = 0

    for batch_texts in data_loader:
        print(f"Processed {i} batches.")

        # record the question in the extractive QA pair
        with open(question_file_path, 'a', encoding='utf-8') as file:
            file.write(batch_texts["question"][0] + '\n')

        # record the context in the extractive QA pair
        with open(context_file_path, 'a', encoding='utf-8') as file:
            file.write(batch_texts["context"][0] + '\n')

        target_text = batch_texts["answer"][0]
        with open(target_file_path, 'a', encoding='utf-8') as file:
            target_text = target_text.replace("\n", " ").strip()
            file.write(target_text + '\n')

        question = batch_texts["question"][0]

        prompt = f"Question: {question} Answer: "

        text_tokens = tokenizer(batch_texts["context"][0]+ prompt,
                                      truncation=True,
                                      max_length=context_len+max_new_tokens,
                                      padding="max_length",
                                      return_tensors="pt",
                                      add_special_tokens=False).input_ids.to(device)

        # compress the context
        predict_start = time.time()

        generated_text = []

        past_key_values = None
        # Get target answer length in tokens
        target_tokens = tokenizer(target_text, add_special_tokens=False).input_ids
        max_target_length = len(target_tokens)
        
        # Generate up to target length tokens
        for _ in range(min(max_target_length, max_new_tokens)):
            input_embeddings = model.get_input_embeddings()(text_tokens)

            # predict the next new token by the original LLM
            output = model(
                    inputs_embeds=input_embeddings,
                    past_key_values=past_key_values
            )
            logits = output.logits
            past_key_values = output.past_key_values

            # choose the token id with the highest probability
            next_token = torch.argmax(logits[:, -1, :], dim=-1)

            # stop generating new tokens when meet end tokens
            if  next_token == tokenizer.eos_token_id or next_token == 128001:
                end = True
                break

            # add the new token to generated tokens
            generated_text.append(next_token.item())

            # update the input token
            text_tokens = next_token.unsqueeze(0)

        # length of the generated tokens
        generated_token_length = len(generated_text)
        generated_text = tokenizer.decode(
            generated_text,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        predict_end = time.time()
        # record the time for compression
        predict_time = predict_end - predict_start
        print(f"predict in {predict_time:.2f} seconds.")

        with open(predict_time_path, 'a', encoding='utf-8') as file:
            file.write(str(predict_time) + '\n')

        with open(generated_token_length_path, 'a', encoding='utf-8') as file:
            file.write(str(generated_token_length) + '\n')

        # record the generated answer
        with open(output_file_path, 'a', encoding='utf-8') as file:
            generated_text = generated_text.replace("\n", " ").strip()
            file.write(generated_text + '\n')

        i += 1

        # stop if reach the maximum number of processed data records
        if i == num_texts:
            break


