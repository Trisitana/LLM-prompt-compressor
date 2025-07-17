import json
import sys
import os

import wandb

import torch
import numpy as np
import torch.nn as nn

import torch.optim as optim
from peft import LoraConfig
from L3LoraQwen import L3LoraQwen
from torch.utils.data import Dataset, DataLoader


from transformers import AutoTokenizer, LlamaForCausalLM, TrainingArguments, Trainer


base_url = "/seu_nvme/home/jiayuheng/213210050/llama3/Meta-Llama-3-8B-Instruct_20250227105222"

def read_text_from_file(file_path):
    """
    Load lines of texts.

    Args:
        file_path (str): Path for lines of texts.

    Returns:
        (List[str]): List of texts.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

class TextDataset(Dataset):
    def __init__(self, text_file, qwen_path, max_length):
        """
        Create the training or evaluation dataset.

        Args:
            text_file (str): Path for lines of texts.
            qwen_path (str): Path for the base Qwen model.
            max_length (int): Max number of tokens to be compressed.
        """
        self.text = read_text_from_file(text_file)
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_path, use_auth_token="<to be filled>")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        print(self.tokenizer.pad_token)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text_tokens = self.tokenizer(
            self.text[idx], 
            truncation=True, 
            padding="max_length", 
            max_length=self.max_length, 
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids
        input_ids = text_tokens.squeeze()
        # input text tokens + EOS token
        target_tokens = torch.full((len(input_ids)+1,), -100, dtype=torch.long)
        text_eos_tokens = input_ids.tolist()
        text_eos_tokens.append(self.tokenizer.eos_token_id)
        text_eos_tokens_len = len(text_eos_tokens)
        target_tokens[0:0+text_eos_tokens_len] = torch.tensor(text_eos_tokens, dtype=torch.long, device=device)
        return {"input_ids": input_ids, "labels": target_tokens}


if __name__ == "__main__":
   
    device = torch.device(f"cuda")
    # ====================
    # training configurations
    # ====================
    project_name = "pretraining_500"
    # training corpus (lines of texts)
   
    # path to save the results
  
    current_file_name = os.path.splitext(os.path.basename(__file__))[0]

    output_dir = f"output_dir_500_save_lora/{current_file_name}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # number of compressed tokens
    num_mem = 1
    # max number of tokens to be compressed
    max_length = 512
    # resume training from specific checkpoint
    resume_from_checkpoint = None
    # (huggingface) path for the base Qwen model
    qwen_path = os.path.join(base_url, "Qwen2.5-7B-Instruct_20250426235044/Qwen2.5-7B-Instruct")

    # path for the deepspeed configuration
    deepspeed_config = "../deepspeed_configurations.json"
    logging_dir = "logging_dir_500"
    num_train_epochs = 3
    per_device_train_batch_size = 8
    per_device_eval_batch_size = 48
    save_strategy = "steps"
    save_steps = 300
    evaluation_strategy = "steps"
    eval_steps = 100
    eval_accumulation_steps = 4
    logging_steps = 1
    learning_rate = 1e-4
    save_total_limit = 1
    lr_scheduler_type = "constant_with_warmup"
    warmup_steps = 300

   
    # create the training and evaluation datasets

    train_text_path = os.path.join(base_url,"split_train.txt")
    test_text_path = os.path.join(base_url,"split_validation.txt")
    train_dataset = TextDataset(train_text_path, qwen_path, max_length)
    test_dataset = TextDataset(test_text_path, qwen_path, max_length)
    print("Dataset created.")

    # train_first_element = train_dataset[0]
    # test_first_element = test_dataset[0]

    # print("Train Dataset - First Element:")
    # print("Input IDs:", train_first_element["input_ids"])
    # print("Labels:", train_first_element["labels"])

    # print("\nTest Dataset - First Element:")
    # print("Input IDs:", test_first_element["input_ids"])
    # print("Labels:", test_first_element["labels"])

    # print(train_first_element["input_ids"].shape)

    lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Add target modules for Qwen attention layers
    ) 
    
    wandb.init(project=project_name, mode="offline")


    # ====================
    # compression model
    # ====================
    print("Loading Qwen + lora + Qwen ...")
    model = L3LoraQwen(
        qwen_path=qwen_path,
        max_length=max_length,
        lora_config=lora_config,
        num_mem=num_mem,
        device=device
    )

    print("Number of trainable parameters in the model: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Model is on CUDA device:", torch.cuda.current_device())
    model.config = model.qwen.config
    print("model.qwen.config: ", model.qwen.config)
    print("Qwen + lora + Qwen loaded successfully.")

    # ====================
    # Training
    # ====================
    # give the detailed information for the error
    torch.autograd.set_detect_anomaly(True)

    # training parameters

    training_args = TrainingArguments(
        output_dir=output_dir,          
        overwrite_output_dir=False,
        num_train_epochs=num_train_epochs,              
        per_device_train_batch_size=per_device_train_batch_size,   
        per_device_eval_batch_size=per_device_eval_batch_size,
        save_strategy=save_strategy,
        save_steps=save_steps,      
        evaluation_strategy=evaluation_strategy,    
        eval_steps=eval_steps, 
        eval_accumulation_steps=eval_accumulation_steps,
        logging_dir=logging_dir,    
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        save_total_limit=save_total_limit,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    # If the resume path is not none
    # continue from the provided checkpoint
    if resume_from_checkpoint == None:
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)


    save_path = "lora_param_500/qwen_param_512to1_3_epoch"
    trainer.model.save_lora_parameters(save_path)
    
    evaluation_results = trainer.evaluate()
    print("evaluation_results: ", evaluation_results)








