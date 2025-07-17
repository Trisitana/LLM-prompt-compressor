import json
import wandb
import torch
import numpy as np
import torch.nn as nn
import os

from ICAEQwen import ICAEQwen
from peft import LoraConfig
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

base_url = "/seu_nvme/home/jiayuheng/213210050/llama3/Meta-Llama-3-8B-Instruct_20250227105222"


def read_text_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    return [line.strip() for line in lines]


class TextDataset(Dataset):
    def __init__(self, text_file, qwen_path, max_length, num_mem):
        self.text = read_text_from_file(text_file)
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        self.num_mem = num_mem

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
        prompt_len = 1
        target_tokens = torch.full((self.num_mem + prompt_len + len(input_ids),), -100, dtype=torch.long)
        text_eos_tokens = input_ids.tolist()
        text_eos_tokens.append(self.tokenizer.eos_token_id)
        text_eos_tokens_len = len(text_eos_tokens)
        target_tokens[self.num_mem + prompt_len - 1:self.num_mem + prompt_len - 1 + text_eos_tokens_len] = torch.tensor(
            text_eos_tokens, dtype=torch.long, device=device)
        return {"input_ids": input_ids, "labels": target_tokens}


if __name__ == "__main__":
    device = torch.device(f"cuda")

    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

    # ====================
    # Training parameters
    # ====================
    project_name = "ICAE"
    resume_from_checkpoint = None

    current_file_name = os.path.splitext(os.path.basename(__file__))[0]

    output_dir = f"/seu_share/home/jiayuheng/213210050/LLM/pretrain_qwen/{current_file_name}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_mem = 1
    max_length = 512
    qwen_path = os.path.join(base_url, "Qwen2.5-7B-Instruct_20250426235044/Qwen2.5-7B-Instruct")

    deepspeed_config = "../deepspeed_configurations.json"
    logging_dir = "ICAE_pretrain_log"
    num_train_epochs = 3
    per_device_train_batch_size = 8
    per_device_eval_batch_size = 12
    save_strategy = "steps"
    save_steps = 300
    evaluation_strategy = "steps"
    eval_steps = 200
    eval_accumulation_steps = 4
    logging_steps = 1
    learning_rate = 1e-4
    save_total_limit = 1
    lr_scheduler_type = "constant_with_warmup"
    warmup_steps = 300

    train_text_path = os.path.join(base_url, "split_train.txt")
    test_text_path = os.path.join(base_url, "split_validation.txt")
    train_dataset = TextDataset(train_text_path, qwen_path, max_length, num_mem)
    test_dataset = TextDataset(test_text_path, qwen_path, max_length, num_mem)
    print("Dataset created.")

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
    # AutoEncoder
    # ====================
    print("Loading qwen + lora + qwen ...")
    model = ICAEQwen(
        qwen_path=qwen_path,
        max_length=max_length,
        lora_config=lora_config,
        num_mem=num_mem,
        device=device
    )
    print("Number of trainable parameters in the model: ",
          sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Model is on CUDA device:", torch.cuda.current_device())
    model.config = model.qwen.config
    print("model.qwen.config: ", model.qwen.config)
    print("qwen + lora + qwen loaded successfully.")

    # ====================
    # Train
    # ====================
    torch.autograd.set_detect_anomaly(True)

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
        # deepspeed=deepspeed_config,
        learning_rate=learning_rate,
        save_total_limit=save_total_limit,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    if resume_from_checkpoint == None:
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    save_path = f"lora_param_500/{current_file_name}"
    trainer.model.save_lora_parameters(save_path)

    evaluation_results = trainer.evaluate()
    print("evaluation_results: ", evaluation_results)

