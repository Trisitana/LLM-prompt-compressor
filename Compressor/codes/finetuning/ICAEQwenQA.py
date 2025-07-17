import torch
import torch.nn as nn
from peft import get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def load_lora_parameters(model, lora_params_path):
    lora_params = torch.load(lora_params_path)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in lora_params: 
                if 'lora' in name or 'memory_embeddings' in name:
                    param.copy_(lora_params[name])
                    if 'memory_embeddings' in name:
                        print("Loaded memory_embeddings!")
                else:
                    print(f"No saved parameter for {name}")
            elif "lora" in name:
                print(f"No saved parameter for {name}")

class ICAEQwenQA(nn.Module):
    def __init__(
        self,
        qwen_path,
        max_context_length,
        lora_path,
        lora_config,
        num_mem,
        device
    ):
        """
        Create the compression model: Qwen-ICAE + Qwen.

        Args:
            qwen_path (str): Path for the base Qwen model.
            max_context_length (int): Max number of context tokens to be compressed.
            lora_path (str): Pretrained LoRA parameters for initialization.
            lora_config (LoraConfig): LoRA configurations.
            num_mem (int): Number of compressed tokens.
            device (torch.device): CPU or GPU.
        """
        super(ICAEQwenQA, self).__init__()
        qwen = AutoModelForCausalLM.from_pretrained(
            qwen_path, 
            cache_dir="<to be filled>", 
            use_auth_token="<to be filled>",
            torch_dtype=torch.bfloat16,
        )
        self.qwen = get_peft_model(qwen, lora_config)
        for name, param in self.qwen.named_parameters():
            param.requires_grad = False
            if 'lora' in name:
                param.requires_grad = True
        print(f"Total parameters of qwen: {sum(p.numel() for p in self.qwen.parameters())}")
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_path, use_auth_token="<to be filled>")
        print("qwen tokenizer loaded.")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_context_length = max_context_length
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.num_mem = num_mem
        self.memory_embeddings = nn.Parameter(torch.randn(1, num_mem, 3584, dtype=torch.bfloat16).to(device))
        self.memory_embeddings.requires_grad = True
        load_lora_parameters(self, lora_path)
        self.device = device

    def forward(self, input_ids, labels):
        ####################
        # Encoder - qwen+lora
        ####################
        text_tokens = input_ids[:, :self.max_context_length]
        qa_tokens = input_ids[:, self.max_context_length:]
        target_tokens = labels
        text_tok_embeddings = self.qwen.get_input_embeddings()(text_tokens).to(self.device)
        memory_tok_embeddings = self.memory_embeddings.repeat(text_tok_embeddings.shape[0], 1, 1).to(self.device)
        encoder_input_embeddings = torch.cat((text_tok_embeddings, memory_tok_embeddings), dim=1)
        encoder_output = self.qwen(inputs_embeds=encoder_input_embeddings, output_hidden_states=True).hidden_states[-1]
        encoder_output = encoder_output[:, -self.num_mem:, :]

        ####################
        # Decoder - qwen
        ####################
        qa_tok_embeddings = self.qwen.get_input_embeddings()(qa_tokens)
        decoder_input_embeddings = torch.cat((encoder_output, qa_tok_embeddings), dim=1)
        with self.qwen.disable_adapter():
            decoder_output = self.qwen(inputs_embeds=decoder_input_embeddings)
        all_logits = decoder_output.logits

        loss = self.criterion(all_logits.view(-1, all_logits.size(-1)), target_tokens.view(-1))

        return {'loss': loss, 'logits': all_logits}

    def save_lora_parameters(self, save_directory):
        """
        保存 LoRA 参数和 memory_embeddings 的权重。
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        # 保存 LoRA 参数和 memory_embeddings
        lora_params_path = os.path.join(save_directory, "lora_params.pth")

        lora_params = {name: param for name, param in self.named_parameters() if 'lora' in name}
        lora_params['memory_embeddings'] = self.memory_embeddings  # 添加 memory_embeddings
        torch.save(lora_params, lora_params_path)

        print(f"LoRA parameters and memory_embeddings saved to {lora_params_path}") 