import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model

def load_lora_parameters(model, lora_params_path):
    lora_params = torch.load(lora_params_path)
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in lora_params: 
                if 'lora' in name or 'memory_embeddings' in name or "ae_embedding" in name:
                    param.copy_(lora_params[name])
                    if 'memory_embeddings' in name:
                         print("Found memory_embeddings!")
                    if "ae_embedding" in name:
                        print("Found ae_embedding!")
                else:
                    print(f"No saved parameter for {name}")
            elif "lora" in name:
                print(f"No saved parameter for {name}")
                
class ICAEQwen(nn.Module):
    def __init__(
        self,
        qwen_path,
        cache_dir,
        use_auth_token,
        max_length,
        lora_path,
        lora_config,
        num_mem,
        device
    ):
        super(ICAEQwen, self).__init__()
        self.qwen = AutoModelForCausalLM.from_pretrained(
            qwen_path, 
            cache_dir=cache_dir, 
            use_auth_token=use_auth_token,
            torch_dtype=torch.bfloat16,
        )
        self.qwen = get_peft_model(self.qwen, lora_config)
        self.qwen.eval()
        for param in self.qwen.parameters():
            param.requires_grad = False
        print(f"Total parameters of qwen: {sum(p.numel() for p in self.qwen.parameters())}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            qwen_path,
            cache_dir=cache_dir, 
            use_auth_token=use_auth_token,
        )
        print("qwen tokenizer loaded.")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        self.num_mem = num_mem
        self.memory_embeddings = nn.Parameter(torch.randn(1, num_mem, 3584, dtype=torch.bfloat16).to(device))
        self.ae_embedding = nn.Parameter(torch.randn(1, 1, 3584, dtype=torch.bfloat16).to(device))
        load_lora_parameters(self, lora_path)
        self.device = device

    def compress(self, text, text_tokens=None, output_path=None):
        if text_tokens is None:
            text_tokens = self.tokenizer(
                text, 
                truncation=True, 
                padding="max_length", 
                max_length=self.max_length, 
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids.to(self.device)
        else:
            text_tokens = text_tokens.to(self.device)
        text_tok_embeddings = self.qwen.get_input_embeddings()(text_tokens)
        memory_tok_embeddings = self.memory_embeddings.repeat(text_tok_embeddings.shape[0], 1, 1).to(self.device)
        encoder_input_embeddings = torch.cat((text_tok_embeddings, memory_tok_embeddings), dim=1)
        encoder_output = self.qwen(
            inputs_embeds=encoder_input_embeddings, 
            output_hidden_states=True
        ).hidden_states[-1]
        mem_vec = encoder_output[:, -self.num_mem:, :]

        return mem_vec

    def predict(self, mem_vec, max_new_tokens, prompt):
        end = False
        
        if prompt == "ae":
            prompt_embedding = self.ae_embedding
        else:
            prompt_tokens = self.tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                add_special_tokens=False
            ).to(self.device)
            prompt_embedding = self.qwen.get_input_embeddings()(prompt_tokens).to(self.device)

        input_embeddings = torch.cat((mem_vec, prompt_embedding), dim=1)

        generated_text = []
        for i in range(max_new_tokens):
            with self.qwen.disable_adapter():
                if i == 0:
                    output = self.qwen(inputs_embeds=input_embeddings)
                else:
                    output = self.qwen(
                        inputs_embeds=input_embeddings, 
                        past_key_values=past_key_values
                    )
            logits = output.logits
            past_key_values = output.past_key_values

            next_token = torch.argmax(logits[:, -1, :], dim=-1)

            if next_token == self.tokenizer.eos_token_id or next_token == torch.tensor([151643], device='cuda:0'):
                end = True
                break

            generated_text.append(next_token.item())
        
            input_tokens = next_token.unsqueeze(0)
            input_embeddings = self.qwen.get_input_embeddings()(input_tokens)

        generated_token_len = len(generated_text)
        generated_text = self.tokenizer.decode(
            generated_text, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        return generated_text, end, generated_token_len 