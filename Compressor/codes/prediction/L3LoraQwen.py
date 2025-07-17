import torch
import torch.nn as nn
from peft import get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_lora_parameters(model, lora_params_path):
    """
    Initialize the LoRA parameters.

    model (AutoModelForCausalLM): LLM with LoRA parameters.
    lora_params_path (str): LoRA parameters path for initialization.
    """
    # load the LoRA parameters
    lora_params = torch.load(lora_params_path)

    # initialize the LoRA parameters and the compressed token in the LLM
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in lora_params: 
                if 'lora' in name or 'memory_embeddings' in name or 'trigger_embedding' in name:
                    param.copy_(lora_params[name])
                    if 'memory_embeddings' in name:
                        print("Find memory_embeddings!")
                    if "trigger_embedding" in name:
                        print("Find trigger_embedding!")
                else:
                    print(f"No saved parameter for {name}")
            elif "lora" in name:
                print(f"No saved parameter for {name}")

class L3LoraQwen(nn.Module):
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
        """
        Create the compression model: Qwen-LoRA + Qwen.

        Args:
            qwen_path (str): Path for the base Qwen model.
            cache_dir (str): Cache path to save or load the Qwen model.
            use_auth_token (str): Huggingface token to use the Qwen model.
            max_length (int): Max number of context tokens to be compressed (truncation or padding).
            lora_path (str): Pretrained LoRA parameters for initialization.
            lora_config (LoraConfig): LoRA configurations.
            num_mem (int): Number of compressed tokens.
            device (torch.device): CPU or GPU.
        """
        super(L3LoraQwen, self).__init__()
        # load the original base Qwen model
        self.qwen = AutoModelForCausalLM.from_pretrained(
            qwen_path, 
            cache_dir=cache_dir, 
            use_auth_token=use_auth_token,
            torch_dtype=torch.bfloat16,
        )
        # add LoRA parameters to the LLM
        self.qwen = get_peft_model(self.qwen, lora_config)
        # all the parameters are not trainable
        self.qwen.eval()
        for param in self.qwen.parameters():
            param.requires_grad = False
        print(f"Total parameters of qwen: {sum(p.numel() for p in self.qwen.parameters())}")
        # load the Qwen tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            qwen_path, 
            cache_dir=cache_dir, 
            use_auth_token=use_auth_token,
        )
        print("qwen tokenizer loaded.")
        # set the padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # max number of context tokens to be compressed (truncation or padding)
        self.max_length = max_length
        # number of compressed tokens
        self.num_mem = num_mem
        # create the compressed token
        self.memory_embeddings = nn.Parameter(torch.randn(1, num_mem, 3584, dtype=torch.bfloat16).to(device))
        # create the trigger token embedding
        self.trigger_embedding = nn.Parameter(torch.randn(1, 1, 3584, dtype=torch.bfloat16).to(device))
        # initialize the LoRA parameters and the compressed token
        load_lora_parameters(self, lora_path)
        self.device = device

    def compress(self, text, text_tokens=None, output_path=None):
        """
        Compress the context into compressed tokens.

        Args:
            text (List[str]): context to be compressed.
            text_token (torch.tensor): context tokens.
            output_path (str): Path to save the K V values for the compressed tokens.

        Returns:
            trimmed_past_key_values (Tuple[Tuple[torch.Tensor, torch.Tensor], ...]): K V values for the compressed tokens.
        """
        # If the input is not context token
        # use the tokenizer to tokenize the context
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
        # initialize compressed tokens
        memory_tok_embeddings = self.memory_embeddings.repeat(text_tok_embeddings.shape[0], 1, 1).to(self.device)
        
        # encoder input: original context + compressed tokens
        encoder_input_embeddings = torch.cat((text_tok_embeddings, memory_tok_embeddings), dim=1)
        encoder_output = self.qwen(inputs_embeds=encoder_input_embeddings)
        
        # K V values for the encoder output
        past_key_values = encoder_output.past_key_values
        # K V values for the compressed tokens in the encoder output
        trimmed_past_key_values = tuple(
            (layer_key[:, :, -self.num_mem:, :], layer_value[:, :, -self.num_mem:, :]) 
            for layer_key, layer_value in past_key_values
        )

        # save the K V values for the compressed tokens
        if output_path is not None:
            torch.save(trimmed_past_key_values, output_path)
            print(f"Saved compressed past_key_values to {output_path}")

        return trimmed_past_key_values

    def predict(self, past_key_values, max_new_tokens, prompt):
        """
        Regenerate the compressed text or do QA based on the compressed tokens.

        Args:
            past_key_values (Tuple[Tuple[torch.Tensor, torch.Tensor], ...]): K V values for the compressed tokens.
            max_new_tokens (int): Maximum number of new tokens to generate.
            prompt (str): Prompt text.
        
        Returns:
            generated_text (str): Generated text. 
        """
        # whether the model ends automatically
        end = False

        # input prompt tokens: 
        # trigger token for regenerating the compressed text
        # question tokens for QA
        if prompt == "bos":
            # Use trigger embedding instead of BOS token
            input_embeddings = self.trigger_embedding.repeat(past_key_values[0][0].size(0), 1, 1)
        else:
            prompt_tokens = self.tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                add_special_tokens=False
            )
            prompt_tokens = prompt_tokens * past_key_values[0][0].size(0)
            input_tokens = torch.tensor(prompt_tokens, device=self.device)
            input_embeddings = self.qwen.get_input_embeddings()(input_tokens)

        generated_text = []
        for _ in range(max_new_tokens):
            # predict the next new token by the original LLM
            with self.qwen.disable_adapter():
                output = self.qwen(
                    inputs_embeds=input_embeddings, 
                    past_key_values=past_key_values
                )
            logits = output.logits
            past_key_values = output.past_key_values

            # choose the token id with the highest probability
            next_token = torch.argmax(logits[:, -1, :], dim=-1)

            # stop generating new tokens when meet end tokens
            if next_token == self.tokenizer.eos_token_id or next_token == torch.tensor([151643], device='cuda:0'):
                end = True
                break

            # add the new token to generated tokens
            generated_text.append(next_token.item())
        
            # update the input token
            input_tokens = next_token.unsqueeze(0)
            input_embeddings = self.qwen.get_input_embeddings()(input_tokens)

        # length of the generated tokens
        generated_token_length = len(generated_text)
        generated_text = self.tokenizer.decode(
            generated_text, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        return generated_text, end, generated_token_length 