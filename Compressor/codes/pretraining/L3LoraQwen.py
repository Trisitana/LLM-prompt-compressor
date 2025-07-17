import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model

class L3LoraQwen(nn.Module):
    def __init__(
        self,
        qwen_path,
        max_length,
        lora_config,
        num_mem,
        device
    ):
        """
        Create the compression model: Qwen-LoRA + Qwen.

        Args:
            qwen_path (str): Path for the base Qwen model.
            max_length (int): Max number of tokens to be compressed.
            lora_config (LoraConfig): LoRA configurations.
            num_mem (int): Number of compressed tokens.
            device (torch.device): CPU or GPU.
        """
        super(L3LoraQwen, self).__init__()
        # load the original base Qwen model
        qwen = AutoModelForCausalLM.from_pretrained(
            qwen_path, 
            # cache path to save the Qwen model
            cache_dir="<to be filled>", 
            # huggingface token to use the Qwen model
            use_auth_token="<to be filled>",
            torch_dtype=torch.bfloat16,
        )
        # add LoRA parameters to the LLM
        self.qwen = get_peft_model(qwen, lora_config)
        # only LoRA parameters are trainable
        for name, param in self.qwen.named_parameters():
            param.requires_grad = False
            if 'lora' in name:
                param.requires_grad = True
        print(f"Total parameters of qwen: {sum(p.numel() for p in self.qwen.parameters())}")
        # load the Qwen tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_path, use_auth_token="<to be filled>")
        print("qwen tokenizer loaded.")
        # set the padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Create a special trigger token embedding that will be trained
        self.trigger_embedding = nn.Parameter(torch.randn(1, 1, 3584, dtype=torch.bfloat16).to(device))
        self.trigger_embedding.requires_grad = True
        # max number of tokens to be compressed
        self.max_length = max_length
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        # number of compressed tokens
        self.num_mem = num_mem
        # compressed token - using Qwen's hidden size (3584 for Qwen2.5-7B)
        self.memory_embeddings = nn.Parameter(torch.randn(1, num_mem, 3584, dtype=torch.bfloat16).to(device))
        self.memory_embeddings.requires_grad = True
        self.device = device

    def forward(self, input_ids, labels):
        ####################
        # Encoder - qwen+lora
        ####################
        # input text tokens to be compressed
        text_tokens = input_ids
        # target tokens: input text tokens + EOS token
        target_tokens = labels
        text_tok_embeddings = self.qwen.get_input_embeddings()(text_tokens).to(self.device)
        # compressed tokens
        memory_tok_embeddings = self.memory_embeddings.repeat(text_tok_embeddings.shape[0], 1, 1).to(self.device)
        # encoder input: text tokens + compressed tokens
        encoder_input_embeddings = torch.cat((text_tok_embeddings, memory_tok_embeddings), dim=1)
        encoder_output = self.qwen(inputs_embeds=encoder_input_embeddings)
        # get the K V values for the encoder output
        past_key_values = encoder_output.past_key_values
        # get the K V values for the compressed tokens
        trimmed_past_key_values = tuple(
            (layer_key[:, :, -self.num_mem:, :], layer_value[:, :, -self.num_mem:, :]) 
            for layer_key, layer_value in past_key_values
        )

        ####################
        # Decoder - qwen
        ####################
        # Use the trainable trigger token instead of BOS
        trigger_embeddings = self.trigger_embedding.repeat(text_tok_embeddings.shape[0], 1, 1)

        # decoder input: trigger token + text tokens
        decoder_input_embeddings = torch.cat((trigger_embeddings, text_tok_embeddings), dim=1)
        # use the original LLM without LoRA parameters
        with self.qwen.disable_adapter():
            decoder_output = self.qwen(inputs_embeds=decoder_input_embeddings, past_key_values=trimmed_past_key_values)
        # logits for the decoder output
        all_logits = decoder_output.logits

        # target output: text tokens + EOS token
        # calculate the cross entropy
        loss = self.criterion(all_logits.view(-1, all_logits.size(-1)), target_tokens.view(-1))

        return {'loss': loss, 'logits': all_logits}

    def save_lora_parameters(self, save_directory):
        """
        Save LoRA parameters, memory embeddings and trigger token embedding.
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        # Save LoRA parameters, memory embeddings and trigger token
        lora_params_path = os.path.join(save_directory, "lora_params.pth")

        lora_params = {name: param for name, param in self.named_parameters() if 'lora' in name}
        lora_params['memory_embeddings'] = self.memory_embeddings
        lora_params['trigger_embedding'] = self.trigger_embedding  # Add trigger token embedding
        torch.save(lora_params, lora_params_path)
        
        print(f"LoRA parameters, memory embeddings and trigger token saved to {lora_params_path}")