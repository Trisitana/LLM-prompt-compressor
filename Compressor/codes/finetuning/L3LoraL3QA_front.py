import torch
import torch.nn as nn
from peft import get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
import os


def load_lora_parameters(model, lora_params_path):
    """
    Initialize the LoRA parameters.

    model (AutoModelForCausalLM): LLM with LoRA parameters.
    lora_params_path (str): Pretrained LoRA parameters path for initialization.
    """
    # load the pretrained LoRA parameters
    lora_params = torch.load(lora_params_path)

    # initialize the LoRA parameters and the compressed token in the LLM
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in lora_params:
                if 'lora' in name or 'memory_embeddings' in name:
                    param.copy_(lora_params[name])
                else:
                    print(f"No saved parameter for {name}")
            elif "lora" in name:
                print(f"No saved parameter for {name}")


class L3LoraL3QA(nn.Module):
    def __init__(self,
                 llama_path,
                 max_context_length,
                 lora_path,
                 lora_config,
                 num_mem,
                 device
                 ):
        """
        Create the compression model: LLM-LoRA + LLM.

        Args:
            llama_path (str): Path for the base LLM.
            max_context_length (int): Max number of context tokens to be compressed.
            lora_path (sttr): Pretrained LoRA parameters for initialization.
            lora_config (LoraConfig): LoRA configurations.
            num_mem (int): Number of compressed tokens.
            device (torch.device): CPU or GPU.
        """
        super(L3LoraL3QA, self).__init__()
        # load the original base LLaMA model
        llama = AutoModelForCausalLM.from_pretrained(
            llama_path,
            # cache path to save and load the LLM
            cache_dir="<to be filled>",
            # huggingface token to use the LLaMA model
            use_auth_token="<to be filled>",
            torch_dtype=torch.bfloat16,
        )
        # add LoRA parameters to the LLM
        self.llama = get_peft_model(llama, lora_config)
        # only LoRA parameters are trainable
        for name, param in self.llama.named_parameters():
            param.requires_grad = False
            if 'lora' in name:
                param.requires_grad = True
        print(f"Total parameters of llama: {sum(p.numel() for p in self.llama.parameters())}")
        # load the LLaMA tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llama_path, use_auth_token="<to be filled>")
        print("llama tokenizer loaded.")
        # set the padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # max number of context tokens to be compressed
        self.max_context_length = max_context_length
        # self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        # number of compressed tokens
        self.num_mem = num_mem
        # initialize the compressed token
        self.memory_embeddings = nn.Parameter(torch.randn(1, num_mem, 4096, dtype=torch.bfloat16).to(device))
        self.memory_embeddings.requires_grad = True
        # initialize the LoRA parameters
        load_lora_parameters(self, lora_path)
        self.device = device

    def forward(self, input_ids, labels):
        ####################
        # Encoder - llama+lora
        ####################
        # context tokens to be compressed
        text_tokens = input_ids[:, :self.max_context_length]
        # question and answer tokens
        qa_tokens = input_ids[:, self.max_context_length:]
        # target tokens: answer
        target_tokens = labels
        text_tok_embeddings = self.llama.get_input_embeddings()(text_tokens).to(self.device)
        memory_tok_embeddings = self.memory_embeddings.repeat(text_tok_embeddings.shape[0], 1, 1).to(self.device)
        # encoder input: context tokens + compressed tokens
        encoder_input_embeddings = torch.cat((text_tok_embeddings, memory_tok_embeddings), dim=1)
        encoder_output = self.llama(inputs_embeds=encoder_input_embeddings)
        # K V values for the encoder output
        past_key_values = encoder_output.past_key_values
        # K V values for the compressed tokens in the encoder output
        trimmed_past_key_values = tuple(
            (layer_key[:, :, -self.num_mem:, :], layer_value[:, :, -self.num_mem:, :])
            for layer_key, layer_value in past_key_values
        )

        ####################
        # Decoder - llama
        ####################
        # decoder input: QA tokens
        qa_tok_embeddings = self.llama.get_input_embeddings()(qa_tokens)
        decoder_input_embeddings = qa_tok_embeddings
        # decoder is the original base LLM without LoRA parameters
        with self.llama.disable_adapter():
            decoder_output = self.llama(inputs_embeds=decoder_input_embeddings, past_key_values=trimmed_past_key_values)
        all_logits = decoder_output.logits
        losses = self.criterion(all_logits.view(-1, all_logits.size(-1)), target_tokens.view(-1))
        losses = losses.view(target_tokens.shape)  # 恢复为[batch_size, seq_len]
        
        # 2. 创建权重掩码（初始为1）
        batch_size, seq_len = target_tokens.shape
        weight_mask = torch.ones_like(target_tokens, dtype=torch.float, device=self.device)
        
        # 3. 动态调整答案部分的权重
        for i in range(batch_size):
            # 找到答案开始位置（第一个非-100的token）
            non_pad_indices = (target_tokens[i] != -100).nonzero(as_tuple=True)[0]
            if len(non_pad_indices) > 0:
                answer_start = non_pad_indices[0].item()
                answer_length = len(non_pad_indices)
                
                # 为答案部分设置递减权重（开头权重高，尾部权重低）
                for pos in range(answer_start, answer_start + answer_length):
                    rel_pos = pos - answer_start  # 相对答案起始位置
                    # 线性衰减：从1.0降到0.5，可根据任务调整系数
                    decay_factor = 0.5  # 衰减系数，控制权重下降速度
                    min_weight = 0.3    # 最小权重
                    weight_mask[i, pos] = max(1.0 - (rel_pos / answer_length) * decay_factor, min_weight)
        
        # 4. 应用权重并计算最终损失
        weighted_losses = losses * weight_mask
        # 只计算非padding位置的损失（target != -100）
        non_pad_mask = (target_tokens != -100).float()
        loss = (weighted_losses * non_pad_mask).sum() / non_pad_mask.sum()


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


