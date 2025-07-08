import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer

class EncodingModel(nn.Module):
    def __init__(self, config):
        super(EncodingModel, self).__init__()
        self.config = config
        

        special_tokens = ['[unused0]', '[unused1]', '[unused2]', '[unused3]']
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, 
            additional_special_tokens=special_tokens,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.mask_token is None: self.tokenizer.add_special_tokens({'mask_token': '[MASK]'})

        self.encoder = AutoModel.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"
        )
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        self.embedding_dim = self.encoder.config.hidden_size

        if self.config.pattern == 'marker':
            # Gán ID cho marker pattern
            self.config.h_ids = self.tokenizer.convert_tokens_to_ids("[unused0]")
            self.config.t_ids = self.tokenizer.convert_tokens_to_ids("[unused2]")

        elif self.config.pattern in ['softprompt', 'hybridprompt']:
            self.config.prompt_token_ids = self.tokenizer.convert_tokens_to_ids("[unused0]")
            
            self.word_embedding = self.encoder.get_input_embeddings()
            
            self.prompt_lens = self.config.prompt_len * self.config.prompt_num
            self.softprompt_encoder = nn.Embedding(self.prompt_lens, self.embedding_dim).to(self.encoder.device)
            
            torch.nn.init.normal_(self.softprompt_encoder.weight, std=0.02)
            self.prompt_ids = torch.LongTensor(list(range(self.prompt_lens))).to(self.encoder.device)

    def embedding_input(self, input_ids):
        inputs_embeds = self.word_embedding(input_ids)
        prompt_embeds = self.softprompt_encoder(self.prompt_ids)
        
        p = 0
        for i in range(input_ids.size(0)):
            for j in range(input_ids.size(1)):
                if input_ids[i, j] == self.config.prompt_token_ids and p < self.prompt_lens:
                    inputs_embeds[i, j] = prompt_embeds[p]
                    p += 1
        return inputs_embeds

    def get_last_token_embedding(self, hidden_states, attention_mask):
        sequence_lengths = torch.sum(attention_mask, dim=1) - 1
        return hidden_states[torch.arange(hidden_states.size(0)), sequence_lengths]

    def forward(self, inputs, is_des=False):
        batch_size = inputs['ids'].size(0)
        attention_mask = inputs['mask']
        input_ids_gpu = inputs['ids']


        if self.config.pattern in ['softprompt', 'hybridprompt'] and not is_des:
            inputs_embeds = self.embedding_input(input_ids_gpu)
            outputs = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            outputs = self.encoder(input_ids=input_ids_gpu, attention_mask=attention_mask)
        
        outputs_words = outputs.last_hidden_state

        if self.config.pattern == 'marker' and not is_des:
            h_mask, t_mask = (input_ids_gpu == self.config.h_ids), (input_ids_gpu == self.config.t_ids)
            h_indices, t_indices = torch.argmax(h_mask.long(), dim=1), torch.argmax(t_mask.long(), dim=1)
            h_state = outputs_words[torch.arange(batch_size), h_indices]
            t_state = outputs_words[torch.arange(batch_size), t_indices]
            return (h_state + t_state) / 2
        
        elif self.config.pattern == 'hybridprompt' and not is_des:
            mask_token_id = self.config.mask_token_id
            mask_indices = torch.argmax((input_ids_gpu == mask_token_id).long(), dim=1)
            return outputs_words[torch.arange(batch_size), mask_indices]
        
        else:
            if is_des: # Nếu là câu mô tả, lấy trung bình các token
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs_words.size()).float()
                sum_embeddings = torch.sum(outputs_words * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                return sum_embeddings / sum_mask
            else: # Lấy embedding của token cuối cùng (hoặc [CLS] nếu là BERT)
                return self.get_last_token_embedding(outputs_words, attention_mask)
