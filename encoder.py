import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig

class EncodingModel(nn.Module):
    def __init__(self, config):
        super(EncodingModel, self).__init__()
        self.config = config


        if self.config.model == 'qwen':

            self.encoder = AutoModel.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager" # Buộc dùng attention gốc, không dùng flash-attn

            )
            self.embedding_dim = self.encoder.config.hidden_size
        else:
            raise ValueError(f"Model '{self.config.model}' không được hỗ trợ. Vui lòng chọn 'qwen'.")


        if self.config.pattern in ['softprompt', 'hybridprompt']:
            self.word_embedding = self.encoder.get_input_embeddings()
            
            self.prompt_lens = self.config.prompt_len * self.config.prompt_num
            self.softprompt_encoder = nn.Embedding(self.prompt_lens, self.embedding_dim).to(self.encoder.device)
            
            self._init_prompt()
            
            self.prompt_ids = torch.LongTensor(list(range(self.prompt_lens))).to(self.encoder.device)

        self.info_nce_fc = nn.Linear(self.embedding_dim, self.embedding_dim).to(self.encoder.device)

    def _init_prompt(self):
        torch.nn.init.normal_(self.softprompt_encoder.weight, std=0.02)

    def embedding_input(self, input_ids):

        input_embedding = self.word_embedding(input_ids)
        prompt_embedding = self.softprompt_encoder(self.prompt_ids)

        for i in range(input_ids.size(0)):
            p = 0
            for j in range(input_ids.size(1)):
                if input_ids[i, j] == self.config.prompt_token_ids:
                    if p < self.prompt_lens:
                        input_embedding[i, j] = prompt_embedding[p]
                        p += 1
        return input_embedding

    def get_last_token_embedding(self, hidden_states, attention_mask):
 
        sequence_lengths = torch.sum(attention_mask, dim=1) - 1
        batch_range = torch.arange(hidden_states.size(0), device=hidden_states.device)
        
        last_token_embeddings = hidden_states[batch_range, sequence_lengths]
        return last_token_embeddings

    def forward(self, inputs, is_des=False):
        batch_size = inputs['ids'].size(0)
        attention_mask = inputs['mask']

        if self.config.pattern in ['softprompt', 'hybridprompt']:
            inputs_embeds = self.embedding_input(inputs['ids'])
            outputs = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        else:
            outputs = self.encoder(input_ids=inputs['ids'], attention_mask=attention_mask)
        
        outputs_words = outputs.last_hidden_state
        
        if self.config.pattern == 'marker':
            h1, t1 = [], []
            for i in range(batch_size):
                ids = inputs['ids'][i].cpu().numpy()
                h1_index = np.argwhere(ids == self.config.h_ids)
                t1_index = np.argwhere(ids == self.config.t_ids)
                
                h1.append(h1_index[0][0] if h1_index.size > 0 else 0)
                t1.append(t1_index[0][0] if t1_index.size > 0 else 0)
            
            h_state = outputs_words[torch.arange(batch_size), torch.tensor(h1)]
            t_state = outputs_words[torch.arange(batch_size), torch.tensor(t1)]
            
            final_embedding = (h_state + t_state) / 2
            return final_embedding
        
        # Các pattern còn lại ('cls', 'hardprompt', 'softprompt', etc.) đều dùng last token embedding
        else:
            if is_des:
                # Nếu là description, có thể lấy trung bình các token
                average_outputs_words = torch.mean(outputs_words, dim=1)
                return average_outputs_words
            else:
                # Lấy embedding của token cuối cùng
                last_token_embedding = self.get_last_token_embedding(outputs_words, attention_mask)
                return last_token_embedding
