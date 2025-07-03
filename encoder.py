import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer

class EncodingModel(nn.Module):
    def __init__(self, config):
        super(EncodingModel, self).__init__()
        self.config = config

        self.encoder = AutoModel.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"
        )
        self.embedding_dim = self.encoder.config.hidden_size

        # --- QUAN TRỌNG: Lấy Tokenizer đã được dạy các marker ---
        # Chúng ta cần tokenizer ở đây để lấy ID của các marker
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, 
            additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"],
            trust_remote_code=True
        )

        # --- KHÔI PHỤC: Định nghĩa các ID cho marker ---
        # Đây là bước kết nối quan trọng bị thiếu trước đây
        if self.config.pattern == 'marker':
            # Lấy ID của các token marker từ tokenizer
            self.config.h_ids = self.tokenizer.convert_tokens_to_ids("[E11]")
            self.config.t_ids = self.tokenizer.convert_tokens_to_ids("[E21]")
            print(f"--- Marker IDs đã được thiết lập: H_ID={self.config.h_ids}, T_ID={self.config.t_ids} ---")

    def get_last_token_embedding(self, hidden_states, attention_mask):
        sequence_lengths = torch.sum(attention_mask, dim=1) - 1
        batch_range = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_token_embeddings = hidden_states[batch_range, sequence_lengths]
        return last_token_embeddings

    def forward(self, inputs, is_des=False):
        batch_size = inputs['ids'].size(0)
        attention_mask = inputs['mask']

        outputs = self.encoder(input_ids=inputs['ids'], attention_mask=attention_mask)
        outputs_words = outputs.last_hidden_state
        
        # --- KHÔI PHỤC: Logic xử lý marker ---
        if self.config.pattern == 'marker':
            h1, t1 = [], []
            for i in range(batch_size):
                # Tìm vị trí của các ID marker trong chuỗi input
                ids = inputs['ids'][i].cpu().numpy()
                h1_index = np.argwhere(ids == self.config.h_ids)
                t1_index = np.argwhere(ids == self.config.t_ids)
                
                # Nếu tìm thấy, lấy vị trí đầu tiên. Nếu không, mặc định là 0 (CLS token)
                h1.append(h1_index[0][0] if h1_index.size > 0 else 0)
                t1.append(t1_index[0][0] if t1_index.size > 0 else 0)
            
            # Lấy hidden state tại vị trí của các marker
            h_state = outputs_words[torch.arange(batch_size), torch.tensor(h1)]
            t_state = outputs_words[torch.arange(batch_size), torch.tensor(t1)]
            
            # Kết hợp embedding của head và tail
            final_embedding = (h_state + t_state) / 2
            return final_embedding
        
        # --- Logic cho các trường hợp khác hoặc mặc định ---
        else: # 'cls' hoặc các pattern khác không dùng marker
            if is_des:
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs_words.size()).float()
                sum_embeddings = torch.sum(outputs_words * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return self.get_last_token_embedding(outputs_words, attention_mask)
