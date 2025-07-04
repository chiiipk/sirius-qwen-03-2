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

# File: encoder.py

# ... (các hàm __init__ và get_last_token_embedding giữ nguyên) ...

    # --- THAY THẾ TOÀN BỘ HÀM NÀY BẰNG PHIÊN BẢN MỚI ---
    def forward(self, inputs, is_des=False):
        batch_size = inputs['ids'].size(0)
        attention_mask = inputs['mask']

        # inputs['ids'] đã ở trên GPU, chúng ta sẽ giữ nó ở đó
        input_ids_gpu = inputs['ids']

        outputs = self.encoder(input_ids=input_ids_gpu, attention_mask=attention_mask)
        outputs_words = outputs.last_hidden_state
        
        if self.config.pattern == 'marker':
            # --- LOGIC MỚI (Vector hóa, chạy hoàn toàn trên GPU) ---
            # Tạo một mask boolean cho head và tail markers trên toàn bộ batch
            h_mask = (input_ids_gpu == self.config.h_ids)
            t_mask = (input_ids_gpu == self.config.t_ids)

            # Dùng argmax để tìm chỉ số của marker đầu tiên cho mỗi câu trong batch
            # .long() để đảm bảo kiểu dữ liệu là integer cho việc lấy chỉ số
            h1_indices = torch.argmax(h_mask.long(), dim=1)
            t1_indices = torch.argmax(t_mask.long(), dim=1)
            
            # Lấy hidden state tại vị trí của các marker một cách hiệu quả
            h_state = outputs_words[torch.arange(batch_size), h1_indices]
            t_state = outputs_words[torch.arange(batch_size), t1_indices]
            
            # Kết hợp embedding của head và tail
            final_embedding = (h_state + t_state) / 2
            return final_embedding
        
        else: 
            if is_des:
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs_words.size()).float()
                sum_embeddings = torch.sum(outputs_words * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return self.get_last_token_embedding(outputs_words, attention_mask)
        
        # --- Logic cho các trường hợp khác hoặc mặc định ---
        else: # 'cls' hoặc các pattern khác không dùng marker
            if is_des:
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs_words.size()).float()
                sum_embeddings = torch.sum(outputs_words * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return self.get_last_token_embedding(outputs_words, attention_mask)
