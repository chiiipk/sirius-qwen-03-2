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

        # Get the tokenizer to find marker IDs
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, 
            additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"],
            trust_remote_code=True
        )

        # Define marker IDs
        if self.config.pattern == 'marker':
            self.config.h_ids = self.tokenizer.convert_tokens_to_ids("[E11]")
            self.config.t_ids = self.tokenizer.convert_tokens_to_ids("[E21]")
            print(f"--- Marker IDs have been set: H_ID={self.config.h_ids}, T_ID={self.config.t_ids} ---")

    def get_last_token_embedding(self, hidden_states, attention_mask):
        sequence_lengths = torch.sum(attention_mask, dim=1) - 1
        batch_range = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_token_embeddings = hidden_states[batch_range, sequence_lengths]
        return last_token_embeddings

    def forward(self, inputs, is_des=False):
        batch_size = inputs['ids'].size(0)
        attention_mask = inputs['mask']
        input_ids_gpu = inputs['ids']

        outputs = self.encoder(input_ids=input_ids_gpu, attention_mask=attention_mask)
        outputs_words = outputs.last_hidden_state
        
        if self.config.pattern == 'marker':
            # Vectorized logic for markers on GPU
            h_mask = (input_ids_gpu == self.config.h_ids)
            t_mask = (input_ids_gpu == self.config.t_ids)

            h1_indices = torch.argmax(h_mask.long(), dim=1)
            t1_indices = torch.argmax(t_mask.long(), dim=1)
            
            h_state = outputs_words[torch.arange(batch_size), h1_indices]
            t_state = outputs_words[torch.arange(batch_size), t1_indices]
            
            final_embedding = (h_state + t_state) / 2
            return final_embedding
        else:
            # Logic for other patterns
            if is_des:
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs_words.size()).float()
                sum_embeddings = torch.sum(outputs_words * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return self.get_last_token_embedding(outputs_words, attention_mask)
