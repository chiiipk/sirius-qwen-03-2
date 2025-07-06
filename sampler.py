import pickle
import os 
import random
import numpy as np
from transformers import AutoTokenizer
import json
from nltk import word_tokenize # Cần cho việc tìm vị trí token

# Hàm get_tokenizer đã được cải tiến để tương thích với nhiều pattern
def get_tokenizer(config):
    model_name = config.model_name if config.model == 'qwen' else config.bert_path
    
    # Các token đặc biệt sẽ được dùng cho cả marker và hybridprompt
    special_tokens = ['[unused0]', '[unused1]', '[unused2]', '[unused3]']
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        additional_special_tokens=special_tokens,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.mask_token is None: tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    
    config.pad_token_id = tokenizer.pad_token_id
    config.mask_token_id = tokenizer.mask_token_id
    
    # Gán ID cho các pattern khác nhau
    if config.pattern == 'marker':
        config.h_ids = tokenizer.convert_tokens_to_ids(special_tokens[0])
        config.t_ids = tokenizer.convert_tokens_to_ids(special_tokens[2])
    elif config.pattern in ['softprompt', 'hybridprompt']:
        config.prompt_token_ids = tokenizer.convert_tokens_to_ids(special_tokens[0])

    return tokenizer

def _extract_entity_info(tokens, start_marker, end_marker):
    """Trích xuất text và vị trí của một thực thể từ danh sách token."""
    try:
        start_idx = tokens.index(start_marker)
        end_idx = tokens.index(end_marker)
        
        # Text của thực thể
        entity_text = " ".join(tokens[start_idx + 1:end_idx])
        
        # Vị trí của thực thể trong câu *sau khi đã loại bỏ các marker*
        # Đây là bước quan trọng để các pattern khác hoạt động đúng
        offset = 0
        if start_marker in ['[E11]', '[E21]']: offset += 1
        if start_marker == '[E21]': offset += 2 # Vì có [E11] và [E12] đứng trước
            
        clean_start = start_idx - offset
        clean_end = end_idx - offset - 1
        
        return entity_text, [[clean_start, clean_end]]

    except ValueError:
        return None, None

class data_sampler_CFRL(object):

    def __init__(self, config, seed=None):
        self.config = config
        self.set_path(self.config)
        self.tokenizer = get_tokenizer(self.config)

        # --- Bước 1: Đọc thông tin quan hệ TRƯỚC TIÊN ---
        self.id2rel, self.rel2id = self._read_relations(self.config.relation_file)
        self.config.num_of_relation = len(self.id2rel)
        
        # --- Bước 2: Bây giờ mới thiết lập seed, vì set_seed cần id2rel ---
        self.seed = seed
        self.set_seed(self.seed)
        
        # --- Các bước còn lại ---
        self.rel2des, self.id2des = self._read_descriptions(self.config.relation_description)
        self.seen_descriptions = {}
        
        mid_dir = os.path.join(self.config.data_path, "_processed_cache")
        if not os.path.exists(mid_dir): os.makedirs(mid_dir, exist_ok=True)
        file_name = f"{self.config.task_name}_{self.config.pattern}_{config.prompt_len if 'prompt' in config.pattern else ''}_{self.config.seed}.pkl"
        self.save_data_path = os.path.join(mid_dir, file_name)

        self.training_dataset, self.valid_dataset, self.test_dataset = self._read_data(self.config.data_file)
        
        self.batch = 0
        self.task_length = len(self.id2rel) // self.config.rel_per_task
        self.seen_relations = []
        self.history_test_data = {}


    def set_path(self, config):
        if config.task_name == 'FewRel':
            config.data_file = os.path.join(config.data_path, "data_with_marker.json")
            config.relation_file = os.path.join(config.data_path, "id2rel.json")
            config.relation_description = os.path.join(config.data_path, config.task_name, "relation_description_new.txt")
        elif config.task_name == 'TACRED':
            config.data_file = os.path.join(config.data_path, "data_with_marker_tacred.json")
            config.relation_file = os.path.join(config.data_path, "id2rel_tacred.json")
            config.relation_description = os.path.join(config.data_path, config.task_name, "relation_description.txt")

    def set_seed(self, seed):
        if seed is not None:
            self.seed = seed
            random.seed(self.seed)
        self.shuffle_index = list(range(len(self.id2rel)))
        random.shuffle(self.shuffle_index)

    def __iter__(self): return self

    def __next__(self):
        if self.batch >= self.task_length: raise StopIteration()
        indexs = self.shuffle_index[self.config.rel_per_task*self.batch : self.config.rel_per_task*(self.batch+1)]
        self.batch += 1
        current_relations, cur_training_data, cur_valid_data, cur_test_data = [], {}, {}, {}
        for index in indexs:
            relation_name = self.id2rel[index]
            current_relations.append(relation_name)
            self.seen_relations.append(relation_name)
            cur_training_data[relation_name] = self.training_dataset[index]
            cur_valid_data[relation_name] = self.valid_dataset[index]
            cur_test_data[relation_name] = self.test_dataset[index]
            self.history_test_data[relation_name] = self.test_dataset[index]
            if index in self.id2des: self.seen_descriptions[relation_name] = self.id2des[index]
        return cur_training_data, cur_valid_data, cur_test_data, current_relations, self.history_test_data, self.seen_relations, self.seen_descriptions

def _read_data(self, file):
    if os.path.isfile(self.save_data_path):
        print(f"Tải dữ liệu cache từ: {self.save_data_path}")
        with open(self.save_data_path, 'rb') as f: return pickle.load(f)
    
    print(f"Xử lý dữ liệu từ file JSON: {file}")
    data = json.load(open(file, 'r', encoding='utf-8'))
    train_d, val_d, test_d = [[] for _ in range(self.config.num_of_relation)], [[] for _ in range(self.config.num_of_relation)], [[] for _ in range(self.config.num_of_relation)]
    
    legacy_markers = ['[E11]', '[E12]', '[E21]', '[E22]']

    for relation in data.keys():
        rel_samples = data[relation]
        if self.seed is not None: random.seed(self.seed)
        random.shuffle(rel_samples)
        
        # Sửa lại logic chia tách cho nhất quán với Few-shot setting
        num_train = self.config.num_k 
        num_val = self.config.num_k # Thường tập val/test trong few-shot cũng nhỏ
        
        train_count, val_count, test_count = 0, 0, 0

        for sample in rel_samples:
            # ### SỬA LỖI: BẮT ĐẦU PHẦN LOGIC MỚI ###
            
            # 1. Trích xuất thông tin thực thể từ các marker trong `sample['tokens']`
            raw_tokens = sample['tokens']
            
            head_text, head_pos = _extract_entity_info(raw_tokens, '[E11]', '[E12]')
            tail_text, tail_pos = _extract_entity_info(raw_tokens, '[E21]', '[E22]')

            # Nếu không tìm thấy đủ marker trong câu, bỏ qua mẫu này
            if head_text is None or tail_text is None:
                continue

            # 2. Tạo một dictionary 'processed_sample' mới có cấu trúc chuẩn
            processed_sample = {
                'relation': self.rel2id[sample['relation']],
                # Tạo một list token sạch không chứa marker
                'tokens': [token for token in raw_tokens if token not in legacy_markers],
                'h': [head_text, 'ID_h', head_pos], # Cấu trúc [text, id, [[start, end]]]
                't': [tail_text, 'ID_t', tail_pos], # Cấu trúc [text, id, [[start, end]]]
            }
            # ### KẾT THÚC PHẦN LOGIC MỚI ###
            
            # 3. Giờ hàm tokenize sẽ nhận được đúng định dạng nó cần
            tokenized_sample = self.tokenize(processed_sample)
            
            # 4. Phân chia dữ liệu
            # Sử dụng num_k cho tập train, phần còn lại cho val/test
            if train_count < num_train:
                train_d[self.rel2id[relation]].append(tokenized_sample)
                train_count += 1
            # elif val_count < num_val:
            #     val_d[self.rel2id[relation]].append(tokenized_sample)
            #     val_count += 1
            else:
                test_d[self.rel2id[relation]].append(tokenized_sample)
                test_count += 1
            
    with open(self.save_data_path, 'wb') as f: pickle.dump((train_d, val_d, test_d), f)
    return train_d, val_d, test_d

    # --- CÁC HÀM TOKENIZE ĐA DẠNG ---
    def tokenize(self, sample):
        tokenized_sample = {'relation': sample['relation']}
        if self.config.pattern == 'hybridprompt':
            ids, mask = self._tokenize_hybridprompt(sample)                     
        elif self.config.pattern == 'marker':
            ids, mask = self._tokenize_marker(sample)
        else: # Mặc định là cls
            ids, mask = self._tokenize_cls(sample)            
        tokenized_sample['ids'], tokenized_sample['mask'] = ids, mask
        return tokenized_sample

    def _tokenize_template(self, prompt_text):
        tokenized = self.tokenizer(prompt_text, padding='max_length', truncation=True, max_length=self.config.max_length, return_tensors='pt')
        return tokenized['input_ids'].squeeze().tolist(), tokenized['attention_mask'].squeeze().tolist()

    def _tokenize_hybridprompt(self, sample):
        prompt_len = self.config.prompt_len
        text = ' '.join(sample['tokens'])
        h, t = sample['h'][0], sample['t'][0]
        prompt_seg = ' '.join(['[unused0]'] * prompt_len)
        prompt_text = f"{text} {prompt_seg} {h} {prompt_seg} {self.tokenizer.mask_token} {prompt_seg} {t} {prompt_seg}"
        return self._tokenize_template(prompt_text)

    def _tokenize_marker(self, sample):
        raw_tokens = sample['tokens']
        h_start, h_end = sample['h'][2][0][0], sample['h'][2][0][-1]
        t_start, t_end = sample['t'][2][0][0], sample['t'][2][0][-1]
        temp_tokens = list(raw_tokens)
        markers = sorted([(h_start, "[unused0]"), (h_end + 1, "[unused1]"), (t_start, "[unused2]"), (t_end + 1, "[unused3]")], key=lambda x: x[0], reverse=True)
        for index, marker_token in markers:
            temp_tokens.insert(index, marker_token)
        return self._tokenize_template(' '.join(temp_tokens))

    def _tokenize_cls(self, sample):
        return self._tokenize_template(' '.join(sample['tokens']))

    # --- CÁC HÀM ĐỌC DỮ LIỆU ---
    def _read_relations(self, file):
        id2rel = json.load(open(file, 'r', encoding='utf-8'))
        return id2rel, {name: i for i, name in enumerate(id2rel)}

    def _read_descriptions(self, file):
        rel2des, id2des = {}, {}
        try:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = line.split(None, 1)
                    if len(parts) >= 2 and parts[0] in self.rel2id:
                        id2des[self.rel2id[parts[0]]] = [parts[1]]
                        rel2des[parts[0]] = parts[1]
        except FileNotFoundError:
            print(f"CẢNH BÁO: Không tìm thấy file description tại {file}")
        for rel_id, rel_name in enumerate(self.id2rel):
            if rel_id not in id2des:
                print(f"CẢNH BÁO: Quan hệ '{rel_name}' không có mô tả, dùng tên làm mặc định.")
                id2des[rel_id] = [rel_name]
                rel2des[rel_name] = rel_name
        return rel2des, id2des
