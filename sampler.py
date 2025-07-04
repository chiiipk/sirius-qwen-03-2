import pickle
import random
import json, os
from transformers import AutoTokenizer # Dùng AutoTokenizer để linh hoạt với Qwen
import numpy as np

# Hàm get_tokenizer đã được cải tiến để tương thích với config object
def get_tokenizer(config):
    model_name = config.model_name if config.model == 'qwen' else config.bert_path
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"],
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Lưu pad_token_id vào config để data_loader.py có thể sử dụng
    config.pad_token_id = tokenizer.pad_token_id
    return tokenizer

class data_sampler_CFRL(object): # Đổi tên class để khớp với train.py

    def __init__(self, config, seed=None):
        self.config = config
        self.set_path(self.config) # set_path sẽ đọc từ config
        
        # Tạo đường dẫn cache file .pkl
        temp_name = [self.config.task_name, self.config.seed]
        file_name = "{}.pkl".format("-".join([str(x) for x in temp_name]))
        mid_dir = os.path.join(self.config.data_path, "_processed_cache")
        if not os.path.exists(mid_dir):
            os.makedirs(mid_dir, exist_ok=True)
        self.save_data_path = os.path.join(mid_dir, file_name)

        self.tokenizer = get_tokenizer(self.config)

        # Đọc dữ liệu relation từ file .json
        self.id2rel, self.rel2id = self._read_relations(self.config.relation_file)
        self.config.num_of_relation = len(self.id2rel)

        self.seed = seed
        self.set_seed(self.seed)

        # Đọc và xử lý dữ liệu từ file .json chính
        self.training_dataset, self.valid_dataset, self.test_dataset = self._read_data(self.config.data_file)
        
        # <<< PHẦN THÊM MỚI QUAN TRỌNG >>>
        # Đọc các mô tả quan hệ, một chức năng mà sampler gốc không có nhưng mô hình của bạn cần
        self.rel2des, self.id2des = self._read_descriptions(self.config.relation_description)
        self.seen_descriptions = {}
        # <<< KẾT THÚC PHẦN THÊM MỚI >>>

        self.batch = 0
        self.task_length = len(self.id2rel) // self.config.rel_per_task
        self.seen_relations = []
        self.history_test_data = {}

    def set_path(self, config):
        if config.task_name == 'FewRel':
            config.data_file = os.path.join(config.data_path, "data_with_marker.json")
            config.relation_file = os.path.join(config.data_path, "id2rel.json")
            # Trỏ đến file description mới theo yêu cầu của bạn
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

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch >= self.task_length:
            raise StopIteration()
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
            # <<< PHẦN THÊM MỚI QUAN TRỌNG >>>
            if index in self.id2des:
                self.seen_descriptions[relation_name] = self.id2des[index]
        
        # Trả về 7 giá trị, bao gồm cả seen_descriptions mà train.py cần
        return cur_training_data, cur_valid_data, cur_test_data, current_relations, self.history_test_data, self.seen_relations, self.seen_descriptions

    def _read_data(self, file):
        if os.path.isfile(self.save_data_path):
            print(f"Tải dữ liệu cache từ: {self.save_data_path}")
            with open(self.save_data_path, 'rb') as f: return pickle.load(f)
        
        print(f"Xử lý dữ liệu từ file JSON: {file}")
        data = json.load(open(file, 'r', encoding='utf-8'))
        train_d, val_d, test_d = [[] for _ in range(self.config.num_of_relation)], [[] for _ in range(self.config.num_of_relation)], [[] for _ in range(self.config.num_of_relation)]
        
        for relation in data.keys():
            rel_samples = data[relation]
            if self.seed is not None: random.seed(self.seed)
            random.shuffle(rel_samples)
            num_train = int(0.7 * len(rel_samples)); num_val = int(0.1 * len(rel_samples))
            
            for i, sample in enumerate(rel_samples):
                tokenized_sample = {
                    'relation': self.rel2id[sample['relation']],
                    'tokens': self.tokenizer.encode(' '.join(sample['tokens']), truncation=True, max_length=self.config.max_length)
                }
                if i < num_train: train_d[self.rel2id[relation]].append(tokenized_sample)
                elif i < num_train + num_val: val_d[self.rel2id[relation]].append(tokenized_sample)
                else: test_d[self.rel2id[relation]].append(tokenized_sample)
                
        with open(self.save_data_path, 'wb') as f: pickle.dump((train_d, val_d, test_d), f)
        return train_d, val_d, test_d

    def _read_relations(self, file):
        id2rel = json.load(open(file, 'r', encoding='utf-8'))
        return id2rel, {name: i for i, name in enumerate(id2rel)}

    # <<< HÀM THÊM MỚI QUAN TRỌNG >>>
    # Hàm này lấy logic đọc mô tả đúng nhất mà chúng ta đã tìm ra
    def _read_descriptions(self, file):
        rel2des, id2des = {}, {}
        try:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = line.split(None, 1)
                    if len(parts) >= 2:
                        rel_name, description = parts[0], parts[1]
                        if rel_name in self.rel2id:
                            id2des[self.rel2id[rel_name]] = [description]
                            rel2des[rel_name] = description
        except FileNotFoundError:
            print(f"CẢNH BÁO: Không tìm thấy file description tại {file}")
            
        for rel_id, rel_name in enumerate(self.id2rel):
            if rel_id not in id2des:
                print(f"CẢNH BÁO: Quan hệ '{rel_name}' (ID: {rel_id}) không có mô tả, dùng tên làm mặc định.")
                id2des[rel_id] = [rel_name]
                rel2des[rel_name] = rel_name
                
        return rel2des, id2des
