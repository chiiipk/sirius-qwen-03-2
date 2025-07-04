import pickle
import random
import json, os
from transformers import AutoTokenizer
import numpy as np

def get_tokenizer(config):
    if config.model == 'qwen':
        model_path = config.model_name
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"],
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else: # Bert
        model_path = config.bert_path
        tokenizer = AutoTokenizer.from_pretrained(model_path, additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"])
    
    config.pad_token_id = tokenizer.pad_token_id
    return tokenizer

class data_sampler_CFRL(object):
    def __init__(self, config, seed=None):
        self.config = config
        self.set_path(self.config)
        
        temp_name = [self.config.task_name, self.config.seed]
        file_name = "{}.pkl".format("-".join([str(x) for x in temp_name]))
        mid_dir = os.path.join(self.config.data_path, "_processed_cache")
        if not os.path.exists(mid_dir):
            os.makedirs(mid_dir, exist_ok=True)
        self.save_data_path = os.path.join(mid_dir, file_name)

        self.tokenizer = get_tokenizer(self.config)
        self.id2rel, self.rel2id = self._read_relations(self.config.relation_file)
        self.config.num_of_relation = len(self.id2rel)

        self.seed = seed
        self.set_seed(self.seed)

        self.training_dataset, self.valid_dataset, self.test_dataset = self._read_data(self.config.data_file)
        self.batch = 0
        self.task_length = len(self.id2rel) // self.config.rel_per_task
        self.seen_relations = []
        self.history_test_data = {}
        self.seen_descriptions = {}
        self.rel2des, self.id2des = self._read_descriptions(self.config.relation_description)


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
            if index in self.id2des: self.seen_descriptions[relation_name] = self.id2des[index]
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
            num_train, num_val = int(0.7 * len(rel_samples)), int(0.1 * len(rel_samples))
            for i, sample in enumerate(rel_samples):
                tokenized_sample = {'relation': self.rel2id[sample['relation']], 'tokens': self.tokenizer.encode(' '.join(sample['tokens']), truncation=True, max_length=self.config.max_length)}
                if i < num_train: train_d[self.rel2id[relation]].append(tokenized_sample)
                elif i < num_train + num_val: val_d[self.rel2id[relation]].append(tokenized_sample)
                else: test_d[self.rel2id[relation]].append(tokenized_sample)
        with open(self.save_data_path, 'wb') as f: pickle.dump((train_d, val_d, test_d), f)
        return train_d, val_d, test_d

    def _read_relations(self, file):
        id2rel = json.load(open(file, 'r', encoding='utf-8'))
        return id2rel, {name: i for i, name in enumerate(id2rel)}

    def _read_descriptions(self, file):
        rel2des, id2des = {}, {}
        try:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:  # Bỏ qua các dòng trống
                        continue

                    # --- PHẦN SỬA LỖI QUAN TRỌNG NHẤT ---
                    # Tách dòng thành 3 phần: ID, Tên, và Mô tả (toàn bộ phần còn lại)
                    # maxsplit=2 sẽ tách tối đa 2 lần, đảm bảo mô tả chứa dấu cách không bị vỡ
                    parts = line.split(maxsplit=2)
                    
                    if len(parts) >= 3:
                        # parts[0] là ID (ví dụ: 'NA'), parts[1] là tên (ví dụ: 'P1411')
                        rel_name = parts[1]
                        description = parts[2]
                        
                        if rel_name in self.rel2id:
                            rel_id = self.rel2id[rel_name]
                            rel2des[rel_name] = description
                            id2des[rel_id] = [description] # Giữ định dạng là list
                    elif len(parts) == 2:
                        # Trường hợp phòng thủ: Nếu dòng chỉ có ID và tên, không có mô tả
                        rel_name = parts[1]
                        if rel_name in self.rel2id:
                            print(f"CẢNH BÁO: Quan hệ '{rel_name}' có trong file nhưng thiếu mô tả. Sử dụng tên làm mô tả mặc định.")
                            rel_id = self.rel2id[rel_name]
                            rel2des[rel_name] = rel_name
                            id2des[rel_id] = [rel_name]

        except FileNotFoundError:
            print(f"CẢNH BÁO: Không tìm thấy file description tại {file}")

        for rel_id, rel_name in self.id2rel.items():
            if rel_id not in id2des:
                print(f"CẢNH BÁO: Quan hệ '{rel_name}' (ID: {rel_id}) không được tìm thấy trong file mô tả. Sử dụng tên làm mô tả mặc định.")
                id2des[rel_id] = [rel_name]
                rel2des[rel_name] = rel_name

        return rel2des, id2des

