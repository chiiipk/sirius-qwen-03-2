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





class data_sampler_CFRL(object):

    # Trong file sampler.py, thay thế hàm __init__ cũ bằng hàm này

# Trong file sampler.py, thay thế __init__ cũ bằng __init__ này

    def __init__(self, config, seed=None):
        self.config = config
        self.seed = seed

        # BƯỚC 1: Tự động cấu hình MỌI THỨ dựa trên task_name
        # Hàm này sẽ thiết lập các đường dẫn, rel_per_task, và task_length
        self._configure_for_current_task()

        # BƯỚC 2: Tải tokenizer, vì bây giờ mọi config đã sẵn sàng
        self.tokenizer = get_tokenizer(self.config)

        # BƯỚC 3: Đọc các file dữ liệu cốt lõi
        self.id2rel, self.rel2id = self._read_relations(self.config.relation_file)
        self.config.num_of_relation = len(self.id2rel)
        
        self.rel2des, self.id2des = self._read_descriptions(self.config.relation_description)
        
        # BƯỚC 4: Thiết lập seed và thứ tự tác vụ ngẫu nhiên
        # set_seed giờ đây có thể dùng self.config.task_length đã được tính
        self.set_seed(self.seed)
        
        # BƯỚC 5: Đọc và xử lý dữ liệu chính
        save_data_path = self._temp_datapath()
        self.training_dataset, self.valid_dataset, self.test_dataset = self._read_data(
            self.config.json_data_file,
            save_data_path
        )
        
        # BƯỚC 6: Khởi tạo các biến để lặp qua các tác vụ
        self.batch = 0
        # self.task_length không cần tính lại ở đây nữa
        self.seen_relations = []
        self.history_test_data = {}
        self.seen_descriptions = {}
    # Trong file sampler.py, bên trong lớp data_sampler_CFRL

    def _configure_for_current_task(self):
        """
        ### HÀM MỚI QUAN TRỌNG NHẤT ###
        Tự động cấu hình tất cả các tham số dựa trên task_name:
        - Đường dẫn file
        - Số quan hệ mỗi tác vụ
        - Tổng số tác vụ
        """
        task_name = self.config.task_name
        data_root = self.config.data_root
        
        print(f"--- Đang tự động cấu hình cho dataset: {task_name} ---")
        
        if task_name == 'FewRel':
            data_suffix = ''
            total_relations = 80
            # Theo bài báo, để có 10 tác vụ, mỗi tác vụ phải có 8 quan hệ
            self.config.rel_per_task = 8
            
        elif task_name == 'TACRED':
            data_suffix = '_tacred'
            total_relations = 40
            # Theo bài báo, để có 10 tác vụ, mỗi tác vụ phải có 4 quan hệ
            self.config.rel_per_task = 4
            
        else:
            raise ValueError(f"Dataset '{task_name}' không được hỗ trợ để cấu hình tự động.")

        # Tính toán và ghi đè task_length vào config
        self.config.task_length = total_relations // self.config.rel_per_task
        print(f" -> Số quan hệ mỗi tác vụ được đặt là: {self.config.rel_per_task}")
        print(f" -> Tổng số tác vụ được tính toán: {self.config.task_length}")

        # Gán các đường dẫn file vào config
        self.config.json_data_file = os.path.join(data_root, f"data_with_marker{data_suffix}.json")
        self.config.relation_file = os.path.join(data_root, f"id2rel{data_suffix}.json")
        self.config.relation_description = os.path.join(data_root, task_name, "relation_description.txt")

    def set_path(self, config):
        if config.task_name == 'FewRel':
            config.data_file = os.path.join(config.data_path, "data_with_marker.json")
            config.relation_file = os.path.join(config.data_path, "id2rel.json")
            config.relation_description = os.path.join(config.data_path, config.task_name, "relation_description_new.txt")
        elif config.task_name == 'TACRED':
            config.data_file = os.path.join(config.data_path, "data_with_marker_tacred.json")
            config.relation_file = os.path.join(config.data_path, "id2rel_tacred.json")
            config.relation_description = os.path.join(config.data_path, config.task_name, "relation_description.txt")

    # Dán đoạn code này vào bên trong class data_sampler_CFRL trong file sampler.py
    # Trong file sampler.py, bên trong lớp data_sampler_CFRL

    def _configure_paths(self):
        """
        Tự động tạo các đường dẫn file cần thiết dựa trên task_name
        và gán chúng vào đối tượng config.
        """
        task_name = self.config.task_name
        data_root = self.config.data_root # Đọc từ config.ini
        
        print(f"Đang cấu hình đường dẫn cho dataset: {task_name}")
        
        if task_name == 'FewRel':
            data_suffix = ''
        elif task_name == 'TACRED':
            data_suffix = '_tacred'
        else:
            raise ValueError(f"Dataset '{task_name}' không được hỗ trợ.")
            
        # ### SỬA LỖI ###: Gán các đường dẫn đã tạo vào chính self.config
        self.config.json_data_file = os.path.join(data_root, f"data_with_marker{data_suffix}.json")
        self.config.relation_file = os.path.join(data_root, f"id2rel{data_suffix}.json")
        self.config.relation_description = os.path.join(data_root, task_name, "relation_description.txt")
        
        print(f" - Data file được xác định: {self.config.json_data_file}")
        print(f" - Relation file được xác định: {self.config.relation_file}")
        print(f" - Description file được xác định: {self.config.relation_description}")
    def _temp_datapath(self):
        """
        Tạo đường dẫn file cache để lưu dữ liệu đã được xử lý.
        Tên file sẽ bao gồm tên task, pattern, và seed để đảm bảo mỗi
        cấu hình thử nghiệm có một file cache riêng.
        """
        # Tạo thư mục cache chính nếu chưa có
        cache_root = os.path.join('data_cache')
        if not os.path.exists(cache_root):
            os.makedirs(cache_root, exist_ok=True)

        # Tạo thư mục con cho từng task
        task_dir = os.path.join(cache_root, self.config.task_name)
        if not os.path.exists(task_dir):
            os.makedirs(task_dir, exist_ok=True)
            
        # Tạo tên file cache dựa trên các tham số
        # Ví dụ: FewRel_hybridprompt_k5_seed2021_full_data.pkl
        file_name = (
            f"{self.config.task_name}_"
            f"{self.config.pattern}_"
            f"k{self.config.num_k}_" # Thêm num_k vào để phân biệt các lần chạy few-shot
            f"seed{self.seed}_"
            f"full_data.pkl" # Đánh dấu đây là dữ liệu được chia theo tỷ lệ
        )
        
        save_path = os.path.join(task_dir, file_name)
        print(f"Đường dẫn file cache được tạo: {save_path}")
        return save_path

    def _extract_entity_info(self, tokens, start_marker, end_marker): 
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

    def set_seed(self, seed):
        if seed is not None:
            self.seed = seed
            random.seed(self.seed)
        self.shuffle_index = list(range(len(self.id2rel)))
        random.shuffle(self.shuffle_index)

    def __iter__(self): return self

    def __next__(self):
        if self.batch >= self.config.task_length: raise StopIteration()
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

# Trong class data_sampler_CFRL của file sampler.py
    # Dán đoạn code này vào bên trong class data_sampler_CFRL trong file sampler.py

    def _read_data(self, json_file_path, save_data_path):
        # ### THAY ĐỔI LỚN: Tên file cache giờ sẽ phản ánh chiến lược chia tách mới ###
        # Thêm hậu tố "_full" để phân biệt với cache của few-shot
        save_data_path = save_data_path.replace(".pkl", "_full_data.pkl")
    
        if os.path.isfile(save_data_path):
            with open(save_data_path, 'rb') as f:
                datas = pickle.load(f)
                print(f"Đã tải dữ liệu (full) đã xử lý từ cache: {save_data_path}")
            return datas['train'], datas['valid'], datas['test']
        
        print(f"Cache không tồn tại. Đang xử lý dữ liệu (full) từ file JSON: {json_file_path}")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        train_dataset = [[] for _ in range(self.config.num_of_relation)]
        val_dataset = [[] for _ in range(self.config.num_of_relation)]
        test_dataset = [[] for _ in range(self.config.num_of_relation)]
        
        legacy_markers = ['[E11]', '[E12]', '[E21]', '[E22]']
        
        def process_and_append(target_dataset, source_samples):
                    for sample in source_samples:
                        # Logic trích xuất thông tin thực thể (giữ nguyên từ trước)
                        raw_tokens = sample['tokens']
                        head_text, head_pos = self._extract_entity_info(raw_tokens, '[E11]', '[E12]')
                        tail_text, tail_pos = self._extract_entity_info(raw_tokens, '[E21]', '[E22]')
        
                        if head_text is None or tail_text is None: continue
        
                        processed_sample = {
                            'relation': relation_id,
                            'tokens': [token for token in raw_tokens if token not in legacy_markers],
                            'h': [head_text, 'ID_h', head_pos],
                            't': [tail_text, 'ID_t', tail_pos],
                        }
                        tokenized = self.tokenize(processed_sample)
                        target_dataset[relation_id].append(tokenized)
                        
        for relation_name, samples in raw_data.items():
            if self.seed is not None:
                random.seed(self.seed)
            random.shuffle(samples)
            
            relation_id = self.rel2id.get(relation_name)
            if relation_id is None: continue
    
            # ### THAY ĐỔI CỐT LÕI: Chia dữ liệu theo tỷ lệ, không còn dùng num_k ###
            n_samples = len(samples)
            n_train = int(n_samples * 0.7)  # 70% cho training
            n_val = int(n_samples * 0.1)    # 10% cho validation
            
            train_samples = samples[:n_train]
            val_samples = samples[n_train : n_train + n_val]
            test_samples = samples[n_train + n_val:]
            
            print(f"Quan hệ '{relation_name}': {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test.")
    
            # Hàm helper để tránh lặp code

        
                # Xử lý và thêm vào các tập dữ liệu tương ứng
            process_and_append(train_dataset, train_samples)
            process_and_append(val_dataset, val_samples)
            process_and_append(test_dataset, test_samples)
    
        # Lưu vào cache
        datas_to_save = {'train': train_dataset, 'valid': val_dataset, 'test': test_dataset}
        with open(save_data_path, 'wb') as f:
            pickle.dump(datas_to_save, f)
            print(f"Đã lưu dữ liệu (full) đã xử lý vào cache: {save_data_path}")
    
        return train_dataset, val_dataset, test_dataset

# Bạn cũng cần đảm bảo có hàm _extract_entity_info trong class hoặc bên ngoài

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
#{'relation': 10, 'ids': [101, ...], 'mask': [1, ...]}
    # --- CÁC HÀM ĐỌC DỮ LIỆU ---
    def _read_relations(self, file_path):
        """
        Đọc file JSON chứa danh sách các quan hệ.
        File này có định dạng là một list các tên quan hệ.
        Ví dụ: ["org:founded_by", "per:schools_attended", ...]
        """
        print(f"Đang đọc file quan hệ từ: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                id2rel_list = json.load(f)
            
            # Tạo dictionary id -> relation name
            id2rel_dict = {i: name for i, name in enumerate(id2rel_list)}
            
            # Tạo dictionary relation name -> id
            rel2id_dict = {name: i for i, name in enumerate(id2rel_list)}
            
            return id2rel_dict, rel2id_dict
            
        except FileNotFoundError:
            print(f"LỖI NGHIÊM TRỌNG: Không tìm thấy file quan hệ tại '{file_path}'. Vui lòng kiểm tra lại đường dẫn trong config.ini.")
            # Thoát chương trình vì đây là file bắt buộc
            exit()
        except json.JSONDecodeError:
            print(f"LỖI NGHIÊM TRỌNG: File '{file_path}' không phải là file JSON hợp lệ.")
            exit()

# Trong file sampler.py, bên trong lớp data_sampler_CFRL

    def _read_descriptions(self, file_path):
        """
        Đọc file mô tả quan hệ. Hàm này có khả năng xử lý hai định dạng khác nhau
        cho FewRel và TACRED một cách tự động.
        """
        print(f"Đang đọc file mô tả từ: {file_path}")
        rel2des, id2des = {}, {}
        task_name = self.config.task_name

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split('\t')
                    rel_name = ""
                    description = ""

                    # --- LOGIC PHÂN NHÁNH DỰA TRÊN TASK_NAME ---
                    if task_name == 'FewRel':
                        # Định dạng: "ID <tab> Tên đầy đủ <tab> Mô tả" hoặc "ID <tab> Mô tả"
                        if len(parts) >= 2:
                            rel_name = parts[0]       # Ví dụ: P931
                            description = parts[-1]   # Luôn lấy phần cuối cùng làm mô tả
                    
                    elif task_name == 'TACRED':
                        # Định dạng: "ID <tab> Tên đầy đủ <tab> Mô tả"
                        if len(parts) >= 3:
                            rel_name = parts[0]       # Ví dụ: org:founded_by
                            description = parts[2]    # Mô tả luôn ở vị trí thứ 3
                    
                    else:
                        # Mặc định, nếu có task mới, thử phân tích theo kiểu FewRel
                        if len(parts) >= 2:
                            rel_name = parts[0]
                            description = parts[-1]
                    # --- KẾT THÚC LOGIC PHÂN NHÁNH ---

                    # Nếu đã phân tích thành công và quan hệ đó hợp lệ
                    if rel_name and description and rel_name in self.rel2id:
                        rel_id = self.rel2id[rel_name]
                        rel2des[rel_name] = [description]
                        id2des[rel_id] = [description]

        except FileNotFoundError:
            print(f"CẢNH BÁO: Không tìm thấy file description tại '{file_path}'.")

        # Logic fallback: Nếu sau khi đọc mà một quan hệ nào đó vẫn không có mô tả,
        # thì dùng chính tên của nó làm mô tả.
        for rel_id, rel_name in self.id2rel.items():
            if rel_id not in id2des:
                print(f"CẢNH BÁO: Quan hệ '{rel_name}' (ID: {rel_id}) không có mô tả, dùng tên làm mặc định.")
                id2des[rel_id] = [rel_name]
                rel2des[rel_name] = [rel_name]
                
        return rel2des, id2des
