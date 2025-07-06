import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Lớp Buffer không thay đổi
class Buffer:
    def __init__(self, config):
        self.config = config
        self.buffer = {}
    def __len__(self):
        return sum(len(samples) for samples in self.buffer.values())
    def add_exemplars(self, new_exemplars_dict):
        for label, samples in new_exemplars_dict.items():
            # Sửa một lỗi nhỏ, đảm bảo new_exemplars_dict không rỗng
            if samples:
                self.buffer[label] = samples[:self.config.memory_size]
        print(f"-> Buffer đã cập nhật. Tổng số lớp: {len(self.buffer)}. Tổng số mẫu: {len(self)}")
    def get_data(self):
        return [sample for samples in self.buffer.values() for sample in samples]

class CustomDataset(Dataset):
    def __init__(self, data, config=None):
        self.data = data
        self.config = config
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # ### THAY ĐỔI 1 ###
        # Giờ __getitem__ trả về chính dictionary của mẫu và chỉ số của nó.
        # Không còn cấu trúc tuple lồng nhau (item[0], item[1]) nữa.
        return self.data[idx], idx

    def collate_fn(self, batch_data):
        # ### THAY ĐỔI 2 ###
        # Logic của collate_fn được viết lại hoàn toàn để xử lý đúng định dạng mới.
        
        # `batch_data` bây giờ là một list các tuple, ví dụ:
        # [ ({'relation':1, 'ids':[...], 'mask':[...]}, 0),  ({'relation':5, 'ids':[...], 'mask':[...]}, 1), ... ]
        
        # Tách riêng sample_dicts và indices
        samples, indices = zip(*batch_data)

        # Trích xuất dữ liệu từ các dictionary
        labels = torch.tensor([s['relation'] for s in samples])
        
        # ### THAY ĐỔI 3 ###
        # Đổi tên key từ 'tokens' thành 'ids' để khớp với sampler
        ids_list = [torch.tensor(s['ids']) for s in samples]

        # Padding để tạo thành batch
        padded_ids = pad_sequence(
            ids_list, 
            batch_first=True, 
            padding_value=self.config.pad_token_id
        )
        
        # Tạo attention mask từ padded_ids
        attention_mask = (padded_ids != self.config.pad_token_id).long()
        
        # Trả về đúng định dạng mà hàm train_model mong đợi
        batch_instance = {'ids': padded_ids, 'mask': attention_mask}
        return batch_instance, labels, torch.tensor(indices)

def get_data_loader(config, data, shuffle=False, drop_last=False, batch_size=None):
    if not data: 
        return None
        
    dataset = CustomDataset(data, config)
    
    # Xử lý trường hợp batch_size=None hoặc lớn hơn kích thước dataset
    if batch_size is None:
        effective_batch_size = min(config.batch_size, len(data))
    else:
        effective_batch_size = min(batch_size, len(data))

    # Nếu batch_size bằng 0 thì cũng không tạo loader
    if effective_batch_size == 0:
        return None

    return DataLoader(
        dataset=dataset, 
        batch_size=effective_batch_size, 
        shuffle=shuffle, 
        pin_memory=True, 
        num_workers=config.num_workers, 
        collate_fn=dataset.collate_fn, 
        drop_last=drop_last
    )
