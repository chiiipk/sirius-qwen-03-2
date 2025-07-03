import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Lớp Buffer để lưu các mẫu từ tác vụ cũ (đã có từ các bước trước)
class Buffer:
    def __init__(self, config):
        self.config = config
        self.buffer = {}
    def __len__(self):
        return sum(len(samples) for samples in self.buffer.values())
    def add_exemplars(self, new_exemplars_dict):
        for label, samples in new_exemplars_dict.items():
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
        return (self.data[idx], idx)
    def collate_fn(self, data):
        labels = torch.tensor([item[0]['relation'] for item in data])
        indices = torch.tensor([item[1] for item in data])
        tokens_list = [torch.tensor(item[0]['tokens']) for item in data]
        padded_tokens = pad_sequence(tokens_list, batch_first=True, padding_value=self.config.pad_token_id)
        attention_mask = (padded_tokens != self.config.pad_token_id).long()
        batch_instance = {'ids': padded_tokens, 'mask': attention_mask}
        return batch_instance, labels, indices

def get_data_loader(config, data, shuffle=False, drop_last=False, batch_size=None):
    if not data: return None
    dataset = CustomDataset(data, config)
    if batch_size is None: batch_size = min(config.batch_size, len(data))
    else: batch_size = min(batch_size, len(data))
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=config.num_workers, collate_fn=dataset.collate_fn, drop_last=drop_last)
