import argparse
import torch
import random
import sys
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans, AgglomerativeClustering
from config import Config
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

from sampler import data_sampler_CFRL
from data_loader import get_data_loader, Buffer
from utils import Moment, set_seed
from encoder import EncodingModel
from losses import TripletLoss
from transformers import AutoTokenizer

class Manager(object):
    def __init__(self, config, args) -> None:
        super().__init__()
        self.config = config
        self.args = args 
        self.buffer = Buffer(config)
        # Khởi tạo các thuộc tính để tránh AttributeError
        self.id2rel = None
        self.rel2id = None
        self.tokenizer = None
        self.moment = None

    # --- CÁC HÀM HỖ TRỢ ---
    def _cosine_similarity(self, x1, x2):
        x2_aligned = x2.to(device=x1.device, dtype=x1.dtype)
        x1_norm = F.normalize(x1, p=2, dim=1)
        x2_norm = F.normalize(x2_aligned, p=2, dim=1)
        return torch.matmul(x1_norm, x2_norm.T)

    def get_memory_proto(self, encoder, dataset):
        if not dataset: return None, None
        data_loader = get_data_loader(self.config, dataset, shuffle=False, drop_last=False, batch_size=1)
        if data_loader is None: return None, None
        features = []
        encoder.eval()
        for step, (instance, label, idx) in enumerate(data_loader):
            with torch.no_grad():
                for k in instance.keys(): instance[k] = instance[k].to(self.config.device)
                hidden = encoder(instance)
                features.append(hidden.detach().cpu().float())
        if not features: return None, None
        features = torch.cat(features, dim=0)
        return features.mean(0), features

    def get_cluster_and_centroids(self, embeddings):
        embeddings_np = embeddings.cpu().float().numpy()
        clustering_model = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=self.args.distance_threshold)
        clusters = clustering_model.fit_predict(embeddings_np)
        centroids = {cid: torch.mean(embeddings[clusters == cid], dim=0) for cid in np.unique(clusters)}
        return clusters, centroids

    # --- HÀM SELECT_MEMORY ĐÃ ĐƯỢC HOÀN THIỆN ---
    def select_memory(self, encoder, dataset):
        N, M = len(dataset), self.config.memory_size
        if N == 0: return []
        if N <= M: return copy.deepcopy(dataset)
        
        data_loader = get_data_loader(self.config, dataset, shuffle=False, drop_last=False, batch_size=64)
        if not data_loader: return []

        features = []
        encoder.eval()
        for step, (instance, label, idx) in enumerate(data_loader):
            with torch.no_grad():
                for k in instance.keys(): instance[k] = instance[k].to(self.config.device)
                hidden = encoder(instance)
                features.append(hidden.detach().cpu().float())
        
        if not features: return []
        features = torch.cat(features, dim=0).numpy()
        
        distances = KMeans(n_clusters=M, random_state=self.config.seed, n_init=10).fit_transform(features)
        
        mem_set = []
        for k in range(M):
            sel_index = np.argmin(distances[:, k])
            if sel_index != -1:
                mem_set.append(dataset[sel_index])
                distances[sel_index, :] = np.inf
        return mem_set

    # --- HÀM TRAIN_MODEL ĐÃ ĐƯỢC HOÀN THIỆN ---
    def train_model(self, encoder, training_data, seen_descriptions, seen_relations, list_seen_des_tokens):
        data_loader = get_data_loader(self.config, training_data, shuffle=True)
        if not data_loader: return
        
        optimizer = optim.Adam(encoder.parameters(), lr=self.config.lr)
        encoder.train()
        triplet = TripletLoss()
        
        for i in range(self.config.epoch):
            for batch_num, (instance, labels, ind) in enumerate(data_loader):
                optimizer.zero_grad()
                for k in instance.keys(): instance[k] = instance[k].to(self.config.device)
                
                # ### SỬA LỖI ###: Tạo batch mô tả từ list_seen_des_tokens đã được token hóa sẵn
                # Thay vì dùng seen_des, ta dùng list_seen_des_tokens mà sampler đã chuẩn bị
                # Giả định list_seen_des_tokens là một list các dict {'ids':..., 'mask':...}
                # và chúng ta cần lấy ra đúng mô tả cho các label trong batch.
                # Đây là một bước phức tạp, để đơn giản hóa, ta sẽ tạo batch_des_instance
                # bằng cách token hóa lại mô tả từ seen_descriptions
                des_texts = [seen_descriptions[self.id2rel[label.item()]][0] for label in labels] # Lấy mô tả đầu tiên
                batch_des_instance = self.tokenizer(
                    des_texts,
                    padding='max_length',
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors='pt'
                ).to(self.config.device)
    
                hidden = encoder(instance)
                rep_des = encoder(batch_des_instance, is_des=True)
                rep_des_2 = encoder(batch_des_instance, is_des=True)

                with torch.no_grad():
                    if not list_seen_des: continue
                    rep_seen_des_list = [encoder({'ids': torch.tensor([d['ids']]).to(self.config.device), 'mask': torch.tensor([d['mask']]).to(self.config.device)}, is_des=True) for d in list_seen_des]
                    if not rep_seen_des_list: continue
                    rep_seen_des = torch.cat(rep_seen_des_list)
                    clusters, clusters_centroids = self.get_cluster_and_centroids(rep_seen_des)

                relationid2_clustercentroids = {self.rel2id[rel]: clusters_centroids[clusters[idx]] for idx, rel in enumerate(seen_relations)}
                relation_2_cluster = {self.rel2id[rel]: clusters[idx] for idx, rel in enumerate(seen_relations)}

                loss1 = self.moment.contrastive_loss(hidden, labels, False, des=rep_des, relation_2_cluster=relation_2_cluster)
                loss2 = self.moment.mutual_information_loss_cluster(hidden, rep_des, labels, temperature=self.args.temperature, relation_2_cluster=relation_2_cluster)
                loss4 = self.moment.mutual_information_loss_cluster(rep_des, rep_des_2, labels, temperature=self.args.temperature, relation_2_cluster=relation_2_cluster)
                
                loss = self.args.lambda_1*loss1 + self.args.lambda_2*loss2 + self.args.lambda_4*loss4

                if len(relationid2_clustercentroids) > 1:
                    cluster_centroids_tensor = torch.stack([relationid2_clustercentroids[label.item()] for label in labels]).to(self.config.device)
                    cos_sims = torch.nn.functional.cosine_similarity(hidden.unsqueeze(1), cluster_centroids_tensor.unsqueeze(0), dim=2)
                    _, top_indices = torch.topk(cos_sims, k=min(2, cos_sims.size(1)), dim=1)
                    nearest_indices = top_indices[:, 1 if top_indices.size(1) > 1 else 0]
                    nearest_cluster_centroids = cluster_centroids_tensor[nearest_indices]
                    loss3 = triplet(hidden, rep_des, cluster_centroids_tensor) + triplet(hidden, cluster_centroids_tensor, nearest_cluster_centroids)
                    loss += self.args.lambda_3*loss3
                
                loss.backward()
                optimizer.step()
                self.moment.update_des(ind, hidden.detach().cpu().float(), rep_des.detach().cpu().float(), is_memory=False)
        print('')

    def eval_encoder_proto_des(self, encoder, seen_proto, seen_relid, test_data, rep_des):
        data_loader = get_data_loader(self.config, test_data, False, False, 16)
        if not data_loader: return 0.0, 0.0, 0.0
        
        corrects, corrects1, corrects2, total = 0.0, 0.0, 0.0, 0.0
        encoder.eval()
        
        for batch_num, (instance, label, _) in enumerate(data_loader):
            with torch.no_grad():
                for k in instance.keys(): instance[k] = instance[k].to(self.config.device)
                hidden = encoder(instance)

            seen_proto_gpu = seen_proto.to(hidden.device)
            rep_des_gpu = rep_des.to(hidden.device)
            
            logits = self._cosine_similarity(hidden, seen_proto_gpu)
            pred = torch.tensor([seen_relid[i] for i in torch.argmax(logits.cpu(), dim=1)])
            corrects += torch.eq(pred, label.cpu()).sum().item()
            
            if rep_des_gpu.size(0) == seen_proto_gpu.size(0):
                logits_des = self._cosine_similarity(hidden, rep_des_gpu)
                pred1 = torch.tensor([seen_relid[i] for i in torch.argmax(logits_des.cpu(), dim=1)])
                corrects1 += torch.eq(pred1, label.cpu()).sum().item()
                logits_rrf = logits + logits_des
            else:
                logits_rrf = logits
            
            pred2 = torch.tensor([seen_relid[i] for i in torch.argmax(logits_rrf.cpu(), dim=1)])
            corrects2 += torch.eq(pred2, label.cpu()).sum().item()
            total += label.size(0)
            
        return (corrects/total, corrects1/total, corrects2/total) if total > 0 else (0.0, 0.0, 0.0)

    # --- HÀM TRAIN CHÍNH ĐÃ ĐƯỢC HOÀN THIỆN ---
    def train(self):
        # Khởi tạo sampler
        sampler = data_sampler_CFRL(config=self.config, seed=self.config.seed)
        
        # ### SỬA LỖI ###: Lấy tokenizer từ sampler và truyền cho encoder
        self.tokenizer = sampler.tokenizer 
        self.id2rel, self.rel2id = sampler.id2rel, sampler.rel2id
        
        # Truyền config vào encoder, encoder sẽ tự xử lý tokenizer và model
        encoder = EncodingModel(self.config)
        encoder.to(self.config.device)
        
        total_acc, total_acc1, total_acc2 = [], [], []
        memory_for_prototypes = {}
        
        # `seen_descriptions` sẽ được lấy trực tiếp từ sampler
        for step, (training_data, _, test_data, current_relations, historic_test_data, seen_relations, seen_descriptions) in enumerate(sampler):
            print(f"\n{'='*20} BẮT ĐẦU TÁC VỤ {step + 1}/{sampler.task_length} {'='*20}")
            
            # Tạo list các mô tả đã được token hóa sẵn sàng cho việc gom cụm
            # Lấy mô tả đầu tiên của mỗi quan hệ đã thấy
            list_seen_des = [seen_descriptions[rel][0] for rel in seen_relations if rel in seen_descriptions and seen_descriptions[rel]]
            tokenized_list_seen_des = self.tokenizer(
                list_seen_des,
                padding='max_length',
                truncation=True,
                max_length=self.config.max_length,
                return_tensors='pt'
            )
    
            current_task_data = [item for rel in current_relations for item in training_data.get(rel, [])]
            combined_training_data = current_task_data + self.buffer.get_data()
            print(f"Huấn luyện trên {len(combined_training_data)} mẫu ({len(current_task_data)} mới, {len(self.buffer.get_data())} từ buffer).")
            
            if combined_training_data:
                self.moment = Moment(self.config)
                # ### SỬA LỖI ###: Truyền thêm tokenizer và seen_descriptions vào init_moment
                self.moment.init_moment(encoder, self.tokenizer, combined_training_data, seen_descriptions, self.id2rel)
                
                # `train_model` giờ nhận `seen_descriptions` và `tokenized_list_seen_des`
                self.train_model(encoder, combined_training_data, seen_descriptions, seen_relations, tokenized_list_seen_des)

            for rel in current_relations:
                rel_id = self.rel2id[rel]
                exemplars = self.select_memory(encoder, training_data[rel])
                self.buffer.add_exemplars({rel_id: exemplars})
                memory_for_prototypes[rel] = exemplars

            final_protos, final_des_reps, final_relids = [], [], []
            with torch.no_grad():
                encoder.eval()
                for rel in seen_relations:
                    if rel in memory_for_prototypes and rel in seen_des:
                        proto, _ = self.get_memory_proto(encoder, memory_for_prototypes[rel])
                        if proto is not None:
                            des_dict = seen_des[rel]
                            des_input = {'ids': torch.tensor([des_dict['ids']]).to(self.config.device), 
                                         'mask': torch.tensor([des_dict['mask']]).to(self.config.device)}
                            des_rep = encoder(des_input, is_des=True)
                            final_protos.append(proto)
                            final_des_reps.append(des_rep)
                            final_relids.append(self.rel2id[rel])
            
            if not final_protos: continue
            
            seen_proto = torch.stack(final_protos)
            rep_des = torch.cat(final_des_reps)
            
            all_historic_test_data = [item for rel in seen_relations for item in historic_test_data[rel]]
            acc, acc1, acc2 = self.eval_encoder_proto_des(encoder, seen_proto, final_relids, all_historic_test_data, rep_des)
            total_acc.append(acc); total_acc1.append(acc1); total_acc2.append(acc2)
            print(f"-> Accuracy (Proto): {acc:.4f} | (Desc): {acc1:.4f} | (Combined): {acc2:.4f}")

        torch.cuda.empty_cache()
        return total_acc, total_acc1, total_acc2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="FewRel", type=str)
    parser.add_argument("--num_k", default=5, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--lambda_1", default=1, type=float)
    parser.add_argument("--lambda_2", default=1, type=float)
    parser.add_argument("--lambda_3", default=0.25, type=float)
    parser.add_argument("--lambda_4", default=0.25, type=float)
    parser.add_argument("--temperature", default=0.01, type=float)
    parser.add_argument("--distance_threshold", default=0.1, type=float)
    args = parser.parse_args()
    
    config = Config('config.ini')
    for key, value in vars(args).items():
        if value is not None: setattr(config, key, value)
    
    config.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    acc_list, acc_list1, acc_list2 = [], [], []
    seeds_to_run = [args.seed] if args.seed is not None else config.seeds
    print(f"\nBẮT ĐẦU CHẠY VỚI {len(seeds_to_run)} SEED(S): {seeds_to_run}")

    for seed in seeds_to_run:
        set_seed(seed)
        config.seed = seed
        print(f"\n{'#'*25} BẮT ĐẦU LẦN CHẠY VỚI SEED: {seed} {'#'*25}")
        manager = Manager(config, args)
        acc, acc1, acc2 = manager.train()
        acc_list.append(acc); acc_list1.append(acc1); acc_list2.append(acc2)
        print(f"KẾT QUẢ ACCURACY CỦA SEED {seed}: Proto={acc}, Des={acc1}, RRF={acc2}")
    
    print(f"\n{'='*25} KẾT QUẢ TỔNG HỢP SAU {len(seeds_to_run)} LẦN CHẠY {'='*25}")
    
    for name, acc_data in [("Prototype", acc_list), ("Description", acc_list1), ("Combined (RRF)", acc_list2)]:
        if not acc_data: continue
        accs_array = np.array(acc_data)
        if accs_array.size > 0:
            mean_accs = np.mean(accs_array, axis=0)
            print(f"\n--- Accuracy dựa trên {name} ---")
            print('Kết quả trung bình qua các tác vụ: ', np.around(mean_accs, 4))
            if len(mean_accs) > 0: print(f'Kết quả tác vụ cuối cùng: {mean_accs[-1]:.4f}')
