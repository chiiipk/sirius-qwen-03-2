import argparse
import torch
import random
import sys
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from config import Config
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering

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

    def _edist(self, x1, x2):
        b = x1.size()[0]
        L2dist = nn.PairwiseDistance(p=2)
        dist = torch.cat([torch.unsqueeze(L2dist(x2, x1[i]), 0) for i in range(b)], 0)
        return dist

    def _cosine_similarity(self, x1, x2):
        x2_aligned = x2.to(device=x1.device, dtype=x1.dtype)
        x1_norm = F.normalize(x1, p=2, dim=1)
        x2_norm = F.normalize(x2_aligned, p=2, dim=1)
        sim = torch.matmul(x1_norm, x2_norm.T)
        return sim

    def get_memory_proto(self, encoder, dataset):
        if not dataset: return None, None
        data_loader = get_data_loader(config, dataset, shuffle=False, drop_last=False,  batch_size=1)
        features = []
        encoder.eval()
        for step, (instance, label, idx) in enumerate(data_loader):
            with torch.no_grad():
                for k in instance.keys(): instance[k] = instance[k].to(self.config.device)
                hidden = encoder(instance)
                features.append(hidden.detach().cpu().float())
        features = torch.cat(features, dim=0)
        proto = features.mean(0)
        return proto, features
    def get_cluster_and_centroids(self, embeddings):
      embeddings_np = embeddings.cpu().float().numpy()
      clustering_model = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=self.args.distance_threshold)
      clusters = clustering_model.fit_predict(embeddings_np)

      centroids = {}
      for cluster_id in np.unique(clusters):
          cluster_embeddings = embeddings[clusters == cluster_id]
          centroid = torch.mean(cluster_embeddings, dim=0)
          centroids[cluster_id] = centroid

      return clusters, centroids

    def select_memory(self, encoder, dataset):
        N, M = len(dataset), self.config.memory_size
        if N == 0: return []
        if N <= M: return copy.deepcopy(dataset)
        
        data_loader = get_data_loader(self.config, dataset, shuffle=False, drop_last=False, batch_size=64)
        features = []
        encoder.eval()
        for step, (instance, label, idx) in enumerate(data_loader):
            with torch.no_grad():
                for k in instance.keys(): instance[k] = instance[k].to(self.config.device)
                hidden = encoder(instance)
                features.append(hidden.detach().cpu().float())
        features = torch.cat(features, dim=0).numpy()
        
        num_clusters = M
        distances = KMeans(n_clusters=num_clusters, random_state=self.config.seed, n_init=10).fit_transform(features)
        
        mem_set = []
        for k in range(num_clusters):
            sel_index = np.argmin(distances[:, k])
            if sel_index != -1: # Ensure the index is valid
                mem_set.append(dataset[sel_index])
                distances[sel_index, :] = np.inf # Avoid selecting the same sample again
        return mem_set
      # bỏ mem_feas

    # Các hàm train_model và eval_encoder_proto_des không đổi và khá dài,
    # chúng ta có thể giữ nguyên chúng từ file gốc của bạn
    # ... (Giả định train_model và eval_encoder_proto_des đã tồn tại và đúng)
    def train_model(self, encoder, training_data, seen_des, seen_relations, list_seen_des, is_memory=False):
          data_loader = get_data_loader(self.config, training_data, shuffle=True)
          optimizer = optim.Adam(params=encoder.parameters(), lr=self.config.lr)
          encoder.train()
          epoch = self.config.epoch_mem if is_memory else self.config.epoch
          triplet = TripletLoss()
          optimizer.zero_grad()
  
          for i in range(epoch):
              for batch_num, (instance, labels, ind) in enumerate(data_loader):
                  for k in instance.keys():
                      instance[k] = instance[k].to(self.config.device)
  
                  batch_instance = {'ids': [], 'mask': []}
                  batch_instance['ids'] = torch.tensor([seen_des[self.id2rel[label.item()]]['ids'] for label in labels]).to(self.config.device)
                  batch_instance['mask'] = torch.tensor([seen_des[self.id2rel[label.item()]]['mask'] for label in labels]).to(self.config.device)
  
                  hidden = encoder(instance) # b, dim
                  rep_des = encoder(batch_instance, is_des = True) # b, dim
                  rep_des_2 = encoder(batch_instance, is_des = True) # b, dim
  
                  with torch.no_grad():
                      rep_seen_des = []
                      for i2 in range(len(list_seen_des)):
                          sample = {
                              'ids' : torch.tensor([list_seen_des[i2]['ids']]).to(self.config.device),
                              'mask' : torch.tensor([list_seen_des[i2]['mask']]).to(self.config.device)
                          }
                          hidden_des = encoder(sample, is_des=True)
                          rep_seen_des.append(hidden_des)
                      rep_seen_des = torch.cat(rep_seen_des, dim=0)
                      # SỬA LỖI: Gọi hàm get_cluster_and_centroids mà không cần truyền args vì nó đã là thuộc tính của class
                      clusters, clusters_centroids = self.get_cluster_and_centroids(rep_seen_des)
                  flag = 0
                  if len(clusters) == max(clusters) + 1:
                      flag = 1
  
                  relationid2_clustercentroids = {}
                  for index, rel in enumerate(seen_relations):
                      relationid2_clustercentroids[self.rel2id[rel]] = clusters_centroids[clusters[index]]
  
                  relation_2_cluster = {}
                  for i1 in range(len(seen_relations)):
                      relation_2_cluster[self.rel2id[seen_relations[i1]]] = clusters[i1]
  
                  loss2 = self.moment.mutual_information_loss_cluster(hidden, rep_des, labels, temperature=self.args.temperature,relation_2_cluster=relation_2_cluster)
                  loss4 = self.moment.mutual_information_loss_cluster(rep_des, rep_des_2, labels, temperature=self.args.temperature,relation_2_cluster=relation_2_cluster)
  
                  cluster_centroids_list = [relationid2_clustercentroids[label.item()] for label in labels]
                  cluster_centroids  = torch.stack(cluster_centroids_list, dim = 0).to(self.config.device)
  
                  nearest_cluster_centroids = []
                  for hid in hidden:
                      cos_similarities = torch.nn.functional.cosine_similarity(hid.unsqueeze(0), cluster_centroids.to(hid.dtype), dim=1)
  
                      try:
                          k_val = min(2, cos_similarities.shape[0])
                          if k_val > 1:
                              top2_similarities, top2_indices = torch.topk(cos_similarities, k=k_val, dim=0)
                              top2_centroids = relationid2_clustercentroids[labels[top2_indices[1].item()].item()]
                          else:
                              top2_centroids = relationid2_clustercentroids[labels[torch.argmax(cos_similarities).item()].item()]
                      except RuntimeError as e:
                          print(f"RuntimeError in top-k selection: {e}")
                          top2_centroids = relationid2_clustercentroids[labels[torch.argmax(cos_similarities).item()].item()]
  
                      nearest_cluster_centroids.append(top2_centroids)
  
                  nearest_cluster_centroids = torch.stack(nearest_cluster_centroids, dim = 0).to(self.config.device)
                  loss1 = self.moment.contrastive_loss(hidden, labels, is_memory, des =rep_des, relation_2_cluster = relation_2_cluster)
  
                  if flag == 0:
                      loss3 = triplet(hidden, rep_des,  cluster_centroids) + triplet(hidden, cluster_centroids, nearest_cluster_centroids)
                      loss = self.args.lambda_1*(loss1) + self.args.lambda_2*(loss2) + self.args.lambda_3*(loss3) + self.args.lambda_4*(loss4)
                  else:
                      loss = self.args.lambda_1*(loss1) + self.args.lambda_2*(loss2) + self.args.lambda_4*(loss4)
  
                  loss.backward()
                  optimizer.step()
                  optimizer.zero_grad()
                  if is_memory:
                      self.moment.update_des(ind, hidden.detach().cpu().float(), rep_des.detach().cpu().float(), is_memory=True)
                  else:
                      self.moment.update_des(ind, hidden.detach().cpu().float(), rep_des.detach().cpu().float(), is_memory=False)
  
                  if is_memory:
                      sys.stdout.write('MemoryTrain:  epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()) + '\r')
                  else:
                      sys.stdout.write('CurrentTrain: epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()) + '\r')
                  sys.stdout.flush()
          print('')

    def eval_encoder_proto_des(self, encoder, seen_proto, seen_relid, test_data, rep_des):
        batch_size = 16
        test_loader = get_data_loader(self.config, test_data, False, False, batch_size)
        corrects, corrects1, corrects2, total = 0.0, 0.0, 0.0, 0.0
        encoder.eval()
        for batch_num, (instance, label, _) in enumerate(test_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            with torch.no_grad():
                hidden = encoder(instance)

            device = hidden.device
            dtype = hidden.dtype

            seen_proto_aligned = seen_proto.to(device=device, dtype=dtype)
            rep_des_aligned = rep_des.to(device=device, dtype=dtype)

            fea = hidden

            logits = self._cosine_similarity(fea, seen_proto_aligned)
            logits_des = self._cosine_similarity(fea, rep_des_aligned)
            logits_rrf = logits + logits_des

            label = label.cpu()
            cur_index = torch.argmax(logits.cpu(), dim=1)
            pred = torch.tensor([seen_relid[int(i)] for i in cur_index])
            corrects += torch.eq(pred, label).sum().item()

            cur_index1 = torch.argmax(logits_des.cpu(),dim=1)
            pred1 = torch.tensor([seen_relid[int(i)] for i in cur_index1])
            corrects1 += torch.eq(pred1, label).sum().item()

            cur_index2 = torch.argmax(logits_rrf.cpu(),dim=1)
            pred2 = torch.tensor([seen_relid[int(i)] for i in cur_index2])
            corrects2 += torch.eq(pred2, label).sum().item()

            total += label.size(0)
            acc = corrects / total if total > 0 else 0
            acc1 = corrects1 / total if total > 0 else 0
            acc2 = corrects2 / total if total > 0 else 0

            sys.stdout.write(f'[EVAL RRF] batch: {batch_num:4} | acc: {100*acc2:3.2f}%, total acc: {100*(corrects2/total):3.2f}%   ' + '\r')
            sys.stdout.flush()
        print('')
        return corrects / total, corrects1 / total, corrects2 / total

    def train(self):
        sampler = data_sampler_CFRL(config=self.config, seed=self.config.seed)
        self.id2rel = sampler.id2rel
        self.rel2id = sampler.rel2id
        self.r2desc = sampler.rel2des

        if self.config.model == 'qwen':
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        else: self.tokenizer = AutoTokenizer.from_pretrained(self.config.bert_path)
        
        encoder = EncodingModel(self.config)
        encoder.to(self.config.device)

        total_acc2 = []
        memory_for_prototypes = {}
        seen_des = {}

        for step, (training_data, _, test_data, current_relations, historic_test_data, seen_relations, seen_descriptions) in enumerate(sampler):
            print(f"\n{'='*20} BẮT ĐẦU TÁC VỤ {step + 1}/{sampler.task_length} {'='*20}")
            current_task_data = [item for rel in current_relations for item in training_data[rel]]
            buffer_data = self.buffer.get_data()
            combined_training_data = current_task_data + buffer_data
            print(f"Huấn luyện trên {len(combined_training_data)} mẫu ({len(current_task_data)} mới, {len(buffer_data)} từ buffer).")

            if not combined_training_data: continue

            seen_des.update(seen_descriptions)
            list_seen_des = [seen_des[rel] for rel in seen_relations if rel in seen_des]

            self.moment = Moment(self.config)
            self.moment.init_moment(encoder, combined_training_data, is_memory=False)
            self.train_model(encoder, combined_training_data, seen_des, seen_relations, list_seen_des, is_memory=False)

            print("--- Cập nhật buffer bằng K-Means ---")
            new_exemplars_for_buffer = {}
            for rel in current_relations:
                exemplars_for_rel = self.select_memory(encoder, training_data[rel])
                rel_id = self.rel2id[rel]
                new_exemplars_for_buffer[rel_id] = exemplars_for_rel
                memory_for_prototypes[rel] = exemplars_for_rel
            self.buffer.add_exemplars(new_exemplars_for_buffer)
            
            print("--- Đánh giá mô hình ---")
            seen_proto_list = [self.get_memory_proto(encoder, memory_for_prototypes[rel])[0] for rel in seen_relations if rel in memory_for_prototypes]
            if not seen_proto_list: continue
            seen_proto = torch.stack([p for p in seen_proto_list if p is not None], dim=0)
            seen_relid = [self.rel2id[rel] for rel in seen_relations if rel in memory_for_prototypes and self.get_memory_proto(encoder, memory_for_prototypes[rel])[0] is not None]
            
            test_data_all_seen_tasks = [item for rel in seen_relations for item in historic_test_data[rel]]
            
            # Placeholder cho rep_des, bạn cần thay thế bằng logic thực của mình
            rep_des = torch.randn(len(seen_relid), self.config.encoder_output_size) 
            
            _, _, ac2_rrf = self.eval_encoder_proto_des(encoder, seen_proto, seen_relid, test_data_all_seen_tasks, rep_des)
            total_acc2.append(f'{ac2_rrf:.4f}')
            print(f"  -> Độ chính xác trên tất cả các lớp đã thấy (Total Acc): {ac2_rrf:.4f}")

        torch.cuda.empty_cache()
        return [float(acc) for acc in total_acc2]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="FewRel", type=str)
    parser.add_argument("--num_k", default=5, type=int)
    parser.add_argument("--lambda_4", default=0.25, type=float)
    parser.add_argument("--seed", default=None, type=int, help="Chạy với một seed duy nhất")
    args = parser.parse_args()

    config = Config('config.ini')
    for key, value in vars(args).items():
        if value is not None: setattr(config, key, value)
    
    config.device = torch.device(config.device if torch.cuda.is_available() else "cpu")



    print(f'Task={config.task_name}, Model={config.model}')
    
    all_runs_accs = []
    seeds_to_run = [args.seed] if args.seed is not None else config.seeds
    
    print(f"\nBẮT ĐẦU CHẠY VỚI {len(seeds_to_run)} SEED(S): {seeds_to_run}")

    for seed in seeds_to_run:
        set_seed(seed)
        config.seed = seed
        print(f"\n{'#'*25} BẮT ĐẦU LẦN CHẠY VỚI SEED: {seed} {'#'*25}")
        manager = Manager(config, args)
        final_accuracies = manager.train()
        all_runs_accs.append(final_accuracies)
        print(f"KẾT QUẢ ACCURACY CỦA SEED {seed}: {final_accuracies}")
        torch.cuda.empty_cache()

    print(f"\n{'='*25} KẾT QUẢ TỔNG HỢP SAU {len(seeds_to_run)} LẦN CHẠY {'='*25}")
    accs_array = np.array(all_runs_accs)
    
    if accs_array.size > 0:
        if len(seeds_to_run) > 1:
            mean_accs, std_accs = np.mean(accs_array, axis=0), np.std(accs_array, axis=0)
            print("Độ chính xác trung bình qua các tác vụ:\n", np.around(mean_accs, 4))
            print("\nĐộ lệch chuẩn qua các tác vụ:\n", np.around(std_accs, 4))
            if len(mean_accs) > 0: print(f"\nKết quả tác vụ cuối cùng: Trung bình={mean_accs[-1]:.4f}, Std={std_accs[-1]:.4f}")
        else:
            print("Độ chính xác qua các tác vụ:\n", np.around(accs_array[0], 4))
            if len(accs_array[0]) > 0: print(f"\nKết quả tác vụ cuối cùng: {accs_array[0][-1]:.4f}")
