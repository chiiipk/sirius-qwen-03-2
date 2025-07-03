import torch
import openai
import random
import time
import numpy as np
import torch.nn.functional as F
from data_loader import get_data_loader
from nltk import word_tokenize
from retry import retry

# --- BẮT ĐẦU PHẦN THÊM MỚI ---
# Thêm hàm set_seed để đảm bảo tính tái lập cho các lần chạy
def set_seed(seed):
    """
    Thiết lập seed cho các thư viện để đảm bảo kết quả có thể tái lập.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"--- Đã thiết lập seed: {seed} ---")
# --- KẾT THÚC PHẦN THÊM MỚI ---


class Moment:
    def __init__(self, config) -> None:
        self.config = config
        self.features = None
        self.features_des = None
        self.labels = None
        self.mem_samples = None
        self.mem_features = None
        self.mem_labels = None
        self.mem_features_des = None
        self.sample_k = config.sample_k
        self.temperature = config.contrastive_temp
        self.m = config.margin


    def init_moment(self, encoder, dataset, is_memory=False):
        encoder.eval()
        datalen = len(dataset)
        if not dataset: # Tránh lỗi nếu dataset rỗng
            return
            
        data_loader = get_data_loader(self.config, dataset, shuffle=False)
        if data_loader is None:
            return

        if not is_memory:
            self.features = torch.zeros(datalen, self.config.encoder_output_size)
            self.features_des = torch.zeros(datalen, self.config.encoder_output_size)
            lbs = []
            for step, (instance, labels, ind) in enumerate(data_loader):
                with torch.no_grad():
                    for k in instance.keys():
                        instance[k] = instance[k].to(self.config.device)
                    hidden = encoder(instance)
                    fea = hidden.detach().cpu().float() 
                    self.update(ind, fea)
                    lbs.append(labels)
            # --- SỬA LỖI ---: Xóa dòng lặp lại
            # lbs.append(labels) 
            lbs = torch.cat(lbs)
            self.labels = lbs
        else:
            self.mem_samples = dataset
            self.mem_features = torch.zeros(datalen, self.config.encoder_output_size)
            self.mem_features_des = torch.zeros(datalen, self.config.encoder_output_size)
            lbs = []
            for step, (instance, labels, ind) in enumerate(data_loader):
                with torch.no_grad():
                    for k in instance.keys():
                        instance[k] = instance[k].to(self.config.device)
                    hidden = encoder(instance)
                    fea = hidden.detach().cpu().float() 
                    self.update(ind, fea, is_memory)
                    lbs.append(labels)
            # --- SỬA LỖI ---: Xóa dòng lặp lại
            # lbs.append(labels)
            lbs = torch.cat(lbs)
            self.mem_labels = lbs
            # --- SỬA LỖI ---: Xóa dòng lặp lại
            # self.mem_labels = lbs      

    # ... Các hàm còn lại của lớp Moment và các hàm GPT không cần thay đổi ...
    # ... (contrastive_loss, mutual_information_loss_cluster, etc. remain the same)
    def update(self, ind, feature, is_memory=False):
        if not is_memory:
            self.features[ind] = feature
        else:
            self.mem_features[ind] = feature
            
    def update_des(self, ind, feature, feature_des, is_memory=False):
        if not is_memory:
            self.features[ind] = feature
            self.features_des[ind] = feature_des
        else:
            self.mem_features[ind] = feature
            self.mem_features_des[ind] = feature_des

    def update_allmem(self, encoder):
        data_loader = get_data_loader(self.config, self.mem_samples, batch_size=64) # shuffle=False
        for step, (instance, labels, ind) in enumerate(data_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance)
            fea = hidden.detach().cpu().data
            self.update(ind, fea, is_memory=True)
            
    # ... (Các hàm còn lại của file giữ nguyên không đổi)
    def contrastive_loss(self, x, labels, is_memory=False, des=None, relation_2_cluster=None):
        '''
        x (B, H)
        '''
        x = x.float()
        if des is not None:
            des = des.float()
    
        if is_memory:
            ct_x = self.mem_features.to(self.config.device).float()
            ct_x_des = self.mem_features_des.to(self.config.device).float()
            ct_y = self.mem_labels
        else:
            idx = list(range(len(self.features)))
            if len(idx) > self.sample_k:
                sample_id = random.sample(idx, self.sample_k)
            else:
                sample_id = idx
            ct_x = self.features[sample_id].to(self.config.device).float()
            ct_x_des = self.features_des[sample_id].to(self.config.device).float()
            ct_y = self.labels[sample_id]
    
    
        # l2 normalize
        x = F.normalize(x, p=2, dim=1)
        ct_x = F.normalize(ct_x, p=2, dim=1)
        
        t1 = torch.mm(x, ct_x.T) + 1 # 0 <= cos + 1 <= 2
        
        if des is not None:
            des = F.normalize(des, p=2, dim=1)
            ct_x_des = F.normalize(ct_x_des, p=2, dim=1)
            t2 = torch.mm(des, ct_x_des.T)
        else:
            # Nếu des là None, cần khởi tạo t2 để tránh lỗi
            t2 = torch.zeros_like(t1)
    
        zeros = (torch.zeros_like(t1)).to(self.config.device)
        
        pos = torch.ones_like(t1)
        neg = torch.ones_like(t1)
    
        if relation_2_cluster is not None:
            labels_clusters = torch.tensor([relation_2_cluster[label.item()] for label in labels], device=self.config.device)
            ct_y_clusters = torch.tensor([relation_2_cluster[label.item()] for label in ct_y], device=self.config.device)
            relation_match = (labels_clusters.unsqueeze(1) == ct_y_clusters.unsqueeze(0)).float()
            neg = relation_match * (1.0 + 0.2* t2) + (1.0 - relation_match) * 1.0
    
        dot_product_tempered_pos = torch.where(pos > 0, pos * t1 / self.temperature, zeros)
        dot_product_tempered_neg = torch.where(neg > 0, neg * t1 / self.temperature, zeros)
        
        exp_dot_tempered_pos = (
            torch.exp(dot_product_tempered_pos - torch.max(dot_product_tempered_pos, dim=1, keepdim=True)[0].detach()) + 1e-5
        )
        exp_dot_tempered_neg = (
            torch.exp(dot_product_tempered_neg - torch.max(dot_product_tempered_pos, dim=1, keepdim=True)[0].detach()) + 1e-5
        )
        mask_combined_pos = (labels.unsqueeze(1).repeat(1, ct_y.shape[0]) == ct_y).to(self.config.device)
        mask_combined_neg = ~mask_combined_pos
        cardinality_per_samples = torch.sum(mask_combined_pos, dim=1)
        
        # Tránh chia cho 0 nếu một mẫu không có positive pair nào trong batch
        cardinality_per_samples = torch.where(cardinality_per_samples == 0, 1, cardinality_per_samples)
    
        sum_temp = torch.sum(exp_dot_tempered_pos * mask_combined_pos, dim=1, keepdim=True) \
            + torch.sum(exp_dot_tempered_neg * mask_combined_neg, dim=1, keepdim=True)
        log_prob = -torch.log(exp_dot_tempered_pos / (sum_temp + 1e-8)) # Thêm epsilon để tránh log(0)
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined_pos, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)
    
        return supervised_contrastive_loss
    def mutual_information_loss_cluster(self, x_bert, x_stella, labels,  temperature=0.1,  relation_2_cluster = None):
        mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        mask = mask.to(self.config.device)
        x_bert = F.normalize(x_bert, p=2, dim=1)
        x_stella = F.normalize(x_stella, p=2, dim=1)

        t2 = torch.mm(x_stella, x_stella.T) + 1

        similarity_matrix = torch.matmul(x_bert, x_stella.t()) / temperature

        if relation_2_cluster is not None:
            labels_clusters = torch.tensor([relation_2_cluster[label.item()] for label in labels], device=self.config.device)
            relation_match = (labels_clusters.unsqueeze(1) == labels_clusters.unsqueeze(0)).float()
            neg = relation_match * (1.0 + 0.2*t2) + (1.0 - relation_match) * 1.0
            f_neg = similarity_matrix*(~mask)*neg
        else:
            f_neg = similarity_matrix*(~mask)

        f_pos = torch.diag(similarity_matrix)
        f_concat = torch.cat([f_pos.unsqueeze(1), f_neg], dim=1)
        softmax_probs = torch.nn.functional.softmax(f_concat, dim=1)
        infoNCE_loss = -torch.log(softmax_probs[:, 0]).mean()

        return infoNCE_loss

    def distillation_loss_hidden(self,
            hidden_teacher: torch.Tensor, # (b,n)
            hidden_student: torch.Tensor, # (b,n)
        ) -> torch.Tensor:
        device = hidden_student.device
        hidden_teacher = hidden_teacher.to(device) 
        hidden_teacher = F.normalize(hidden_teacher, dim=1)  
        hidden_student = F.normalize(hidden_student, dim=1) 
        cos_sim = F.cosine_similarity(hidden_student, hidden_teacher, dim=1)
        loss = 1.0 - cos_sim
        return loss.mean()





from openai import OpenAI

def gpt(input, t=0, key=None):
    MAX_TRIES = 15
    client = OpenAI(api_key=key)

    while MAX_TRIES > 0:
        try:
            time.sleep(5)
            completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": input}
            ]
            ,
            temperature=t

            )
            return completion.choices[0].message.content
        except Exception as e:
            print(e)
            MAX_TRIES -= 1
    print('gen failed')
    return ''

def parse(rel2id, text):
    cons = ['Relation:', 'Context:', 'Head Entity:', 'Tail Entity:']
    lens = [ len(item) for item in cons]
    parse_text = []

    temp = text
    while True:
        parse_item = {}

        i = temp.find(cons[0])
        temp = temp[i+lens[0]:]
        i = temp.find(cons[1])
        r = temp[:i].strip()
        temp = temp[i+lens[1]:]
        i = temp.find(cons[2])
        c = temp[:i].strip()
        temp = temp[i+lens[2]:]
        i = temp.find(cons[3])
        h = temp[:i].strip()
        temp = temp[i+lens[3]:]
        i = temp.find('\n')
        t = temp[:i].strip()
        i = temp.find(cons[0])

        r = r.split('\n')[0]
        r = r.replace('**', '')
        r = r.replace('\n','')
        r = r.strip()

        parse_item['relation'] = rel2id[r]
        parse_item['index'] = 0
        tokens = word_tokenize(c.lower())
        parse_item['tokens'] = tokens

        headent, tailent = h.lower(), t.lower()
        h_tokens, t_tokens = word_tokenize(headent), word_tokenize(tailent)
        try:
            h1 = tokens.index(h_tokens[0])
        except Exception:
            h1 = 0
        try:
            h2 = tokens.index(h_tokens[-1])
        except Exception:
            h2 = h1        
        try:
            t1 = tokens.index(t_tokens[0])
        except Exception:
            t1 = h2
        try:
            t2 = tokens.index(t_tokens[-1])
        except Exception:
            t2 = t1             
        parse_item['h'] = [headent, '0', [[h1, h2]]]
        parse_item['t'] = [tailent, '0', [[t1, t2]]]

        parse_text.append(parse_item)

        if i == -1:
            break
        temp = temp[i:]

    return parse_text

def prompt_input(rname, rdesc, sample=None, n=10):
    pre_input = 'You are a data scientist working on a relation extraction task. Please do the following task and do not give output in the markdown format.'
    input = ''
    if sample == None:
        input = 'One sample in relation extraction datasets consists of a relation, a context, a pair of head and tail entities in the context.The head entity has the relation with the tail entity. Generate ' \
            + str(n) + ' diversity samples (must have full : Relation , Context , Head Entity , Tail Entity) for the relation "'+ rname \
            + '" which means ' + rdesc \
            + ', and indicate the head entity and tail entity in the following format:\n' \
            + 'Relation: xxx\nContext: xxx\nHead Entity: xxx\nTail Entity: xxx'
    else:
        input = 'One sample in relation extraction datasets consists of a relation, a context, a pair of head and tail entities in the context.The head entity has the relation with the tail entity.\n' \
            + 'Relation "' + rname + '" means ' + rdesc + '.\nHere is an example:\n' \
            + 'Relation: ' + rname + '\nContext: ' + sample['tokens'] + '\nHead Entity: ' + sample['h'] + '\nTail Entity: ' + sample['t'] + '\n' \
            + 'Please generate ' + str(n) + ' diversity samples (must have full : Relation , Context , Head Entity , Tail Entity) like the above example for the relation "'+ rname + '":'
    return pre_input + input


def gen_data(r2desc, rel2id, sample, n=10, t=0, key=None):
    rname = sample['relation']
    rdesc = r2desc[rname]
    print('####', rname ,'####')
    input = prompt_input(rname, rdesc, sample=sample, n=n)
    print(input)
    output = gpt(input=input, t=t, key=key)
    print(output)
    try:
        parse_output = parse(rel2id, output)
    except:
        output = gpt(input=input + "\nRelation: ", t=t, key=key)
        parse_output = parse(rel2id, output)


    return parse_output
