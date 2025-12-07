
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

#======================= utils ==========================================


class PrototypesDataset(torch.utils.data.Dataset):
    def __init__(self, all_prototypes):
        self.features = []
        self.labels = []
        self.domain_ids = [] #domain_ids  # List of domain IDs corresponding to each prototype

        # Store features and labels per domain
        for class_id in range(len(all_prototypes)):
            # print(class_id) #[0-49]
            for domain_id in all_prototypes[class_id]:
                # print(domain_id) [0-5]
                for prototype in all_prototypes[class_id][domain_id]:
                    # print(prototype.shape)
                    self.features.append(prototype)
                    self.labels.append(class_id)  # Use class_id as the label
                    self.domain_ids.append(domain_id) 

        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        self.domain_ids = np.array(self.domain_ids)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        domain_id = self.domain_ids[idx]  # Domain ID for classification
        return feature, label, domain_id




def evalaa(model, test_loader):
    # model.eval()
    # i = 0
    with torch.no_grad():
        total = 0
        top1 = 0
        topk = 0
        for (test_imgs, test_labels) in tqdm(test_loader):
            # i+=1
            # if i>2: break
            test_labels = test_labels.cuda()
            out = model(test_imgs.cuda())
            _,maxk = torch.topk(out,5,dim=-1)
            total += test_labels.size(0)
            test_labels = test_labels.view(-1,1) # reshape labels from [n] to [n,1] to compare [n,k]
            top1 += (test_labels == maxk[:,0:1]).sum().item()
            topk += (test_labels == maxk).sum().item()
    # print(top1,topk,total)
    # model.train()
    return 100 * top1 / total, 100*topk/total


def evalaa_pathological(model, test_loader):
    # model.eval()
    # i = 0
    with torch.no_grad():
        total = 0
        top1 = 0
        topk = 0
        for (test_imgs, test_labels, _) in tqdm(test_loader):
            # i+=1
            # if i>2: break
            test_labels = test_labels.cuda()
            out = model(test_imgs.cuda())
            _,maxk = torch.topk(out,5,dim=-1)
            total += test_labels.size(0)
            test_labels = test_labels.view(-1,1) # reshape labels from [n] to [n,1] to compare [n,k]
            top1 += (test_labels == maxk[:,0:1]).sum().item()
            topk += (test_labels == maxk).sum().item()
    # print(top1,topk,total)
    # model.train()
    return 100 * top1 / total, 100*topk/total


# def evalaa(model, test_loader):
#     model.eval()
#     # i = 0
#     with torch.no_grad():
#         total = 0
#         top1 = 0
#         topk = 0
#         for (test_imgs, test_labels) in tqdm(test_loader):
#             # i+=1
#             # if i>2: break
#             test_labels = test_labels.cuda()
#             out = model(test_imgs.cuda())
#             _,maxk = torch.topk(out,1,dim=-1)
#             total += test_labels.size(0)
#             test_labels = test_labels.view(-1,1) # reshape labels from [n] to [n,1] to compare [n,k]
#             top1 += (test_labels == maxk[:,0:1]).sum().item()
#     # print(top1,topk,total)
#     # model.train()
#     return top1 / total



def evala(model, testdaloader):
    # model.eval()
    # i = 0
    with torch.no_grad():
        total = 0
        top1 = 0
        topk = 0
        for (test_imgs, test_labels) in tqdm(testdaloader):
            # i+=1
            # if i>2: break
            test_labels = test_labels.cuda()
            _,out = model(test_imgs.cuda())
            _,maxk = torch.topk(out,5,dim=-1)
            total += test_labels.size(0)
            test_labels = test_labels.view(-1,1) # reshape labels from [n] to [n,1] to compare [n,k]
            top1 += (test_labels == maxk[:,0:1]).sum().item()
            topk += (test_labels == maxk).sum().item()
    # print(top1,topk,total)
    # model.train()
    return 100 * top1 / total,100*topk/total



def evala_fed_clip(model, visualmodel,testdata):
    with torch.no_grad():
        total = 0
        top1 = 0
        topk = 0
        for test_imgs, test_labels in tqdm(testdata):

            test_labels = test_labels.cuda()
            out = model(visualmodel(test_imgs.cuda()),cls_only=True)
            _,maxk = torch.topk(out,5,dim=-1)
            total += test_labels.size(0)
            test_labels = test_labels.view(-1,1) # reshape labels from [n] to [n,1] to compare [n,k]
            top1 += (test_labels == maxk[:,0:1]).sum().item()
            topk += (test_labels == maxk).sum().item()

    return 100 * top1 / total,100*topk/total




def lr_cos(step):
        # if step < 5:
        #     return float(step) / float(max(1.0, 5))
        # progress after warmup
        progress = float(step) / float(max(
            1, 50))
        return max(
            0.0,
            0.5 * (1. + math.cos(math.pi * 0.5 * 2.0 * progress))
        )
    
    
def make_optimizer_bayes(model,model_2=None,model_3=None,base_lr=0.002,iround=1,WEIGHT_DECAY=1e-5):
    params = []
    meta_params = []
    # only include learnable params
    for key, value in model.named_parameters():
        if value.requires_grad:
            if key.startswith("meta_net") or "meta_net" in key:
                meta_params.append((key, value))
            else:    
                params.append((key, value))
    if model_2!=None:
        for key, value in model_2.named_parameters():
            if value.requires_grad:
                if value.requires_grad:
                    if key.startswith("meta_net") or "meta_net" in key:
                        meta_params.append((key, value))
                    else:
                        params.append((key, value))
    if model_3!=None:
        for key, value in model_3.named_parameters():
            if value.requires_grad:
                if value.requires_grad:
                    if key.startswith("meta_net") or "meta_net" in key:
                        meta_params.append((key, value))
                    else:
                        params.append((key, value))


    _params = []
    for p in params:
        key, value = p
       
        if 'cquery' in key:
            tlr = base_lr#*lr_cos(iround)
        else:
            tlr = base_lr

        _params += [{
            "params": [value],
            "lr": tlr,
            "weight_decay": WEIGHT_DECAY
        }]


    for p in meta_params:
        key, value = p
        tlr = base_lr*5

        _params += [{
            "params": [value],
            "lr": tlr,
            "weight_decay": WEIGHT_DECAY
        }]


    optimizer = torch.optim.SGD(
        _params,momentum=0.9
    )
    return optimizer





def make_optimizer(model,model_2=None,model_3=None,base_lr=0.002,iround=1,WEIGHT_DECAY=1e-5):
    params = []
    # only include learnable params
    for key, value in model.named_parameters():
        if value.requires_grad:
                params.append((key, value))
    if model_2!=None:
        for key, value in model_2.named_parameters():
            if value.requires_grad:
                params.append((key, value))
    if model_3!=None:
        for key, value in model_3.named_parameters():
            if value.requires_grad:
                params.append((key, value))


    _params = []
    for p in params:
        key, value = p
        # print(key)
        # if not value.requires_grad:
        #     continue
        if 'cquery' in key:
            tlr = base_lr#*lr_cos(iround) #if we want diff lr for cquery
        else:
            tlr = base_lr
        weight_decay = WEIGHT_DECAY
        _params += [{
            "params": [value],
            "lr": tlr,
            "weight_decay": weight_decay
        }]

    optimizer = torch.optim.SGD(
        _params,lr = base_lr,momentum=0.9,weight_decay=WEIGHT_DECAY
    )
    return optimizer