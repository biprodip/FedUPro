from os import path
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
import random
import numpy as np
from tqdm import tqdm


# random.seed(1)
# np.random.seed(1)
# torch.manual_seed(1)
# torch.cuda.manual_seed(1)
# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.deterministic = True




def setup_seed(seed):  # setting up the random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



imgsize = 224
def read_domainnet_class(dataset_path, domain_name, split="train"):
    data_paths = []
    data_labels = []
    for i,dom in enumerate(domain_name):
        split_file = path.join(dataset_path, "splits", "{}_{}.txt".format(dom, split))
        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                data_path, label = line.split(' ')
                data_path = path.join(dataset_path, data_path)
                label = int(i)
                data_paths.append(data_path)
                data_labels.append(label)
    return data_paths, data_labels



def get_domloader(domain_name, batch_size,preprocess, num_workers=16):
    dataset_path = path.join('/home/share/DomainNet')
    train_data_paths, train_data_labels = read_domainnet_class(dataset_path, domain_name, split="train")
    test_data_paths, test_data_labels = read_domainnet_class(dataset_path, domain_name, split="test")

    train_dataset = DomainNet(train_data_paths, train_data_labels, preprocess, domain_name)
    test_dataset = DomainNet(test_data_paths, test_data_labels, preprocess, domain_name)
    # train_dloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
    #                            shuffle=True)
    test_dloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                              shuffle=False)
    return test_dloader



def read_domainnet_data(dataset_path, domain_name, split="train"):
    #read all data of given domain
    data_paths = []
    data_labels = []
    split_file = path.join(dataset_path, "splits", "{}_{}.txt".format(domain_name, split)) #splits/clipart_train
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_path, label = line.split(' ')
            data_path = path.join(dataset_path, data_path)
            data_paths.append(data_path)
            data_labels.append(int(label))
    return data_paths, data_labels



def read_domainnet_all(dataset_path, domain_names, split="train"):
    #read all data of all domain
    data_paths = []
    data_labels = []
    for i,domain_name in enumerate(domain_names):
        split_file = path.join(dataset_path, "splits", "{}_{}.txt".format(domain_name, split))
        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                data_path, label = line.split(' ')
                data_path = path.join(dataset_path, data_path)
                label = int(label)
                data_paths.append(data_path)
                data_labels.append(label)
    return data_paths, data_labels



# def read_data_num(dataset_path, domain_name, split="train", maxnum = 10):
#     data_paths = []
#     data_labels = []
#     split_file = path.join(dataset_path, "splits", "{}_{}.txt".format(domain_name, split))
#     ind = [0]*345
#     with open(split_file, "r") as f:
#         lines = f.readlines()
#         for line in lines:
#             line = line.strip()
#             data_path, label = line.split(' ')
#             data_path = path.join(dataset_path, data_path)
#             label = int(label)
#             if ind[label]==maxnum:
#                 continue
#             else:
#                 data_paths.append(data_path)
#                 data_labels.append(label)
#                 ind[label]+=1
#             if sum(ind)>(maxnum*345+5):break
            
#     return data_paths, data_labels




def read_data_num(dataset_path, domain_name, split="train", maxnum = 10, total_classes = 10):
    #maxnum is per class max data. This function reads maxnum*total_classes data
    data_paths = []
    data_labels = []
    split_file = path.join(dataset_path, "splits", "{}_{}.txt".format(domain_name, split))
    #inside text file clipart/aircraft_carrier/clipart_001_000005.jpg 0
    #filename format clipart_001_000005.jpg
    cnt = [0]*total_classes
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_path, label = line.split(' ') #label after space
            data_path = path.join(dataset_path, data_path)
            label = int(label)
            if (label<total_classes): #labels are sorted
                if (cnt[label]==maxnum):
                    continue
                else:
                    data_paths.append(data_path)
                    data_labels.append(label)
                    cnt[label]+=1
            if sum(cnt)>(maxnum*total_classes+5):break################
            
    return data_paths, data_labels




class DomainNet(Dataset):
    def __init__(self, data_paths, data_labels, transforms, domain_name):
        super(DomainNet, self).__init__()
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms
        self.domain_name = domain_name

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.data_labels[index] 
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.data_paths)




def get_domainnet_dloader(seed, datapath,domain_name, batch_size, preprocess, num_workers=2, total_classes=10):
    setup_seed(seed)
    dataset_path = path.join(datapath)

    # Now generate a random integer
    #for 1 cl per domain, all classes has at least 9 data in common, So we use 9 data
    train_data_num = random.randint(30, 40) #random.randint(10, 20) #per class for this domain
    test_data_num = random.randint(30, 40) #random.randint(10, 20) #per class for this domain


    #read maxnum num of data per class per domain########
    train_data_paths, train_data_labels = read_data_num(dataset_path, domain_name, split="train", maxnum = train_data_num, total_classes=total_classes) ###10
    test_data_paths, test_data_labels = read_data_num(dataset_path, domain_name, split="test", maxnum = test_data_num, total_classes=total_classes) ####10    

    #########uncomment folowing 2 lines for original results################
    # train_data_paths, train_data_labels = read_domainnet_data(dataset_path, domain_name, split="train")
    # test_data_paths, test_data_labels = read_domainnet_data(dataset_path, domain_name, split="test")    
    
    
    # # transforms_train = transforms.Compose([
    # #     transforms.RandomResizedCrop(224, scale=(0.75, 1)),
    # #     transforms.RandomHorizontalFlip(),
    # #     transforms.ToTensor()
    # # ])
    # # transforms_test = transforms.Compose([
    # #     transforms.Resize((224,224)),
    # #     transforms.ToTensor()
    # # ])

    train_dataset = DomainNet(train_data_paths, train_data_labels, preprocess, domain_name)
    test_dataset = DomainNet(test_data_paths, test_data_labels, preprocess, domain_name)
    # train_dloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                               # shuffle=True)
    # test_dloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                              # shuffle=False)
    return train_dataset,test_dataset





def get_domainnet_dloader_all(datapath,domain_names, batch_size,preprocess, num_workers=16):
    dataset_path = path.join(datapath)
    train_data_paths, train_data_labels = read_domainnet_all(dataset_path, domain_names, split="train")
    test_data_paths, test_data_labels = read_domainnet_all(dataset_path, domain_names, split="test")

    train_dataset = DomainNet(train_data_paths, train_data_labels, preprocess, domain_names)
    test_dataset = DomainNet(test_data_paths, test_data_labels, preprocess, domain_names)
    # train_dloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
    #                            shuffle=True)
    # test_dloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
    #                           shuffle=False)
    return train_dataset, test_dataset



#called for multi client
def get_domainnet_dataset(
    seed,
    datapath,
    domain_name, 
    batch_size, 
    preprocess, 
    alpha=0.01, 
    dom_clients=5, 
    num_workers=4, 
    total_classes = 10
    ):
    setup_seed(seed)
    dataset_path = path.join(datapath)

    # Now generate a random integer
    # train_data_num = random.randint(30, 40) #per class for this domain #30 40 for 100 class 100*30 samples
    # test_data_num = random.randint(30, 40)  #per class for this domain

    # train_data_num = 30 #per class for this domain #30 40 for 100 class 100*30 samples
    # test_data_num = 30  #per class for this domain

    train_data_num=5000 #inf
    test_data_num=5000 #inf
    #read maxnum num of data per class per domain######## classes are sorted and selected sequentially
    train_data_paths, train_data_labels = read_data_num(dataset_path, domain_name, split="train", maxnum = train_data_num, total_classes=total_classes) ###10
    test_data_paths, test_data_labels = read_data_num(dataset_path, domain_name, split="test", maxnum = test_data_num, total_classes=total_classes) ####10    
 

    # train_data_paths, train_data_labels = read_domainnet_data(dataset_path, domain_name, split="train")
    # test_data_paths, test_data_labels = read_domainnet_data(dataset_path, domain_name, split="test") #all classes
    
    # train_dataset = DomainNet(train_data_paths, train_data_labels, transforms_train, domain_name)
    test_dataset = DomainNet(test_data_paths, test_data_labels, preprocess, domain_name)

    train_data_labels  = np.array(train_data_labels)
    train_data_paths = np.array(train_data_paths)
    
    # net_dataidx_map,traindata_cls_counts =partition(alpha, dom_clients, train_data_labels)
    net_dataidx_map,traindata_cls_counts =partition(alpha, dom_clients, train_data_labels, total_classes)

    # print(traindata_cls_counts)
    clients = [DomainNet(train_data_paths[net_dataidx_map[idxes]], train_data_labels[net_dataidx_map[idxes]], preprocess, domain_name) for idxes in net_dataidx_map]
    clients_loader = [DataLoader(client, batch_size=batch_size, num_workers=num_workers, pin_memory=True,shuffle=True) \
                      for client in clients]
    test_dloader = DataLoader(test_dataset, batch_size=batch_size*2, num_workers=num_workers, pin_memory=True,
                              shuffle=False)
    
    return clients_loader,test_dloader




def get_domainnet_dataset_label_iid(
    seed,
    datapath,
    domain_name,
    batch_size,
    preprocess,
    dom_clients=5,
    num_workers=4,
    total_classes=10,
    per_class_samples=30,   # fixed per-class budget (same for all clients)
):
    """
    Create domain-nonIID (each group of clients tied to one domain), but label-IID across clients:
    every client gets the SAME number of samples per class.

    Returns:
        clients_loader: List[DataLoader] for dom_clients
        test_dloader:   DataLoader for test split of this domain
    """
    setup_seed(seed)

    dataset_path = path.join(datapath)

    # 1) Read fixed budget per class for TRAIN/TEST from this domain
    train_paths, train_labels = read_data_num(
        dataset_path, domain_name, split="train",
        maxnum=per_class_samples * dom_clients,  # need enough to split evenly
        total_classes=total_classes
    )
    test_paths, test_labels = read_data_num(
        dataset_path, domain_name, split="test",
        maxnum=per_class_samples,
        total_classes=total_classes
    )
    # test_paths, test_labels = read_domainnet_data(dataset_path, domain_name, split="test") #all classes



    # 2) For label-IID, split EACH class evenly across dom_clients
    train_paths = np.array(train_paths)
    train_labels = np.array(train_labels)

    per_client_indices = [[] for _ in range(dom_clients)]
    for c in range(total_classes):
        cls_idx = np.where(train_labels == c)[0]
        np.random.shuffle(cls_idx)

        # ensure equal shards per class
        needed = per_class_samples * dom_clients
        cls_idx = cls_idx[:needed]  # trim if we got a few extras
        shards = np.array_split(cls_idx, dom_clients)
        # enforce exactly per_class_samples per client (in case of minor imbalance)
        for ci in range(dom_clients):
            per_client_indices[ci].extend(shards[ci][:per_class_samples].tolist())


    # # Debug: print number of train samples per class per client
    # print(f"\n=== Distribution for domain: {domain_name} ===")
    # for ci, idxs in enumerate(per_client_indices):
    #     labels_ci = train_labels[idxs]
    #     unique, counts = np.unique(labels_ci, return_counts=True)
    #     print(f"Client {ci}: ", {int(u): int(c) for u, c in zip(unique, counts)})



    # 3) Build per-client datasets and loaders
    clients = [
        DomainNet(train_paths[idxs], train_labels[idxs], preprocess, domain_name)
        for idxs in per_client_indices
    ]
    clients_loader = [
        DataLoader(cset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
        for cset in clients
    ]

    # 4) Test loader (all selected test samples of this domain)
    test_dataset = DomainNet(test_paths, test_labels, preprocess, domain_name)
    test_dloader = DataLoader(test_dataset, batch_size=batch_size*2, num_workers=num_workers,
                              pin_memory=True, shuffle=False)

    return clients_loader, test_dloader






def partition(alphaa, n_netss, y_train,classes=345):
    min_size = 0
    n_nets = n_netss
    N = y_train.shape[0]
    net_dataidx_map = {}
    alpha = alphaa
    K=classes
    while min_size < 10:
        idx_batch = [[] for _ in range(n_nets)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
            ## Balance
            proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return net_dataidx_map,traindata_cls_counts




def record_net_data_stats(y_train, net_dataidx_map):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    return net_cls_counts