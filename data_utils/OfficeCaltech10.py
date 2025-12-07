from os import path
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import random

# np.random.seed(1)
# torch.manual_seed(1)
# torch.cuda.manual_seed(1)
# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.deterministic = True
# random.seed(1234)



def setup_seed(seed):  # setting up the random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



import numpy as np

def _partition_dirichlet(y, n_clients, alpha, n_classes=10):
    """
    Return: dict client_id -> indices, and counts per client.
    """
    min_size = 0
    N = y.shape[0]
    while min_size < 10:
        idx_batch = [[] for _ in range(n_clients)]
        for k in range(n_classes):
            idx_k = np.where(y == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
            # simple balance heuristic
            proportions = np.array([p * (len(idx_j) < N / n_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            cuts = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            split = np.split(idx_k, cuts)
            idx_batch = [idx_j + s.tolist() for idx_j, s in zip(idx_batch, split)]
            min_size = min(len(idx_j) for idx_j in idx_batch)
    net_map = {j: idx_batch[j] for j in range(n_clients)}
    return net_map





def listdir_nohidden(path):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
    """
    return [f for f in os.listdir(path) if not f.startswith('.')]


def read_office_caltech10_data(dataset_path, domain_name):
    data_paths = []
    data_labels = []
    domain_dir = path.join(dataset_path, domain_name)
    class_names = listdir_nohidden(domain_dir)
    class_names.sort()
    for label, class_name in enumerate(class_names):
        class_dir = path.join(domain_dir, class_name)
        item_names = listdir_nohidden(class_dir)
        for item_name in item_names:
            item_path = path.join(class_dir, item_name)
            data_paths.append(item_path)
            data_labels.append(label)
    return data_paths, data_labels



def read_office_caltech10_data_equal(dataset_path, domain_name, sample_per_class=70):
    data_paths = []
    data_labels = []
    domain_dir = path.join(dataset_path, domain_name)
    class_names = listdir_nohidden(domain_dir)
    class_names.sort()
    for label, class_name in enumerate(class_names):
        class_dir = path.join(domain_dir, class_name)
        item_names = listdir_nohidden(class_dir)
        print(len(item_names))
        # ─── new: subsample if more than sample_per_class ───
        if len(item_names) > sample_per_class:
            # random.sample guarantees no repeats
            item_names = random.sample(item_names, sample_per_class)
        # ───────────────────────────────────────────────────────

        for item_name in item_names:
            item_path = path.join(class_dir, item_name)
            data_paths.append(item_path)
            data_labels.append(label)
    return data_paths, data_labels



class OfficeCaltech(Dataset):
    def __init__(self, data_paths, data_labels, transforms, domain_name):
        super(OfficeCaltech, self).__init__()
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms
        self.domain_name = domain_name

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        img = img.convert("RGB")
        label = self.data_labels[index]
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.data_paths)


def get_office_caltech10_split_sampler(labels, test_ratio=0.2, num_classes=10): #0.2
    """
    :param labels: torch.array(long tensor)
    :param test_ratio: the ratio to split part of the data for test
    :param num_classes: 10
    :return: sampler_train,sampler_test
    """
    sampler_test = []
    sampler_train = []
    for i in range(num_classes):
        loc = torch.nonzero(labels == i)
        loc = loc.view(-1)
        # do random perm to make sure uniform sample
        test_num = round(loc.size(0) * test_ratio)
        loc = loc[torch.randperm(loc.size(0))]
        sampler_test.extend(loc[:test_num].tolist())
        sampler_train.extend(loc[test_num:].tolist())
    sampler_test = SubsetRandomSampler(sampler_test)
    sampler_train = SubsetRandomSampler(sampler_train)
    return sampler_train, sampler_test





def get_office_caltech10_dloader(seed,datapath,domain_name, batch_size,preprocess, num_workers=2):
    # dataset_path = path.join(base_path, 'dataset', 'OfficeCaltech10')
    setup_seed(seed)
    
    dataset_path = path.join(datapath)
    
    data_paths, data_labels = read_office_caltech10_data(dataset_path, domain_name)

    #70 sample per class, then 80/20 tr/tst split
    # data_paths, data_labels = read_office_caltech10_data_equal(dataset_path, domain_name, sample_per_class=70)



    train_dataset = OfficeCaltech(data_paths, data_labels, preprocess, domain_name)
    test_dataset = OfficeCaltech(data_paths, data_labels, preprocess, domain_name)
    sampler_train, sampler_test = get_office_caltech10_split_sampler(torch.LongTensor(data_labels))
    # train_dloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
    #                            sampler=sampler_train)
    # test_dloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
    #                           sampler=sampler_test)
    return train_dataset, test_dataset, sampler_train,sampler_test





def get_office_caltech10_dirichlet(
    seed,
    datapath,
    domain_name,
    batch_size,
    preprocess,
    alpha=0.5,
    dom_clients=5,
    num_workers=2
):
    """
    Domain non-IID (one domain per call) + label non-IID via Dirichlet across dom_clients.
    Returns: clients_loader (list), test_loader (held-out split per class within this domain)
    """
    setup_seed(seed)

    dataset_path = path.join(datapath)

    # Read ALL data for this domain
    data_paths, data_labels = read_office_caltech10_data(dataset_path, domain_name)
    data_paths = np.array(data_paths)
    data_labels = np.array(data_labels)

    # Create a train/test split per class (reuse existing helper)
    # We'll construct datasets via samplers consistent with your current loader pattern,
    # but first gather explicit index sets for train/test.
    sampler_train, sampler_test = get_office_caltech10_split_sampler(torch.LongTensor(data_labels))
    train_idx = sampler_train.indices if hasattr(sampler_train, 'indices') else sampler_train
    test_idx  = sampler_test.indices  if hasattr(sampler_test, 'indices')  else sampler_test

    train_y = data_labels[train_idx]

    # Dirichlet partition on TRAIN ONLY
    net_map = _partition_dirichlet(train_y, dom_clients, alpha, n_classes=len(np.unique(data_labels)))

    # Build per-client datasets/loaders (train only)
    clients = []
    for j in range(dom_clients):
        idxs = np.array(train_idx)[net_map[j]]
        clients.append(OfficeCaltech(data_paths[idxs], data_labels[idxs], preprocess, domain_name))

    clients_loader = [
        DataLoader(cset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
        for cset in clients
    ]

    # Test: keep the same per-domain held-out split
    test_dataset = OfficeCaltech(data_paths[test_idx], data_labels[test_idx], preprocess, domain_name)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=True, shuffle=False)

    return clients_loader, test_loader



def get_office_caltech10_label_iid(
    seed,
    datapath,
    domain_name,
    batch_size,
    preprocess,
    dom_clients=5,
    per_class_sample=20,
    num_workers=2
):
    """
    For a single domain, split TRAIN so that each client gets the same number of samples per class
    (label IID). Test split is shared (same as your existing helper).
    """
    setup_seed(seed)

    dataset_path = path.join(datapath)

    # Read all domain data
    data_paths, data_labels = read_office_caltech10_data(dataset_path, domain_name)
    data_paths = np.array(data_paths)
    data_labels = np.array(data_labels)

    # Train/Test split per class
    sampler_train, sampler_test = get_office_caltech10_split_sampler(torch.LongTensor(data_labels))
    train_idx = sampler_train.indices if hasattr(sampler_train, 'indices') else sampler_train
    test_idx  = sampler_test.indices  if hasattr(sampler_test, 'indices')  else sampler_test

    # Build equal-per-class shards across clients
    n_classes = len(np.unique(data_labels))
    train_idx = np.array(train_idx)
    train_y   = data_labels[train_idx]

    per_client_indices = [[] for _ in range(dom_clients)]
    needed_per_class = per_class_sample * dom_clients

    for c in range(n_classes):
        cls_idx_all = train_idx[train_y == c]
        np.random.shuffle(cls_idx_all)
        cls_idx_all = cls_idx_all[:needed_per_class]
        shards = np.array_split(cls_idx_all, dom_clients)
        for ci in range(dom_clients):
            per_client_indices[ci].extend(shards[ci][:per_class_sample].tolist())

    # Build client loaders
    clients = [
        OfficeCaltech(data_paths[idxs], data_labels[idxs], preprocess, domain_name)
        for idxs in per_client_indices
    ]
    clients_loader = [
        DataLoader(cset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
        for cset in clients
    ]

    # Test loader
    test_dataset = OfficeCaltech(data_paths[test_idx], data_labels[test_idx], preprocess, domain_name)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=True, shuffle=False)

    return clients_loader, test_loader
