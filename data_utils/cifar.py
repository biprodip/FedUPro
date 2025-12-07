import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100

import numpy as np, random, torch
from torch.utils.data import DataLoader
from utils.datasplit import partition_data
from utils.dataset import CIFAR100_truncated

# def get_cifar100_dataset(datapath, batch_size, preprocess, num_clients):
#     """
#     Load CIFAR-100 and create a pathological non-iID partition where each client has exclusive classes.
#     Returns:
#         client_loaders: list of DataLoader for each client's training data
#         client_testloaders: list of DataLoader for each client's test data
#         class_names: list of class names
#     """
#     # Load datasets
#     train_set = CIFAR100(root=datapath, train=True, download=True, transform=preprocess)
#     test_set  = CIFAR100(root=datapath, train=False, download=True, transform=preprocess)

#     # Class names
#     class_names = train_set.classes  # list of 100 class names

#     # Labels as numpy arrays
#     train_labels = np.array(train_set.targets)
#     test_labels  = np.array(test_set.targets)

#     # Shuffle and split classes
#     total_classes = len(class_names)
#     class_indices = np.arange(total_classes)
#     np.random.shuffle(class_indices)
#     classes_per_client = total_classes // num_clients

#     client_loaders = []
#     client_testloaders = []

#     for i in range(num_clients):
#         # Determine class split for this client
#         start = i * classes_per_client
#         end   = (i + 1) * classes_per_client if i < num_clients - 1 else total_classes
#         client_classes = class_indices[start:end]

#         # Get indices of examples belonging to these classes
#         train_idx = np.where(np.isin(train_labels, client_classes))[0]
#         test_idx  = np.where(np.isin(test_labels,  client_classes))[0]

#         # Create subset and DataLoader
#         train_subset = Subset(train_set, train_idx.tolist())
#         test_subset  = Subset(test_set,  test_idx.tolist())

#         train_loader = DataLoader(
#             train_subset,
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=2,
#             pin_memory=True
#         )
#         test_loader = DataLoader(
#             test_subset,
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=2,
#             pin_memory=True
#         )

#         client_loaders.append(train_loader)
#         client_testloaders.append(test_loader)

#     return client_loaders, client_testloaders, class_names



# pfedmoap


def build_cifar100_clients(root, users, batch_size, train_tfms, test_tfms,
                           partition="noniid-labeldir", beta=0.5, seed=2023):

    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)

    (_, _, lab2cname, classnames,
     net_map_train, net_map_test, *_ ) = partition_data(
        dataset="cifar100", datadir=root,
        partition=partition, n_parties=users, beta=beta
    )

    client_tr, client_te = [], []
    for u in range(users):
        ds_tr = CIFAR100_truncated(root, dataidxs=net_map_train[u], train=True,
                                   transform=train_tfms, download=True)
        ds_te = CIFAR100_truncated(root, dataidxs=net_map_test[u], train=False,
                                   transform=test_tfms, download=True)
        client_tr.append(DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=8, pin_memory=True))
        client_te.append(DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True))

    return client_tr, client_te, classnames

