import os
from data_utils.domain_gh import get_domainnet_dataset, get_domainnet_dataset_label_iid, get_domloader
from data_utils.OfficeCaltech10 import (
    get_office_caltech10_dirichlet,
    get_office_caltech10_label_iid,
)

def load_federated_datasets(args, preprocess):
    """
    Returns: client_dataloaders, client_testloaders, out, domains, numclass
    """

    if args.data == 'domainnet':
        numclass = args.num_classes
        domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        client_testloaders, client_dataloaders, client_datasets = [], [], []

        for domain in domains:
            if args.non_iid == 'FNLN':
                print('Using FNLN non iid setup.')
                train_data, test_data = get_domainnet_dataset(
                    seed = args.seed,
                    datapath = args.datapath,
                    domain_name = domain,
                    batch_size = args.batch_size,
                    preprocess = preprocess,
                    alpha = args.alpha,
                    dom_clients = args.num_clients,
                    num_workers = 2,
                    total_classes = args.num_classes
                )
            elif args.non_iid == 'FNLI':
                print('Using FNLI non iid setup.')
                train_data, test_data = get_domainnet_dataset_label_iid(
                    seed = args.seed,                    
                    datapath = args.datapath,
                    domain_name = domain,
                    batch_size = args.batch_size,
                    preprocess = preprocess,
                    dom_clients = args.num_clients,
                    num_workers=2,
                    total_classes = args.num_classes,
                    per_class_samples=30
                )
            else:
                raise ValueError(f"Non-IID setting '{args.non_iid}' not defined")

            client_dataloaders += train_data
            client_testloaders.append(test_data)

        # get class names
        out = ['None'] * numclass
        with open(os.path.join(args.datapath, 'splits/clipart_test.txt')) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                data_path, label = line.split(' ')
                if int(label) < numclass:
                    out[int(label)] = data_path.split('/')[1]
        out = [name.replace("_", " ") for name in out]

    elif args.data == 'office_caltech10':
        numclass = 10
        domains = ['amazon', 'webcam', 'dslr', "caltech"]
        client_testloaders, client_dataloaders, client_datasets = [], [], []

        for domain in domains:
            if args.non_iid == 'FNLN':
                print('Using FNLN non iid setup.')
                train_loader, test_loader = get_office_caltech10_dirichlet(
                    seed = args.seed,
                    datapath = args.datapath,
                    domain_name = domain,
                    batch_size = args.batch_size,
                    preprocess = preprocess,
                    alpha = args.alpha,
                    dom_clients = args.num_clients,
                    num_workers=2,
                )
            elif args.non_iid == 'FNLI':
                print('Using FNLI non iid setup.')
                train_loader, test_loader = get_office_caltech10_label_iid(
                    seed = args.seed,
                    datapath = args.datapath,
                    domain_name = domain,
                    batch_size = args.batch_size,
                    preprocess = preprocess,
                    dom_clients = args.num_clients,
                    per_class_sample=20,
                    num_workers=2
                )
            else:
                raise ValueError(f"Non-IID setting '{args.non_iid}' not defined")

            client_dataloaders += train_loader
            client_testloaders.append(test_loader)

        out = ['back pack','bike','calculator','headphones','keyboard',
               'laptop computer','monitor','mouse','mug','projector']
    else:
        raise ValueError(f"Dataset '{args.data}' not supported")

    return client_dataloaders, client_testloaders, out, domains, numclass


def print_train_samples_per_class_per_client(client_dataloaders, num_classes):
    import numpy as np
    import torch

    all_counts = []

    for cid, loader in enumerate(client_dataloaders, start=1):
        counts = np.zeros(num_classes, dtype=np.int64)

        for _, labels in loader:
            labels = labels.detach().cpu().numpy().ravel()
            binc = np.bincount(labels, minlength=num_classes)
            counts += binc

        all_counts.append(counts)

        # Build {label: count} dict, but only keep labels with count > 0
        label_counts = {lbl: int(cnt) for lbl, cnt in enumerate(counts) if cnt > 0}
        print(f"Client {cid}: {label_counts}")

    return all_counts