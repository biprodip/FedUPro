import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import copy
import random
from tqdm import tqdm
import numpy as np
import argparse
import logging

# from domain import get_domainnet_dataset
from domain_gh import get_domainnet_dataset
from OfficeCaltech10 import get_office_caltech10_dloader
from PromptModels.clip_base import CustomCLIP_client
import clip
from utils import make_optimizer,evala,evalaa
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', default=0.5,type=float,)
    parser.add_argument('--data', default='domainnet',help='domainnet')
    parser.add_argument('--seed', default=1,type=int,)
    parser.add_argument('--num_clients', default=5,type=int,)
    parser.add_argument('--num_classes', default=345,type=int,)
    parser.add_argument('--asyn', action='store_true') #provide only --asyn
    parser.add_argument('--round',  default=50, type=int)
    parser.add_argument('--local_epochs', default=1, type=int, help='local epochs') #######4
    parser.add_argument('--batch_size', default=256,type=int,)
    parser.add_argument('--learning_rate', default=1,type=float,)
    parser.add_argument('--gctx', default=16,type=int,)
    parser.add_argument('--logname', default='basedevice_gtx16',)
    parser.add_argument('--datapath', default='...',)
    parser.add_argument('--choose', default="rand",) #rand or domain

    return parser


parser = get_parser()
args = parser.parse_args()


def setup_seed(seed):  # setting up the random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(args.seed)


# import logging
# logging.basicConfig(
#     filename=f'./logfinal/{args.data}_{args.logname}.log',
#     format='%(asctime)s %(levelname)-8s %(message)s',
#     datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# logger.info(torch.device('cuda'))


    

    
# model, preprocess = clip.load("ViT-B/32", device='cuda')
model, preprocess = clip.load(
    "ViT-B/32",
    device="cuda",
    download_root="/scratch3/pal194/clip_cache"   # <-- your local copy
)

#======================= dataset ==========================================
if args.data == 'domainnet':
    numclass=args.num_classes
    domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']#['clipart','quickdraw',] #
    client_testloaders,client_dataloaders,client_datasets = [],[],[]
    for domain in domains:
        train_data, test_data = get_domainnet_dataset(args.datapath,domain,args.batch_size,preprocess,args.alpha,args.num_clients,8,numclass)
        client_dataloaders+=train_data
        client_testloaders.append(test_data)
        
    out = ['None']*numclass
    with open(os.path.join(args.datapath,'splits/clipart_test.txt')) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_path, label = line.split(' ')
            #########comment following 2 lines, uncomment third line########################
            if int(label)<numclass:  
                out[int(label)] = data_path.split('/')[1]
            # out[int(label)] = data_path.split('/')[1] ##############################
    out  = [name.replace("_", " ") for name in out]

elif args.data == 'office_caltech10':
    numclass=args.num_classes
    domains = ['amazon', 'webcam', 'dslr', "caltech"]
    client_testloaders,client_dataloaders,client_datasets = [],[],[]
    for domain in domains:
        train_data, test_data ,strain,stest= get_office_caltech10_dloader(args.datapath,domain,args.batch_size,preprocess)
        client_datasets.append(train_data)
        client_dataloaders.append(torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=8, pin_memory=True,sampler=strain))
        client_testloaders.append(torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, num_workers=8, pin_memory=True,sampler=stest))
    
    out = ['back pack','bike','calculator','headphones','keyboard','laptop computer','monitor','mouse','mug','projector']




# # ------------------ 2) inject label noise on the fly ------------------
# from collections import defaultdict
# # import numpy as np

# def permute_client_labels(clients_loader, client_ids, seed=0):
#     rng = np.random.default_rng(seed)
#     mapping_log = defaultdict(dict)

#     for cid in client_ids:
#         ds = clients_loader[cid].dataset       # your DomainNet dataset
#         labels = np.array(ds.data_labels)

#         uniq = np.unique(labels)
#         perm = rng.permutation(uniq)
#         # ensure it actually changes at least one label
#         while np.all(perm == uniq):
#             perm = rng.permutation(uniq)

#         mapping = dict(zip(uniq, perm))
#         ds.data_labels = [mapping[l] for l in labels]   # overwrite in‐place

#         mapping_log[cid] = mapping
#         print(f"[Label-perm] client {cid}: {mapping}")

#     return mapping_log

# # say you want to corrupt clients 2 and 4:
# perm_maps = permute_client_labels(client_dataloaders, client_ids=[2, 4], seed=42)










#=======================  Initialize keys and models   ==========================================
global_model = CustomCLIP_client(out,model,args.gctx,domain_number=len(domains)).cuda()
models  = [CustomCLIP_client(out,model,args.gctx,domain_number=len(domains)).cuda() for i in range(len(client_dataloaders))]
for client in models[1:]:
    client.load_state_dict(models[0].state_dict())

for model in models:
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)
   
#=======================     Federated training  ==========================================         
tot_cl = args.num_clients
for fe in tqdm(range(args.round)):
    if args.choose=="random":
        this_round_clients = np.sort(np.random.choice(list(range(tot_cl*len(domains))), 6, replace=False)).tolist()
    else:
        this_round_clients = [np.random.choice(list(range(tot_cl*k,tot_cl*k+tot_cl)), 1, replace=False)[0] for k in range(len(domains))]    # print(this_round_clients)
    # logger.info(f'------------- federated {fe}-th  --------------------------')
    print('------------- federated ',fe,'-th  --------------------------')

    if fe>0:
        for m,client in enumerate(models):
            client.prompt_learner.load_state_dict(global_model.prompt_learner.state_dict(),strict=False)
        
    for cl in this_round_clients:
        local_epochs = args.local_epochs
        
        # if args.asyn:
        #     print('Asynchronous training.')
        #     local_epochs = max(1, int(client.id%local_epochs))
        # else:
        #     print('Synchronous training.')  

        optimizer = make_optimizer(models[cl].prompt_learner,base_lr=0.01)
        for e in range(local_epochs):
            epoch_loss = 0.0  # Initialize a variable to accumulate the loss for the epoch

            for i, (image, label) in enumerate(client_dataloaders[cl]):
                optimizer.zero_grad()
                image = image.to('cuda')
                label = label.to('cuda')
                out_cls = models[cl](image,True)
                loss = F.cross_entropy(out_cls, label)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()  # Add the loss of the current batch to the epoch loss

            if e == (local_epochs-1) or (e == 0):
                print(f'Client {cl} Ep :{e+1}/{local_epochs} Tr_Data:{len(client_dataloaders[cl].dataset)} Tot Loss: {epoch_loss :.4f}')                

            
    weights = [1/len(this_round_clients)]*len(this_round_clients)

    prompt = [] 
    local_state  = models[0].prompt_learner.state_dict()
    for k, cl in enumerate(this_round_clients):
        client_state = models[cl].prompt_learner.state_dict()
        for st in local_state:
            if k==0:
                local_state[st] = client_state[st]*weights[k]
            else:
                local_state[st] += client_state[st]*weights[k]

    global_model.prompt_learner.load_state_dict(local_state,strict=False)


    # torch.save(global_model.prompt_learner, f'./models_newlr/{args.data}_basepromptlr001_{str(fe)}.pth')
    
    if fe==(args.round-1) or (fe%10==0 and fe>0):
        # logger.info('epoch: %s' % str(fe))
        for te,test_loader in enumerate(client_testloaders):
                top1,topk = evalaa(global_model,test_loader)
                # logger.info('top1: %s' % str(top1))
                print('round '+str(fe)+' in client '+str(te)+' acc: ',top1)        
    

# if (args.round==0):
#         # logger.info('epoch: %s' % str(fe))
#         for te,test_loader in enumerate(client_testloaders):
#             top1,topk = evalaa(global_model,test_loader)
#             # logger.info('top1: %s' % str(top1))
#             print('round '+str(0)+' in client '+str(te)+' acc: ',top1)         