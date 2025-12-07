import torch
import torch.nn.functional as F
import numpy
import copy
import random
from tqdm import tqdm
import numpy as np
import argparse
import logging
# from domain_gh import get_domainnet_dataset,get_domloader

# from OfficeCaltech10 import get_office_caltech10_dloader
# # from PromptModels.clip_queryv2 import CustomCLIP_server,CustomCLIP_client
from PromptModels.clip_query_gh_bayes import CustomCLIP_server,CustomCLIP_client,PromptLearner_client_Bayesian

# import clip
# from utils import WarmupCosineSchedule, make_optimizer,evala,evalaa
# from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
# from utils import * #make_optimizer,evala,evalaa
# from evi_utils_bayes import *
# from evi_helpers import *

 


#===========================   Semantic diversity ===========================================

def compute_prompt_similarity(prompts):
    """
    Computes the pairwise cosine similarity between learned prompts.
    :param prompts: List of prompt tensors (each of shape [n_cls, n_ctx, ctx_dim])
    :return: Similarity matrix of shape [num_clients, num_clients]
    """
    num_clients = len(prompts)
    similarity_matrix = torch.zeros((num_clients, num_clients)).cuda()
    
    for i in range(num_clients):
        for j in range(i, num_clients):
            similarity = F.cosine_similarity(prompts[i].flatten(), prompts[j].flatten(), dim=0)
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Since similarity is symmetric

    return similarity_matrix


def compute_uncertainty_from_similarity(similarity_matrix):
    """
    Compute epistemic uncertainty based on the similarity matrix.
    Low similarity -> High uncertainty (divergent prompts)
    High similarity -> Low uncertainty (convergent prompts)
    :param similarity_matrix: Pairwise similarity matrix (shape: [num_clients, num_clients])
    :return: Uncertainty score for each client
    """
    # Calculate the average similarity for each client
    client_similarity = similarity_matrix.mean(dim=1)
    print(f'Similarity matrix:\n {similarity_matrix}')
    print(f'Client similarity: {client_similarity}')
    # Compute uncertainty as the inverse of similarity (or the variance of similarity)
    # Higher similarity means lower uncertainty
    uncertainty = 1 / client_similarity  # we can also compute variance or other measures of dispersion
    
    return uncertainty


#===========================    Prototype vectors  and Global head training   ==================================================


def compute_prototypes(client_dataloader, model, num_classes, device='cuda'):
    """ Function to compute prototypes for each client based on their local data """
    # model.eval()  # Set model to evaluation mode
    prototypes = {i: [] for i in range(num_classes)}  # Initialize a dictionary to store prototypes per class
    
    model.cquery.eval()
    model.prompt_learner.eval()

    with torch.no_grad():
        for images, labels in client_dataloader:
            images, labels = images.to(device), labels.to(device)
            # print(f'Dim of batch Images: {images.shape}, Labels: {labels.shape}') #32,3,224,224    32
            # Get image features and predictions
            image_features, _, out_dom, out_cls = model(images, True) 

            # Normalize the features before computing prototypes
            # features = image_features / image_features.norm(dim=-1, keepdim=True)  # Normalize per sample
            features = image_features   # Normalize per sample
            # print(f'Dim of Features: {features.shape}, Domain: {out_dom.shape} Class: {out_cls.shape}') #32,512   32,6      32,50


            for label, feature in zip(labels, features):
                # print(f'Prototype for label {label.item()} has shape: {feature.shape}')                
                prototypes[label.item()].append(feature.cpu().numpy())  # Store feature in appropriate class list

    # Compute the averaged prototypes for each class, only storing those with features
    avg_prototypes = {}
    for class_id in range(num_classes):
        if len(prototypes[class_id]) > 0:  # Only store prototype if there are features for this class
            avg_prototypes[class_id] = np.mean(prototypes[class_id], axis=0)
            # Normalize the prototype
            avg_prototypes[class_id] = avg_prototypes[class_id] / np.linalg.norm(avg_prototypes[class_id])
    # print(f'Total classes : {num_classes} Tot prototypes: {len(avg_prototypes)}\n')
    return avg_prototypes




def train_global_cquery(global_model, prototype_dataloader, num_epochs=2, learning_rate=0.01, device='cuda'):
    global_model.cquery.train()  # Set cquery to training mode

    optimizer = torch.optim.SGD(global_model.cquery.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()  # Using CrossEntropyLoss for domain classification

    for epoch in range(num_epochs):
        running_loss = 0.0
        for prototypes, labels, domain_ids in prototype_dataloader:
            prototypes = prototypes.to(device)
            labels = labels.to(device)
            domain_ids = domain_ids.to(device)

            # Forward pass through the cquery module
            optimizer.zero_grad()
            # print(prototypes.shape)  # Ensure this matches (batch_size, 512)
            # print(prototypes[0].type(global_model.dtype))
            outputs = global_model.cquery(prototypes.type(global_model.dtype))  # cquery takes prototypes as input
            # print(f'Domain Predictions:{outputs}')

            # Compute loss for domain classification
            loss = criterion(outputs, domain_ids)  # Use domain_ids for domain classification

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if epoch == 0 or epoch == num_epochs-1:
            print(f'Global Cquery training: Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(prototype_dataloader):.4f}')
