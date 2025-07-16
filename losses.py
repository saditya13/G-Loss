import torch
from torch import nn
import torch.nn.functional as F
import time
from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLoss, BatchHardTripletLossDistanceFunction
from typing import Dict, Iterable, Any
from torch import Tensor
from sentence_transformers.util import fullname
import random 
import optuna
EPS = EPS = 1e-9


### G-Loss ### 
def normalize_adj(adj):
    rowsum = torch.sum(adj, dim=1).to_dense()
    rowsum = rowsum.to(torch.complex64)
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    ret = adj.mm(d_mat_inv_sqrt).transpose(0, 1).mm(d_mat_inv_sqrt)
    return ret

def guassian(emb, sigma):
    sq_dists = torch.cdist(emb, emb, p=2) ** 2
    weight = torch.exp(-sq_dists / (2 * sigma ** 2))
    weight = weight - torch.diag(torch.diag(weight))
    return weight

def drop_edges(adj, noise):
    # print(f"Dropping {noise*100}% edges")
    num_edges = adj.nonzero().size(0)
    num_drop = int(num_edges * noise)
    edge_index = adj.nonzero(as_tuple=False)
    drop_indices = random.sample(range(num_edges), num_drop)
    adj[edge_index[drop_indices, 0], edge_index[drop_indices, 1]] = 0
    return adj

def modified_lpa(train_emb, test_emb, Ytrain, iter, sigma,num_labels, device,noise, trial=None):
    n_val = len(test_emb)
    emb = torch.cat((train_emb, test_emb), dim=0)
    num_nodes = emb.shape[0]
    # labels = torch.cat((torch.tensor(Ytrain, dtype=torch.float32, device=device), 
    #                 torch.zeros(test_emb.shape[0], dtype=torch.float32, device=device)), dim=0)
    labels = torch.cat((Ytrain, torch.zeros(test_emb.shape[0], device=device)), dim=0)
    # print(type(num_nodes), type(num_labels), type(labels), type(Ytrain), type(n_val))
    Y = torch.zeros((num_nodes, num_labels)).to(device)
    Y = Y.to(torch.complex64)
    for k in range(num_labels):
        Y[labels == k, k] = 1
    labels = Y
    train_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
    train_mask[:Ytrain.shape[0]] = 1
    test_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
    test_mask[Ytrain.shape[0]:Ytrain.shape[0]+n_val]=1

    if (torch.sum(torch.isnan(emb))):
        print("Trial Pruned 1")
        raise optuna.exceptions.TrialPruned()
    
    emb = emb / emb.norm(dim=1, keepdim=True)
    adj = guassian(emb, sigma).to(device)
    if (torch.sum(torch.isnan(adj))):
        print("Trial Pruned 2")
        raise optuna.exceptions.TrialPruned()
    
    adj = adj.to(torch.complex64)
    adj = adj + adj.t()
    adj = normalize_adj(adj)
    adj = adj.to_dense()
    if noise is not None:
        adj = drop_edges(adj, noise)

    if (torch.sum(torch.isnan(adj))):
        print("Trial Pruned 3")
        raise optuna.exceptions.TrialPruned()

    if (torch.sum(torch.isnan(Y))):
        print("Trial Pruned 4")
        raise optuna.exceptions.TrialPruned()
    
    F = torch.zeros_like(Y, dtype=torch.complex64)
    F[train_mask] = Y[train_mask]
    F[test_mask] = 0
    
    for i in range(iter):
        F = adj.mm(F)
        if (torch.sum(torch.isnan(F))):
            raise optuna.exceptions.TrialPruned()
        F /= torch.sum(F, axis=1, keepdims=True) + EPS
        F[train_mask] = Y[train_mask]
        if trial is not None:
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
    F = F.real + EPS
    F /= torch.sum(F, axis=1, keepdims=True)
    return F[test_mask]

def predict_lpa(model, input_ids_lpa, attention_mask_lpa, labels, iter, sigma, num_labels, gamma, device,class_weights,noise=None, trial=None): 
    outputs = model(input_ids_lpa, attention_mask_lpa)[0][:, 0]
    # Extract CLS token for ALL batch items
    # pred_embedding = outputs[0][:, 0] # [batch_size, hidden_size]
    pred_embedding = outputs
    # LPA Processing
    mask1 = torch.randperm(pred_embedding.size(0)) < pred_embedding.size(0) * gamma
    mask2 = ~mask1
    emb_lab_set = pred_embedding[mask1]
    emb_eval_set = pred_embedding[mask2]

    labels_lab_set = labels[mask1]
    labels_eval_set = labels[mask2]
    if trial:
        predicted_labels = modified_lpa(emb_lab_set, emb_eval_set, labels_lab_set, iter, sigma, num_labels, device=device,noise=noise,trial=trial)
    else:
        predicted_labels = modified_lpa(emb_lab_set, emb_eval_set, labels_lab_set, iter, sigma, num_labels, device=device,noise=noise)
    loss = F.cross_entropy(predicted_labels, labels_eval_set, weight=torch.tensor(class_weights, dtype=torch.float32).to(device))

    return loss


### Supervised contrastive loss ###
def supervised_contrastive_loss(model, input_ids, attention_mask, labels, temperature):
    outputs = model(input_ids, attention_mask)
    # outputs = model(input_ids, attention_mask, output_hidden_states=True)
    embeddings = outputs[0][:, 0]  # Extract CLS token embeddings
    embeddings = F.normalize(outputs.last_hidden_state, p=2, dim=-1)

    loss = SupConLoss(temperature=temperature)(embeddings, labels)
    return loss
    # batch_size = embeddings.size(0)
    # labels = labels.unsqueeze(1) #vertical column
    # mask = torch.eq(labels, labels.T).float().to(embeddings.device)
    # mask.fill_diagonal_(0) # exclude self-similarity

    # # Compute the dot product and divide by temperature
    # dot_product = torch.matmul(embeddings, embeddings.T) / temperature
    # exp_dot = torch.exp(dot_product)  # Compute the exp(dot_product) for all samples
    # exp_dot = exp_dot * (1 - torch.eye(batch_size, device=embeddings.device))  # remove self-similarity - Zero out the diagonal (self-similarity) 

    # # Sum of exp values for all samples except itself
    # denominator = exp_dot.sum(dim=1, keepdim=True) + EPS  # Add eps to avoid division by zero
    # # positive_dot = exp_dot * mask  # Keep only the positive pairs
    # # numerator = positive_dot.sum(dim=1, keepdim=True)  # Sum of exp values for all positive pairs
    
    # log_probs = torch.log(exp_dot+EPS) - torch.log(denominator + EPS)  # Add eps to log for numerical stability
    # loss = - (mask * log_probs).sum(dim=1) / mask.sum(dim=1).clamp(min=EPS)  # Avoid division by zero with clamp
    # return loss.mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device('cuda')

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



### BatchAllTriplet ###
class BatchAllTripletLoss(nn.Module):
    def __init__(
        self,
        model,  # This will be your AutoModel instance
        distance_metric=BatchHardTripletLossDistanceFunction.eucledian_distance,
        margin: float = 5,
    ) -> None:
        super(BatchAllTripletLoss, self).__init__()
        self.model = model
        # self.sentence_embedder = model.bert_model
        self.sentence_embedder = model
        self.triplet_margin = margin
        self.distance_metric = distance_metric
        self.current_epoch_loss = 0

    def forward(self, input_ids: torch.Tensor=None, attention_mask: torch.Tensor=None, labels: torch.Tensor=None,emb:torch.Tensor=None,eval_mode=False) -> torch.Tensor:
        if eval_mode:
            print(emb.shape)
            batch_loss = self.batch_all_triplet_loss(labels,emb)
        else:
            outputs = self.sentence_embedder(input_ids, attention_mask)[0][:, 0]
            # print(outputs[0].shape)
            # Get the CLS token embedding as sentence representation
            rep = outputs  # [batch_size, hidden_size]
            batch_loss = self.batch_all_triplet_loss(labels, rep)
            # self.current_epoch_loss += batch_loss.item()
        return batch_loss

    def batch_all_triplet_loss(self, labels: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        pairwise_dist = self.distance_metric(embeddings)
        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.triplet_margin
        mask = BatchHardTripletLoss.get_triplet_mask(labels)  # Put to zero the invalid triplets
        triplet_loss = mask.float() * triplet_loss
        triplet_loss[triplet_loss < 0] = 0         # Remove negative losses (i.e. the easy triplets)
        valid_triplets = triplet_loss[triplet_loss > 1e-16]  # Count number of positive triplets
        num_positive_triplets = valid_triplets.size(0)
        triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
        return triplet_loss



### Custom BatchCosineSimilarityLoss ###
class BatchCosineSimilarityLoss(nn.Module):
    def __init__(self, model,
                 loss_fct: nn.Module = nn.MSELoss(), 
                 cos_score_transformation=nn.Identity()):
        super(BatchCosineSimilarityLoss, self).__init__()
        # self.model = model.bert_model
        self.model = model
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Tensor) -> Tensor:
        outputs = self.model(input_ids, attention_mask)
        embeddings = outputs[0][:, 0]  # [batch_size, hidden_size]
        batch_loss = self.cos_similarity_loss(labels,embeddings)
        # self.model.current_epoch_loss = self.model.current_epoch_loss + batch_loss.item()
        return batch_loss
    def cos_similarity_loss(self, labels:Tensor, embeddings: Tensor) -> Tensor:
        """
        create pairs inside the batch and then compute cosine similarity loss
        """
        i, j = torch.triu_indices(len(embeddings), len(embeddings), 1)
        train_pairs = torch.stack((embeddings[i], embeddings[j]), dim=0)  # Shape: [ 2, num_pairs,embedding_dim]
        train_labels = (labels[i] == labels[j]).float()          # Shape: [num_pairs]
        output = self.cos_score_transformation(torch.cosine_similarity(train_pairs[0], train_pairs[1]))
        loss = self.loss_fct(output, train_labels.float().view(-1))
        return loss
    def get_config_dict(self) -> dict[str, Any]:
        return {"loss_fct": fullname(self.loss_fct)}
