# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division

import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from myutils import pdist
from embed2Dist import embed2Dist
import torch.nn.functional as F
# from torch.nn.modules.distance import CosineSimilarity, PairwiseDistance

class RateDistAttentionGMM(nn.Module):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, dim_input_hidden, n_class, use_cuda=False):
        """ Initialize the attention layer
        # Arguments:
            attention_size: Size of the attention vector.
        """
        super(RateDistAttentionGMM, self).__init__()
        self.dim_input_hidden = dim_input_hidden
        self.use_cuda = use_cuda
        self.n_class = n_class
        self.ffn1 = embed2Dist(dim_input_hidden, self.n_class)
        # self.ffn1 = nn.Linear(dim_input_hidden, self.n_class)
        # self.ffn2 = embed2Dist(self.n_class * 2, self.n_class)
        # self.W_H_D = Parameter(torch.FloatTensor(dim_input_hidden, self.n_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim_input_hidden)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = '{name}({attention_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs, input_lengths, input_dist):
        """ Forward pass.
        # Arguments:
            inputs (Torch.Variable): Tensor of input sequences
            input_lengths (torch.LongTensor): Lengths of the sequences
        # Return:
            representations and attentions.
        """
        max_len = inputs.size(1)
        idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        mask = Variable((idxes < input_lengths.unsqueeze(1)).float(), requires_grad=False)
        mask = mask.cuda() if self.use_cuda else mask
        # usr_dist = self.pdr_embed2dist(input_usr_dist)
        # pdr_dist = self.pdr_embed2dist(input_pdr_dist)

        # 1. how can I learn more active embed2dist function ? the final layer backwards ..
        # 2. reflects my ideas in loss function ..

        # cat_usr_pdr_dist = torch.cat((input_usr_dist, input_pdr_dist), dim=1)
        # usr_pdr_dist = self.ffn2(cat_usr_pdr_dist)
        usr_pdr_dist = input_dist.view(-1, inputs.shape[1], self.n_class)
        inputs_dist = self.ffn1(inputs)
        # score_temp = F.cosine_similarity(inputs_dist, usr_pdr_dist, dim=2)
        # dist = F.pairwise_distance(inputs_dist, usr_pdr_dist, keepdim=False)
        inputs_dist = torch.clamp(inputs_dist, min=1e-8)
        usr_pdr_dist = torch.clamp(usr_pdr_dist, min=1e-8)
        score_temp = F.kl_div(inputs_dist.log(), usr_pdr_dist, reduction='none').sum(dim=2)
        score_temp = -1 * score_temp
        # normalize
        score_temp = (score_temp - score_temp.max()).ex()
        score_temp_mask = score_temp * mask
        score_temp_sum = torch.sum(score_temp, dim=-1, keepdim=True)
        attended_weights = score_temp_mask.div(score_temp_sum)

        weighted_inputs = inputs.mul(attended_weights.unsqueeze(-1).expand_as(inputs))
        weighted_inputs = weighted_inputs.sum(dim=1)
        return weighted_inputs, attended_weights
        # score_temp_mask = score_temp * mask
        # F.softmin(score_temp_mask)
        # score_temp = -1 * score_temp
        # cos = CosineSimilarity(dim=2)
        # test = cos(inputs_dist, usr_pdr_dist)

        # inputs_dist = inputs_dist.unsqueeze(-2)
        # w_h_d_expand = self.W_H_D.expand((inputs.shape[0], self.W_H_D.shape[0], self.W_H_D.shape[1]))
        # score_temp = torch.squeeze(inputs_dist.matmul(usr_pdr_dist))
        # score_temp = torch.squeeze(inputs_dist.bmm(w_h_d_expand).unsqueeze(-2).matmul(usr_pdr_dist))
        # usr_dist = self.linear2(input_usr)
        # inputs_dist = inputs_dist.view(-1, self.n_class)
        # user weights
        # norm_dim = torch.Tensor([input_usr.shape[-1]]).cuda() if self.use_cuda else torch.Tensor([input_usr.shape[-1]])
        # torch.mm(inputs_dist, input_usr.transpose(0, 1)
        # u_alpha = torch.div().sum(dim=1), torch.sqrt(norm_dim))
        # u_alpha = inputs_dist_rep
        # u_alpha_temp = (u_alpha - u_alpha.max()).exp().view(mask.shape)
        # u_alpha_temp_mask =None
        # u_alpha_temp_mask = score_temp * mask
        # u_alpha_temp_mask_sum = torch.clamp(u_alpha_temp_mask.sum(dim=1, keepdim=True), min=1e-6)
        # u_alpha_temp_mask_sum = u_alpha_temp_mask.sum(dim=1, keepdim=True)
        # attended_weights = u_alpha_temp_mask.div(u_alpha_temp_mask_sum)
        # u_alpha_norm = torch.div(u_alpha_temp_mask, torch.sum(u_alpha_temp_mask, dim=1, keepdim=True))
        # u_alpha_norm_mask = u_alpha_temp_mask * mask
        # weighted_inputs = inputs.mul(attended_weights.unsqueeze(-1).expand_as(inputs))
        # weighted_inputs = weighted_inputs.sum(dim=1)
        # weighted_inputs = self.layer_norm(weighted_inputs) # layer norm doesn't work

        # return weighted_inputs, attended_weights


        # product
        # pdr_embed_dist = F.sigmoid(self.pdr_embed2dist(inputs))
        # p_alpha = pdr_embed_dist.matmul(input_pdr_dist_centers)
        # p_alpha_temp = (p_alpha.log() - p_alpha.max()).exp()
        # p_alpha_norm = torch.div(p_alpha_temp, torch.sum(p_alpha_temp, dim=1, keepdim=True))

        # # 1. find nearest centers, 2. compute mean centers, 3. compute triplet loss
        # usr_center_dists = pdist(usr_input, self.usr_centers)
        # norm_usr_center_dists_temp = (usr_center_dists - usr_center_dists.max()).exp()
        # usr_center_similarity = norm_usr_center_dists_temp.div(torch.sum(norm_usr_center_dists_temp, dim=1, keepdim=True))
        # usr_normal_center = usr_center_similarity.matmul(self.usr_centers)
        #
        # pdr_center_dists = pdist(pdr_input, self.pdr_centers)
        # norm_pdr_center_dists_temp = (pdr_center_dists - pdr_center_dists.max()).exp()
        # pdr_center_similarity = norm_pdr_center_dists_temp.div(torch.sum(norm_pdr_center_dists_temp, dim=1, keepdim=True))
        # product_normal_center = pdr_center_similarity.matmul(self.pdr_centers)
        #
        # usr_min_centers_idx = torch.argmin(usr_center_similarity, dim=1, keepdim=True)
        # usr_min_dist = torch.gather(input=usr_center_similarity, index=usr_min_centers_idx, dim=1).squeeze()
        # usr_tol_dist = torch.sum(usr_center_similarity, dim=1, keepdim=True).squeeze()
        # usr_trip_loss = 1 + (usr_min_dist - (usr_tol_dist - usr_min_dist)) # margin = 1, normalized triplet loss by usr-center distance
        #
        # pdr_min_centers_idx = torch.argmin(pdr_center_similarity, dim=1, keepdim=True)
        # pdr_min_dist = torch.gather(input=pdr_center_similarity, index=pdr_min_centers_idx, dim=1).squeeze()
        # pdr_tol_dist = torch.sum(pdr_center_similarity, dim=1, keepdim=True).squeeze()
        # pdr_trip_loss = 1 + (pdr_min_dist - (pdr_tol_dist - pdr_min_dist))  # margin = 1, normalized triplet loss by usr-center distance
        #
        # max_len = inputs.size(1)
        # idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        # mask = Variable((idxes < input_lengths.unsqueeze(1)).float())
        # mask = mask.cuda() if self.use_cuda else mask
        #
        # input_logit = inputs.matmul(self.W_H)
        # usr_center_logits = usr_normal_center.matmul(self.W_U_C).view(input_logit.shape)
        # pdr_center_logits = product_normal_center.matmul(self.W_P_C).view(input_logit.shape)
        # # test = self.W_C_B.expand_as(input_logit)
        # # test2 = self.W_V_B.expand_as(input_logit)
        # usr_input_weight = torch.tanh(torch.tanh(input_logit + usr_center_logits + pdr_center_logits).matmul(self.W_V))
        # usr_input_weight_norm = (usr_input_weight - usr_input_weight.max()).exp()
        # usr_input_weight_norm_mask = usr_input_weight_norm * mask
        #
        # att_sums = usr_input_weight_norm_mask.sum(dim=1, keepdim=True)  # sums per sequence
        # attentions = usr_input_weight_norm_mask.div(att_sums)
        # weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))
        # representations = weighted.sum(dim=1)
        #
        # # product & user center distance,
        # # mean_udr_center_dists = torch.mean(pdist(self.usr_centers, self.usr_centers))
        # # mean_pdr_center_dists = torch.mean(pdist(self.pdr_centers, self.pdr_centers))
        # return representations, attentions, [torch.mean(usr_trip_loss), torch.mean(pdr_trip_loss)]


        '''
        # usr_input_weight = inputs.matmul(self.W_H.unsqueeze(-1).expand_as(expand_shape) + weighted_usr_centers.matmul(self.W_C.unsqueeze(-1).expand_as(expand_shape)))\
        #     .matmul(self.W_V.unsqueeze(-1).expand_as(expand_shape))
        # usr_input_weight_temp = (usr_input_weight - usr_input_weight.max()).exp()
        # usr_input_weight_temp_mask = usr_input_weight_temp * mask
        # usr_clus_atten = torch.div(usr_input_weight_temp_mask, usr_input_weight_temp_mask.sum())
        # usr_clus_weighted_input = torch.mul(inputs, usr_clus_atten.unsqueeze(-1).expand_as(inputs))

        # logits = inputs.matmul(self.attention_vector)
        # unnorm_ai = (logits - logits.max()).exp()
        # # print(unnorm_ai.shape)
        # # Compute a mask for the attention on the padded sequences
        # # See e.g. https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/5
        # max_len = unnorm_ai.size(1)
        # idxes = torch.arange(0, max_len, out=torch.LongTensor(max_len)).unsqueeze(0)
        # mask = Variable((idxes < input_lengths.unsqueeze(1)).float())
        # mask = mask.cuda() if self.use_cuda else mask

        # apply mask and renormalize attention scores (weights)
        # masked_weights = unnorm_ai * mask
        # att_sums = masked_weights.sum(dim=1, keepdim=True)  # sums per sequence
        # attentions = masked_weights.div(att_sums)

        # apply attention weights
        '''
        # get the final fixed vector representations of the sentences
        # return representations, attentions
