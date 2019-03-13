# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from RateDistAttentionGMM2 import RateDistAttentionGMM2
from DynamicRNN import DynamicRNN
# from embed2Dist import embed2Dist
# from torch.distributions.distribution import Distribution
from torch.nn.parameter import Parameter

class bi_lstm_trans_gmm(nn.Module):
    '''
    Decoding the sentences in feedbacks
    Inout: sentences
    Output: sentence vectors, feedback vector and attention values
    '''
    def __init__(self,
                 dim_word_input, dim_sentence_hidden, dim_doc_hidden,
                 dim_user_product_input, dim_user_product_hidden,
                 dim_user_doc_hidden, dim_product_doc_hidden, dim_user_product_doc_hidden,
                 init_usr_centers, init_pdr_centers, num_cluster,
                 doc_embed_list, usr_embed_list, pdr_embed_list,
                 usr_rate_dist, pdr_rate_dist,
                 n_layers, n_classes,
                 embed_dropout_rate, cell_dropout_rate,
                 bidirectional, rnn_type, use_cuda):

        super(bi_lstm_trans_gmm, self).__init__()
        self.n_layers = n_layers
        self.dim_word_input = dim_word_input
        self.dim_sentence_hidden = dim_sentence_hidden
        self.dim_doc_hidden = dim_doc_hidden

        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.use_cuda = use_cuda
        self.n_classes = n_classes
        self.num_cluster = num_cluster

        self.usr_rate_dist = usr_rate_dist
        self.pdr_rate_dist = pdr_rate_dist

        self.embed_dropout = nn.Dropout(embed_dropout_rate)
        self.doc_embed = nn.Embedding(num_embeddings=doc_embed_list.shape[0], embedding_dim=doc_embed_list.shape[1])
        self.usr_embed = nn.Embedding(num_embeddings=usr_embed_list.shape[0], embedding_dim=usr_embed_list.shape[1])
        self.pdr_embed = nn.Embedding(num_embeddings=pdr_embed_list.shape[0], embedding_dim=pdr_embed_list.shape[1])

        # self.usr_embed2dist = embed2Dist(usr_embed_list.shape[1], self.n_classes)
        # self.pdr_embed2dist = embed2Dist(pdr_embed_list.shape[1], self.n_classes)

        self.sentence_encoder = DynamicRNN(input_size=dim_word_input, hidden_size=self.dim_sentence_hidden, num_layers=n_layers, dropout=cell_dropout_rate if n_layers > 1 else 0,
                                           bidirectional=bidirectional, rnn_type=rnn_type, use_cuda=use_cuda)
        self.word_atten = RateDistAttentionGMM2(dim_input_hidden=self.dim_sentence_hidden * (2 if bidirectional else 1), n_class=n_classes, use_cuda=use_cuda)

        self.doc_encoder = DynamicRNN(self.dim_sentence_hidden * (2 if bidirectional else 1), self.dim_doc_hidden, n_layers, dropout=cell_dropout_rate if n_layers > 1 else 0,
                                      bidirectional=bidirectional, rnn_type=rnn_type, use_cuda=use_cuda)

        self.sen_atten = RateDistAttentionGMM2(self.dim_doc_hidden * (2 if bidirectional else 1),  n_class=n_classes, use_cuda=use_cuda)
        # self.p_sen_atten = ClusterAttention(self.dim_doc_hidden * (2 if bidirectional else 1), dim_usr=dim_user_product_input, dim_pdr=dim_user_product_input, n_center=num_cluster,
        #                                     n_class=n_classes, use_cuda=use_cuda)
        # map user and product into a probablity space
        self.usr_linear = nn.Linear(dim_user_product_input, self.n_classes)
        self.pdr_linear = nn.Linear(dim_user_product_input, self.n_classes)
        self.usr_pdr_out = nn.Linear(self.n_classes * 2, self.n_classes)

        self.doc_linear = nn.Linear(in_features=(self.dim_doc_hidden * (2 if bidirectional else 1) + self.n_classes), out_features=self.n_classes, bias=True)

        # init weights
        self.init_weights(doc_embed_list, usr_embed_list, pdr_embed_list)
        ignored_params = list(map(id, self.doc_embed.parameters())) + list(map(id, self.usr_embed.parameters())) + list(map(id, self.pdr_embed.parameters()))
        self.base_params = filter(lambda p: id(p) not in ignored_params, self.parameters())

    def init_random_centers(self):
        rdn_seed = torch.Tensor(self.num_cluster, self.n_classes).uniform_(1, 100)
        dist = torch.div(rdn_seed, torch.sum(rdn_seed, dim=1, keepdim=True))
        dist = Parameter(dist)
        return dist

    def init_weights(self, doc_embed_list, usr_embed_list, pdr_embed_list):
        self.doc_embed.weight.data.copy_(torch.from_numpy(doc_embed_list))
        self.usr_embed.weight.data.copy_(torch.from_numpy(usr_embed_list))
        self.pdr_embed.weight.data.copy_(torch.from_numpy(pdr_embed_list))

    def find_nearest_centers(self, rate_dist, centers):
        rate_dist_row, rate_dim = rate_dist.shape
        centers_row, center_dim = centers.shape
        centers_exp = torch.FloatTensor(rate_dist_row, centers_row, center_dim)
        rate_dist_exp = torch.FloatTensor(rate_dist_row, centers_row, rate_dim)

        for i in range(0, rate_dist_row):
            centers_exp[i, :] = centers.expand(1, centers_row, center_dim)

        for i in range(0, rate_dist_row):
            rate_dist_exp[i, :] = rate_dist[i, :].expand(centers_row, rate_dim)

        centers_exp = centers_exp.view(-1, center_dim)
        rate_dist_exp = rate_dist_exp.view(-1, rate_dim)

        kl_rate_dist_centers = F.kl_div(rate_dist_exp.log(), centers_exp, reduction='none').sum(dim=1)
        kl_rate_dist_centers = kl_rate_dist_centers.view(rate_dist_row, centers_row)
        nearest_centers_idx = torch.argmin(kl_rate_dist_centers, dim=1)
        nearest_centers_idx = Variable(nearest_centers_idx, requires_grad=False)
        nearest_centers_idx = nearest_centers_idx.cuda() if self.use_cuda else nearest_centers_idx
        nearest_centers = torch.index_select(input=centers, index=nearest_centers_idx, dim=0)

        return nearest_centers.clone()

    def compute_intra_centers_kl_loss(self, centers):
        n_row, rate_dim = centers.shape
        centers_exp_0 = centers.data.new(n_row, n_row, rate_dim)
        centers_exp_1 = centers.data.new(n_row, n_row, rate_dim)
        for i in range(0, n_row):
            centers_exp_0[i, :] = centers.expand(1, n_row, rate_dim)
            centers_exp_1[i, :] = centers[i, :].expand(n_row, rate_dim)

        centers_exp_0 = centers_exp_0.view(-1, rate_dim)
        centers_exp_1 = centers_exp_1.view(-1, rate_dim)
        mean_kl = F.kl_div(centers_exp_0.log(), centers_exp_1, reduction='mean')

        return mean_kl

    def aline_to_sen_doc(self, dict_inst, input_vec):
        # nearest usr_dist for words, sentences
        input_vectors_sentence = input_vec.data.new(size=(len(dict_inst['len_docs']) * max(dict_inst['len_docs']), input_vec.shape[1]))
        input_vector_word = input_vec.data.new(size=(len(dict_inst['len_sens']) * max(dict_inst['len_sens']), input_vec.shape[1]))
        idx_begin_doc, idx_end_doc = 0, 0
        idx_begin_word, idx_end_word = 0, 0
        for i, len_sen in enumerate(dict_inst['len_docs']):
            idx_begin_doc, idx_end_doc = idx_end_doc, idx_end_doc + max(dict_inst['len_docs'])
            input_vectors_sentence[idx_begin_doc:idx_end_doc] = input_vec[i]
            idx_begin_word, idx_end_word = idx_end_word, idx_end_word + len_sen * max(dict_inst['len_sens'])
            input_vector_word[idx_begin_word: idx_end_word] = input_vec[i]

        return input_vector_word, input_vectors_sentence

    def forward(self, dict_inst):
        doc_embedded = self.doc_embed(dict_inst['docs'])
        usr_embeded = self.usr_embed(dict_inst['usrs'])
        pdr_embeded = self.pdr_embed(dict_inst['prds'])

        doc_embedded = self.embed_dropout(doc_embedded)
        # usr_embeded = self.embed_dropout(usr_embeded)
        # pdr_embeded = self.embed_dropout(pdr_embeded)

        # doc_embedded = self.embed_dropout(doc_embedded)

        # input_usr_dist = dict_inst['usr_rate_dist']
        # input_pdr_dist = dict_inst['pdr_rate_dist']
        # input_usr_dist = self.usr_embed2dist(usr_embeded)
        # input_pdr_dist = self.pdr_embed2dist(pdr_embeded)

        # input_usr_dist_centers = self.find_nearest_centers(rate_dist=input_usr_dist, centers=self.usr_centers)  # no update center here, by clone value
        # input_pdr_dist_centers = self.find_nearest_centers(rate_dist=input_pdr_dist, centers=self.pdr_centers)  # no update center here, by clone value

        # input_usr_dist_centers = torch.clamp(input=input_usr_dist_centers, min=1e-6)
        # input_pdr_dist_centers = torch.clamp(input=input_pdr_dist_centers, min=1e-6)

        # nearest usr_dist for words, sentences
        # nearest_u_centers_sentence = input_usr_dist_centers.data.new(size=(len(dict_inst['len_docs']) * max(dict_inst['len_docs']), input_usr_dist_centers.shape[1]))
        # nearest_u_centers_word = input_usr_dist_centers.data.new(size=(len(dict_inst['len_sens']) * max(dict_inst['len_sens']), input_usr_dist_centers.shape[1]))
        # idx_begin_doc, idx_end_doc = 0, 0
        # idx_begin_word, idx_end_word = 0, 0
        # for i, len_sen in enumerate(dict_inst['len_docs']):
        #     idx_begin_doc, idx_end_doc = idx_end_doc, idx_end_doc + max(dict_inst['len_docs'])
        #     nearest_u_centers_sentence[idx_begin_doc:idx_end_doc] = input_usr_dist_centers[i]
        #     idx_begin_word, idx_end_word = idx_end_word, idx_end_word + len_sen * max(dict_inst['len_sens'])
        #     nearest_u_centers_word[idx_begin_word: idx_end_word] = input_usr_dist_centers[i]

        # look for closest centers for each input usr, pdr by KL, return clone to avoid update in cluster attention
        # sentence encoder
        # input_usr_dist_word, input_usr_dist_sentence = self.aline_to_sen_doc(dict_inst=dict_inst, input_vec=usr_embeded)

        pdr_out = self.pdr_linear(pdr_embeded)
        usr_out = self.usr_linear(usr_embeded)
        pdr_usr_prob = self.usr_pdr_out(torch.cat((pdr_out, usr_out), dim=1))
        pdr_usr_prob = F.softmax(pdr_usr_prob, dim=-1)

        input_dist_word, input_dist_sentence = self.aline_to_sen_doc(dict_inst=dict_inst, input_vec=pdr_usr_prob)
        # input_pdr_dist_word, input_pdr_dist_sentence = self.aline_to_sen_doc(dict_inst=dict_inst, input_vec=pdr_usr_prob)

        output_pad_sens, hidden = self.sentence_encoder(doc_embedded, lengths=dict_inst['len_sens'], flag_ranked=False)
        v_s_a, v_s_a_score = self.word_atten(inputs=output_pad_sens, input_dist=input_dist_word, input_lengths=torch.LongTensor(dict_inst['len_sens']))
        idx_begin, idx_end = 0, 0
        u_out_dims = (len(dict_inst['len_docs']), max(dict_inst['len_docs'])) + v_s_a.size()[1:]
        # p_out_dims = (len(dict_inst['len_docs']), max(dict_inst['len_docs'])) + p_v_s_a.size()[1:]
        u_out_tensor = v_s_a[0].data.new(*u_out_dims).fill_(0)
        # p_out_tensor = p_v_s_a[0].data.new(*p_out_dims).fill_(0)

        for i, len_current in enumerate(dict_inst['len_docs']):
            idx_begin, idx_end = idx_end, idx_end + len_current
            u_out_tensor[i, :len_current, ...] = v_s_a[idx_begin: idx_end]

        # document encoder
        output_pad_doc, hidden_doc = self.doc_encoder(u_out_tensor, lengths=dict_inst['len_docs'], flag_ranked=False)
        vs_d_doc_a, vs_d_doc_a_score = self.sen_atten(inputs=output_pad_doc, input_dist=input_dist_sentence, input_lengths=torch.LongTensor(dict_inst['len_docs']))

        # doc_and_user = torch.cat((usr_embeded, pdr_embeded), dim=1)
        # doc_and_user_out = self.usr_pdr_linear(doc_and_user)
        # doc_and_user_out = F.relu(doc_and_user_out)
        # prob = F.log_softmax(self.doc_linear(doc_and_user), dim=-1)
        doc_and_dist = torch.cat((vs_d_doc_a, pdr_usr_prob), dim=1)
        prob = F.log_softmax(self.doc_linear(doc_and_dist), dim=-1)

        # joint learning : usr + doc, pdr + docsel
        # doc_and_user = torch.cat((u_vs_d_doc_a, usr_embeded), dim=1)
        # doc_and_product = torch.cat((u_vs_d_doc_a, pdr_embeded), dim=1)
        # doc_and_user_out = F.relu(self.doc_linear_1(doc_and_user))
        # doc_and_product_out = F.relu(self.doc_linear_1(doc_and_product))

        # out = self.doc_linear_3(torch.cat((doc_and_user_out, doc_and_product_out), dim=1))
        # prob = F.log_softmax(out, dim=-1)

        # update center losses
        # 1. pairwise KL for usr centers, pdr centers
        # 2. KL between center, and input usr dist
        # kl_intra_u_center_loss = self.compute_intra_centers_kl_loss(self.usr_centers) # maximize
        # kl_u_center_loss = F.kl_div(input_usr_dist.log(), input_usr_dist_centers) # minimize
        return prob #, kl_intra_u_center_loss, kl_u_center_loss
