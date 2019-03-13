#-*- coding: utf-8 -*-
#author: Zhen Wu

import numpy as np
import pickle


def load_embedding(embedding_file_path, embedding_dim):
    words_dict = dict()
    word_embedding = []
    index = 1
    words_dict['$EOF$'] = 0  # add EOF
    word_embedding.append(np.zeros(embedding_dim))
    with open(embedding_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for idx, line in enumerate(f):
            if idx > 100: break
            check = line.strip().split()
            if len(check) == 2 or len(check) > embedding_dim + 1:
                continue
            embedding = [float(s) for s in check[-embedding_dim:]]
            word_embedding.append(embedding)
            words_dict[check[0]] = index
            index += 1
    return np.asarray(word_embedding), words_dict


def load_user_product_embeddings(file_path):
    data_ = pickle.load(open(file_path, 'rb'))
    user_embed = data_['user_embed']
    product_embed = data_['product_embed']
    usr_rate_dist = data_['usr_rate_dist']
    pdr_rate_dist = data_['pdr_rate_dist']
    usr_dict = data_['usr_dict']
    pdr_dict = data_['pdr_dict']
    # usr_pdr_rate_mx = data_['usr_pdr_rate_mx']

    usr_empty_embedding = np.zeros((1, user_embed.shape[1]))
    pdr_empty_embedding = np.zeros((1, product_embed.shape[1]))
    user_embed = np.concatenate((usr_empty_embedding, user_embed), axis=0)
    product_embed = np.concatenate((pdr_empty_embedding, product_embed), axis=0)

    return [usr_dict, pdr_dict], [user_embed, product_embed], [usr_rate_dist, pdr_rate_dist]


class Dataset(object):
    def __init__(self, data_file, max_doc_len, max_sen_len, vocab_dict, usr_dict, pdr_dict, usr_rate_dist_list, pdr_rate_dist_list):
        self.t_usr = []
        self.t_prd = []
        self.t_usr_rate_dist = []
        self.t_pdr_rate_dist = []
        self.t_label = []
        self.t_docs = []
        self.t_docs_len = []

        with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
            for idx, line in enumerate(f):
                if idx > 100: break
                fields = line.strip().split('\t\t')
                review = fields[3].lower()
                doc_idx, doc_len = self.doc_to_index(review, vocab_dict, max_doc_len, max_sen_len)
                if len(doc_len) > 0:
                    usr_idx = self.usr_to_indx(usr_dict, fields[0])
                    pdr_idx = self.pdr_to_idx(pdr_dict, fields[1])
                    self.t_usr.append(usr_idx)
                    self.t_prd.append(pdr_idx)

                    usr_rate_dist = self.get_rate_dist(usr_dict, fields[0], usr_rate_dist_list)
                    pdr_rate_dist = self.get_rate_dist(pdr_dict, fields[1], pdr_rate_dist_list)

                    self.t_usr_rate_dist.append(usr_rate_dist)
                    self.t_pdr_rate_dist.append(pdr_rate_dist)

                    self.t_label.append(int(fields[2])-1)
                    self.t_docs.extend(doc_idx)
                    self.t_docs_len.extend(doc_len)

        self.t_usr = np.asarray(self.t_usr)
        self.t_prd = np.asarray(self.t_prd)

        self.t_usr_rate_dist = np.asarray(self.t_usr_rate_dist)
        self.t_pdr_rate_dist = np.asarray(self.t_pdr_rate_dist)

        self.t_label = np.asarray(self.t_label)
        self.t_docs = np.asarray(self.t_docs)
        self.t_docs_len = np.asarray(self.t_docs_len)
        self.data_size = len(self.t_usr)
        # gen batch

        self.gen_batched_data = []

    def doc_to_index(self, doc, vocab_dict, max_doc_len, max_sen_len):
        doc_len = []
        doc_idx = []
        sentences = doc.split('<sssss>')
        sen_word_idx = []
        sen_len = []
        for _idx, sen in enumerate(sentences):
            if len(sen_word_idx) >= max_doc_len: break  # max no. of sentences per doc
            word_idx = []
            words = sen.strip().split()
            for word in words:
                if len(word_idx) >= max_sen_len: break  # max no. of words per sentence
                if vocab_dict.get(word, None) is not None:
                    word_idx.append(vocab_dict[word])

            if len(word_idx) > 0:
                sen_len.append(len(word_idx))
                # pad sen_word_idx to max_len
                word_idx = word_idx + [0] * (max_sen_len - len(word_idx))
                sen_word_idx.append(word_idx)

        if len(sen_word_idx) > 0:
            doc_len.append(sen_len)
            doc_idx.append(sen_word_idx)

        return doc_idx, doc_len

    def usr_to_indx(self, usr_dict, usr):
        # increase idx by 1, 1st role is empty usr, check load usr embedding
        usr_idx = usr_dict.get(usr, None)
        usr_idx = usr_idx + 1 if usr_idx is not None else 0
        return usr_idx

    def pdr_to_idx(self, pdr_dict, pdr):
        # increase idx by 1, 1st role is empty usr, check load pdr embedding
        pdr_idx = pdr_dict.get(pdr, None)
        pdr_idx = 0 if pdr_idx is None else pdr_idx + 1
        return pdr_idx

    def get_rate_dist(self, pdr_or_usr_dict, usr_or_pdr, pdr_usr_rate_dist):
        pdr_or_usr_random_dist = [round(1 / pdr_usr_rate_dist.shape[1], 4)] * pdr_usr_rate_dist.shape[1]
        pdr_or_usr_idx = pdr_or_usr_dict.get(usr_or_pdr, None)
        if pdr_or_usr_idx is not None:
            pdr_or_usr_random_dist = pdr_usr_rate_dist[pdr_or_usr_idx, :]
        return pdr_or_usr_random_dist

    # def get_usr_prd_dict(self):
    #     usrdict, prddict = dict(), dict()
    #     usridx, prdidx = 0, 0
    #     for u in self.t_usr:
    #         if u not in usrdict:
    #             usrdict[u] = usridx
    #             usridx += 1
    #     for p in self.t_prd:
    #         if p not in prddict:
    #             prddict[p] = prdidx
    #             prdidx += 1
    #     # return usrdict, prddict

    def genBatch(self, batch_size):
        data_size = len(self.t_docs)
        num_steps = data_size // batch_size + (1 if data_size % batch_size else 0)
        for i in range(num_steps):
            usr = self.t_usr[i*batch_size:(i+1)*batch_size]
            prd = self.t_prd[i*batch_size:(i+1)*batch_size]

            t_usr_rate_dist = self.t_usr_rate_dist[i*batch_size:(i+1)*batch_size]
            t_pdr_rate_dist = self.t_pdr_rate_dist[i*batch_size:(i+1)*batch_size]

            label = self.t_label[i*batch_size:(i+1)*batch_size]
            docs = self.t_docs[i*batch_size:(i+1)*batch_size]
            docs_len = self.t_docs_len[i*batch_size:(i+1)*batch_size]
            # b_docs, b_doc_len = fit_transform(self.t_docs[i*batch_size:(i+1)*batch_size], wordsdict, max_sen_len, max_doc_len)
            batch_data = list(zip(usr, prd, t_usr_rate_dist, t_pdr_rate_dist, label, docs, docs_len))
            self.gen_batched_data.append(batch_data)
        print('total batches:', len(self.gen_batched_data))

    def batch_iter(self, batch_size, num_epochs, shuffle=True):
        data_size = len(self.t_docs)
        num_batches_per_epoch = data_size // batch_size + (1 if data_size % batch_size else 0)

        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                self.t_usr = self.t_usr[shuffle_indices]
                self.t_prd = self.t_prd[shuffle_indices]
                self.t_label = self.t_label[shuffle_indices]
                self.t_docs = self.t_docs[shuffle_indices]
                self.t_docs_len = self.t_docs_len[shuffle_indices]
                self.t_usr_rate_dist = self.t_usr_rate_dist[shuffle_indices]
                self.t_pdr_rate_dist = self.t_pdr_rate_dist[shuffle_indices]

            for batch_num in range(num_batches_per_epoch):
                start = batch_num * batch_size
                end = min((batch_num + 1) * batch_size, data_size)
                usr = self.t_usr[start:end]
                prd = self.t_prd[start:end]
                usr_rate_dist = self.t_usr_rate_dist[start:end]
                pdr_rate_dist = self.t_pdr_rate_dist[start:end]

                docs = self.t_docs[start:end]
                docs_len = self.t_docs_len[start:end]
                # label = np.eye(n_class, dtype=np.float32)[self.t_label[start:end]]
                label = self.t_label[start:end]
                batch_data = zip(usr, prd, usr_rate_dist, pdr_rate_dist, label, docs, docs_len)

                # docs, doc_len = fit_transform(self.t_docs[start:end], vocab_dict, max_sen_len, max_doc_len)
                # batch_data = zip(usr, prd, docs, label, doc_len)
                yield batch_data
