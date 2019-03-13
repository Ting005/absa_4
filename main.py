# -*- coding: utf-8 -*-

import time, random, logging, argparse
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from tensorboard_logger import configure, log_value
import torch
from DataHelper import Dataset, load_embedding, load_user_product_embeddings
from bridgeModelHuapa import bridgeModelHuapa

parser = argparse.ArgumentParser()
# parser.add_argument('--voc_size', type=int, default=32768)
parser.add_argument('--use_attention', type=bool, default=True, choices=[True, False])

parser.add_argument('--dim_word', type=int, default=200, choices=[100, 200, 300])
parser.add_argument('--dim_sen_hidden', type=int, default=256, choices=[128, 256, 512])
parser.add_argument('--dim_doc_hidden', type=int, default=512, choices=[128, 256, 512])
parser.add_argument('--dim_pre_usr_pdr_input', type=int, default=300, choices=[100, 200, 300])
parser.add_argument('--dim_pre_usr_pdr_hidden', type=int, default=256, choices=[128, 256, 512])

parser.add_argument('--dim_user_doc_hidden', type=int, default=512, choices=[128, 256, 512])
parser.add_argument('--dim_product_doc_hidden', type=int, default=512, choices=[128, 256, 512])
parser.add_argument('--dim_user_product_doc_hidden', type=int, default=512, choices=[128, 256, 512])

parser.add_argument('--num_cluster', type=int, default=30, choices=[10, 20, 30, 50])
parser.add_argument('--n_layer', type=int, default=2)
parser.add_argument('--n_class', type=int, default=5, choices=[5, 10])
parser.add_argument('--bidirectional', type=bool, default=True)
parser.add_argument('--learning_rate', type=float, default=1e-3) # 1e-3 by default
parser.add_argument('--lr_word_vector', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--embed_dropout', type=float, default=0.5)
parser.add_argument('--cell_dropout', type=float, default=0.5)
parser.add_argument('--final_dropout', type=float, default=0.5)
parser.add_argument('--lambda1', type=float, default=1e-3)
# parser.add_argument('--k_centers', type=float, default=10 * 3)

parser.add_argument('--max_sen_len', type=int, default=50)
parser.add_argument('--max_doc_len', type=int, default=40)

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_epochs', type=int, default=32*4056)
parser.add_argument('--per_checkpoint', type=int, default=32*8)# around 5.5 per epoch, 26 steps / epoch for imdb
parser.add_argument('--warm_up_steps', type=int, default=32*8*0)# around 5.5 per epoch

parser.add_argument('--seed', type=int, default=810)
parser.add_argument('--rnn_type', type=str, default="LSTM", choices=["LSTM", "GRU"])
parser.add_argument('--optim_type', type=str, default="Adam", choices=["Adam", "Adadelta", "RMSprop", "Adagrad"]) #AMSGrad
parser.add_argument('--data_dir', type=str, default='./acl2015')
parser.add_argument('--breakpoint', type=int, default=-1)
parser.add_argument('--path_character', type=str, default='vocab-c')
parser.add_argument('--dataset', type=str, default='yelp13', choices=['imdb', 'yelp13', 'yelp14'])
parser.add_argument('--use_rdn_ctr', type=bool, default=True, choices=[True, False])

# parser.add_argument('--name_model', type=str, default='Trans_attention_doc_and_user_docdropout')
parser.add_argument('--name_model', type=str, default='RateDistAttentionGMM2')


FLAGS = parser.parse_args()
print(FLAGS)
print(FLAGS.name_model)
# print('use_rdn_ctr:', FLAGS.use_rdn_ctr)

np.random.seed(FLAGS.seed)
random.seed(FLAGS.seed)
torch.manual_seed(FLAGS.seed)
# torch.backends.cudnn.enabled = False
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed(FLAGS.seed)

model_path = '{}/{}'.format(FLAGS.dataset, FLAGS.name_model)

print(model_path)
print(use_cuda)


def get_data_path():
    data_set_name = FLAGS.dataset
    if data_set_name == 'yelp13':
        word_embed_path = FLAGS.data_dir + '/yelp13/yelp-2013-embedding-200d.txt'
        usr_pdr_embed_path = FLAGS.data_dir + '/yelp13/yelp13_mf.pkl'
        data_path = [FLAGS.data_dir + '/yelp13/yelp-2013-seg-20-20.train.ss', FLAGS.data_dir + '/yelp13/yelp-2013-seg-20-20.dev.ss', FLAGS.data_dir + '/yelp13/yelp-2013-seg-20-20.test.ss']
    elif data_set_name == 'yelp14':
        word_embed_path = FLAGS.data_dir + '/yelp14/yelp-2014-embedding-200d.txt'
        usr_pdr_embed_path = FLAGS.data_dir + '/yelp14/yelp14_mf.pkl'
        data_path = [FLAGS.data_dir + '/yelp14/yelp-2014-seg-20-20.train.ss', FLAGS.data_dir + '/yelp14/yelp-2014-seg-20-20.dev.ss', FLAGS.data_dir + '/yelp14/yelp-2014-seg-20-20.test.ss']
    elif data_set_name == 'imdb':
        word_embed_path = FLAGS.data_dir + '/imdb/imdb-embedding-200d.txt'
        usr_pdr_embed_path = FLAGS.data_dir + '/imdb/imdb_mf.pkl'
        data_path = [FLAGS.data_dir + '/imdb/imdb.train.txt.ss', FLAGS.data_dir + '/imdb/imdb.dev.txt.ss', FLAGS.data_dir + '/imdb/imdb.test.txt.ss']

    return word_embed_path, usr_pdr_embed_path, data_path


def evaluate(model, eval_data):
    y_true = []
    all_prob = []
    all_losses = []
    # all_prd_loss = []
    # all_kl_intra = []
    # all_kl_dist = []
    #loss.data.cpu().numpy(), lst_probs.data.cpu().numpy(), kl_intra.data.cpu().numpy(), kl_dis.data.cpu().numpy()
    #lst_probs.data.cpu().numpy(), [prd_loss.data.cpu().numpy(), kl_intra.data.cpu().numpy(), kl_dis.data.cpu().numpy()]
    for idx, batched_data in enumerate(eval_data.gen_batched_data):
        y_true.extend([b[4] for b in batched_data])
        probs, losses = model.predict(batched_data)
        all_prob.append(probs)
        all_losses.append(losses)
        # all_prd_loss.append(prd_loss)
        # all_kl_intra.append(kl_intra)
        # all_kl_dist.append(kl_dist)

    all_prob = np.concatenate(all_prob, axis=0)
    # mean_prd_loss = np.mean(np.asarray(all_prd_loss), axis=0)
    # mean_kl_intra = np.mean(all_kl_intra)
    # mean_kl_dist = np.mean(all_kl_dist)
    loss = np.mean(all_losses)
    pred_vector = np.argmax(all_prob, axis=1)
    y_true = np.squeeze(np.array(y_true))

    c_m = confusion_matrix(y_true=y_true, y_pred=pred_vector)
    f1 = f1_score(y_true, pred_vector, average=None)
    acc = accuracy_score(y_true, pred_vector)

    return loss, acc, f1, c_m


class JointModel(object):
    def __init__(self):
        logging.basicConfig(filename='log/%s.log' % model_path, level=logging.DEBUG, format='%(asctime)s %(filename)s %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
        logging.info(f'model parameters: {str(FLAGS)}')
        logging.info(f"Use cuda: {use_cuda}")
        data_paths = get_data_path()

        self.dataset_name = ('train', 'valid', 'test')
        self.data = dict()
        doc_embed, vocab_dict = load_embedding(data_paths[0], FLAGS.dim_word)
        [usr_dict, pdr_dict], [user_embed, product_embed], [usr_rate_dist, pdr_rate_dist] = load_user_product_embeddings(file_path=data_paths[1])
        assert user_embed.shape[1] == FLAGS.dim_pre_usr_pdr_input

        self.train_set = Dataset(data_paths[2][0], FLAGS.max_doc_len, FLAGS.max_sen_len, vocab_dict, usr_dict, pdr_dict, usr_rate_dist_list=usr_rate_dist, pdr_rate_dist_list=pdr_rate_dist)
        self.valid_set = Dataset(data_paths[2][1], FLAGS.max_doc_len, FLAGS.max_sen_len, vocab_dict, usr_dict, pdr_dict, usr_rate_dist_list=usr_rate_dist, pdr_rate_dist_list=pdr_rate_dist)
        self.test_set = Dataset(data_paths[2][2], FLAGS.max_doc_len, FLAGS.max_sen_len, vocab_dict, usr_dict, pdr_dict, usr_rate_dist_list=usr_rate_dist, pdr_rate_dist_list=pdr_rate_dist)

        int_usr_centers = np.random.normal(size=(FLAGS.num_cluster, FLAGS.dim_pre_usr_pdr_input))
        int_pdr_centers = np.random.normal(size=(FLAGS.num_cluster, FLAGS.dim_pre_usr_pdr_input))

        self.model = bridgeModelHuapa(
                     model_name=FLAGS.name_model,
                     dim_word_input=FLAGS.dim_word, dim_sen_hidden=FLAGS.dim_sen_hidden, dim_doc_hidden=FLAGS.dim_doc_hidden,
                     dim_user_product_input=FLAGS.dim_pre_usr_pdr_input, dim_user_product_hidden=FLAGS.dim_pre_usr_pdr_hidden,
                     dim_user_doc_hidden=FLAGS.dim_user_doc_hidden, dim_product_doc_hidden=FLAGS.dim_product_doc_hidden, dim_user_product_doc_hidden=FLAGS.dim_user_product_doc_hidden,
                     init_usr_centers=int_usr_centers, init_pdr_centers=int_pdr_centers, num_cluster=FLAGS.num_cluster,
                     n_layers=FLAGS.n_layer, n_classes=FLAGS.n_class,
                     batch_size=FLAGS.batch_size, max_length_sen=FLAGS.max_sen_len,
                     learning_rate=FLAGS.learning_rate, lr_word_vector=FLAGS.lr_word_vector,
                     weight_decay=FLAGS.weight_decay,
                     doc_embed=doc_embed, usr_embed=user_embed, pdr_embed=product_embed,
                     usr_rate_dist=usr_rate_dist, pdr_rate_dist=pdr_rate_dist,
                     embed_dropout_rate=FLAGS.embed_dropout, cell_dropout_rate=FLAGS.cell_dropout, final_dropout_rate=FLAGS.final_dropout,
                     bidirectional=FLAGS.bidirectional,
                     optim_type=FLAGS.optim_type,
                     rnn_type=FLAGS.rnn_type,
                     lambda1=FLAGS.lambda1,
                     use_cuda=use_cuda)

    def train(self, breakpoint=-1):
        # train_usrdict, train_prddict = self.train_set.get_usr_prd_dict()
        # comment for testing
        self.train_set.genBatch(FLAGS.batch_size)
        self.valid_set.genBatch(FLAGS.batch_size)
        self.test_set.genBatch(FLAGS.batch_size)
        train_batches = self.train_set.batch_iter(FLAGS.batch_size, FLAGS.num_epochs)

        configure("summary/%s" % model_path, flush_secs=3)

        # if breakpoint > 0:
        #     self.model.load_model('./model/%s' % FLAGS.name_model, FLAGS.breakpoint)

        start_iter = 1 if breakpoint < 0 else (breakpoint * FLAGS.per_checkpoint + 1)
        loss_step = np.zeros((1, 1))
        start_time = time.time()
        for step, batched_data in enumerate(train_batches):
            if step < start_iter: continue # continue from previous break point
            if step % FLAGS.per_checkpoint == 0 and step > FLAGS.warm_up_steps:
                func_show = lambda a: '[%s]' % (' '.join(['%.4f' % x for x in a]))
                time_step = time.time() - start_time
                logging.info('-' * 40)
                logging.info('Time of iter training %.2f s' % time_step)
                # logging.info("On iter step %s:, global step %d Loss-step %s" % (step / FLAGS.per_checkpoint, step, func_show(np.squeeze([loss_step]))))
                # self.model.save_model(f"./model/{FLAGS.dataset}/{FLAGS.name_model}", int(step / FLAGS.per_checkpoint))
                # evaluate
                for (eval_data, name) in zip([self.train_set, self.valid_set, self.test_set], ['train', 'dev', 'test']):
                    loss, acc, f1, c_m = evaluate(self.model, eval_data)
                    #[mean_prd_loss, mean_kl_intra, mean_kl_dist], [acc], f1, c_m
                    log_value(f'loss-prd-{name}', loss, int(step / FLAGS.per_checkpoint))
                    # log_value(f'loss-dist-{name}', loss[1], int(step / FLAGS.per_checkpoint))
                    # log_value(f'loss-dist_2-{name}', loss[2], int(step / FLAGS.per_checkpoint))
                    log_value(f'acc-{name}', acc, int(step / FLAGS.per_checkpoint))
                    logging.info(f"In dataset {name}: Loss is {loss}, Accuracy is {acc}, F1 is {func_show(np.squeeze(f1))}")
                    logging.info(f"\n{c_m}")

                start_time = time.time()
                loss_step = np.zeros((1, 1))

            prd_loss, losses = self.model.stepTrain(batched_data)# usr, prd, docs, label, sen_len, doc_len
            loss_step = np.add(losses, loss_step)

    def load_model(self, path):
        self.model.load_model(path)

if __name__ == "__main__":
    jm = JointModel()
    jm.train(FLAGS.breakpoint)
