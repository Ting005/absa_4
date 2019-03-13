# ref https://github.com/albertauyeung/matrix-factorization-in-python
#     http://activisiongamescience.github.io/2016/01/11/Implicit-Recommender-Systems-Biased-Matrix-Factorization/
# matrix factorization with product, user bias
import time, random, logging, argparse
import collections
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from tensorboard_logger import configure, log_value
import os
import pickle
import collections
from sklearn.decomposition import NMF
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./acl2015')
parser.add_argument('--dim_hidden', type=float, default=300)
# parser.add_argument('--dim_usr_input', type=float, default=4818)
# parser.add_argument('--dim_prd_input', type=float, default=4194)
parser.add_argument('--alpha', type=float, default=1e-2)
parser.add_argument('--beta', type=float, default=1e-2)
parser.add_argument('--iterations', type=int, default=int(1))
parser.add_argument('--threshold', type=float, default=1e-4)
parser.add_argument('--seed', type=int, default=88)


FLAGS = parser.parse_args()
np.random.seed(FLAGS.seed)
random.seed(FLAGS.seed)
print(FLAGS)


def generate_user_item_rating_matrix_(_train_path, _n_class):
    train_path = FLAGS.data_dir + _train_path
    # dev_path = FLAGS.data_dir + '/imdb/imdb.dev.txt.ss'
    # test_path = FLAGS.data_dir + '/imdb/imdb.test.txt.ss'
    # train_path = FLAGS.data_dir + '/yelp-2014-seg-20-20.test.ss'
    # dev_path = FLAGS.data_dir + '/yelp-2014-seg-20-20.test.ss'

    # get user, product dictionary
    usrdict, prddict = dict(), dict()
    data_set = dict()
    for path, name in zip([train_path], ['train']):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            usridx, prdidx = 0, 0
            data_usr, data_pdr, data_label = [], [], []
            for idx, line in enumerate(f):
                fields = line.strip().split('\t\t')[0:3]
                usr = fields[0]
                pdr = fields[1]
                lbl = int(fields[2]) - 1
                if usr not in usrdict:
                    usrdict[usr] = usridx
                    usridx += 1
                if pdr not in prddict:
                    prddict[pdr] = prdidx
                    prdidx += 1
                data_usr.append(usrdict.get(usr, -1))
                data_pdr.append(prddict.get(pdr, -1))
                data_label.append(lbl)

            data_set[name] = (data_usr, data_pdr, data_label)

    dim_user, dim_prd = len(usrdict), len(prddict)
    user_product_rating_mx = np.zeros((dim_user, dim_prd), dtype=float)
    train_data = data_set['train']

    for usr, pdr, label in zip(train_data[0], train_data[1], train_data[2]):
        user_product_rating_mx[usr, pdr] = label

    user_rating_distribution = np.zeros(shape=(len(usrdict), _n_class))
    product_rating_distribution = np.zeros(shape=(len(prddict), _n_class))

    for u_idx in range(len(usrdict)):
        all_votes = user_product_rating_mx[u_idx, :]
        rating_cnt = [0] * _n_class
        tol_ = 0
        for rating, tol_count in collections.Counter([r for r in all_votes]).items():
            if rating > 0:
                rating_cnt[int(rating) - 1] = tol_count
                tol_ += tol_count
        user_rating_distribution[u_idx, :] = [round(r / (1 if tol_ == 0 else tol_), 4) for r in rating_cnt]

    for p_idx in range(len(prddict)):
        all_votes = user_product_rating_mx[:, p_idx]
        rating_cnt = [0] * _n_class
        tol_ = 0
        for rating, tol_count in collections.Counter([r for r in all_votes]).items():
            if rating > 0:
                rating_cnt[int(rating) - 1] = tol_count
                tol_ += tol_count
        product_rating_distribution[p_idx, :] = [round(r / (1 if tol_ == 0 else tol_), 4) for r in rating_cnt]

    return [usrdict, prddict], [user_rating_distribution, product_rating_distribution], user_product_rating_mx


class MF():
    def __init__(self, R, K, alpha, beta, iterations, threshold):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """

        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.threshold = threshold

        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1. / self.K, size=(self.num_users, self.K)) # users
        self.Q = np.random.normal(scale=1. / self.K, size=(self.num_items, self.K)) # products

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

    def train(self):
        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if (i + 1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i + 1, mse))

            if mse <= self.threshold:
                break

        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i, :])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j, :])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return self.b + self.b_u[:, np.newaxis] + self.b_i[np.newaxis:, ] + self.P.dot(self.Q.T)


if __name__ == '__main__':
    # without matrix factorization
    [usr_dict, pdr_dict], [usr_rate_dist, pdr_rate_dist], usr_pdr_rate_mx = generate_user_item_rating_matrix_(_train_path='/yelp14/yelp-2014-seg-20-20.train.ss', _n_class=5)

    mf = MF(R=usr_pdr_rate_mx, K=FLAGS.dim_hidden, alpha=FLAGS.alpha, beta=FLAGS.beta, iterations=FLAGS.iterations, threshold=FLAGS.threshold)
    train_process = mf.train()
    user_embed = mf.P
    product_embed = mf.Q
    # product_user_mat = user_product_mat.transpose()
    pickle.dump({'user_embed': user_embed, 'product_embed': product_embed, 'usr_rate_dist': usr_rate_dist, 'pdr_rate_dist': pdr_rate_dist,
                 'usr_dict': usr_dict, 'pdr_dict': pdr_dict, 'usr_pdr_rate_mx': usr_pdr_rate_mx}, open(FLAGS.data_dir + '/yelp14/yelp14_mf.pkl', 'wb'))

    data = pickle.load(open(FLAGS.data_dir + '/yelp14/yelp14_mf.pkl', 'rb'))
    print('finished')


    # with matrix factorization

    # print(mf.P)
    # print(mf.Q)
    # print(mf.full_matrix())
    # pickle.dump({'usrdict': usrdict, 'prddict': prddict, 'mf_p': mf.P, 'mf_Q': mf.Q}, open(FLAGS.data_dir + '/yelp-2013-usr-pdr-mf.pkl', 'wb'))
    # data = pickle.load(open(FLAGS.data_dir + '/yelp-2013-usr-pdr-mf.pkl', 'rb'))
    # print('finished')
