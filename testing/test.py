""" file for unit testing
"""

import random
import logging

import tensorflow as tf

from model import Model
import util.test_util as test_util


def run_all_tests(config, train_data, word2vec, id2word, word2id):
    # set config and random seed
    logging.debug('setup for testing')
    test_util.set_test_config(config)
    tf.random.set_random_seed(284586569)
    random.seed(206738548)

    all_weights_updated_test(config, train_data, word2vec, id2word, word2id)


## Tests
def all_weights_updated_test(config, train_data, word2vec, id2word, word2id):
    logging.debug('running all weights updated test')

    # init model
    tf.reset_default_graph()
    model = Model(config, word2vec, id2word, word2id)

    # get model weights before
    before = test_util.get_trainable_variables_numpy()

    # run a single training step
    model.train(train_data, None, 1)

    # ensure weights are changed
    after = test_util.get_trainable_variables_numpy()

    for i in range(len(before)):
        b = before[i]
        a = after[i]
        assert (b != a), "test found a weight that was not updated!"

    logging.debug('finished all weights updated test')

def loss_nonzero_test():
    pass

def loss_at_init_test():
    pass
