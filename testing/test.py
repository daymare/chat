""" file for unit testing
"""
# TODO add "check that inference runs" test
# TODO add "check that eval runs" test

import random
import logging

import tensorflow as tf

from model import Model
from util.train_util import get_sample
from util.train_util import pad_persona
import util.test_util as test_util


def run_all_tests(config, train_data, word2vec, id2word, word2id):
    # set config and random seed
    logging.debug('setup for testing')
    test_util.set_test_config(config)
    tf.random.set_random_seed(284586569)
    random.seed(206738548)

    check_training_runs(config, train_data, word2vec, id2word, word2id)

    all_weights_updated_test(config, train_data, word2vec, id2word, word2id)

    save_load_test(config, train_data, word2vec, id2word, word2id)


## Tests
def check_training_runs(config, train_data, word2vec, id2word, word2id):
    """ run training for a few steps to ensure it does not crash
            more of a development aid than anything really
    """
    logging.debug('running training check')

    tf.reset_default_graph()
    model = Model(config, word2vec, id2word, word2id)

    model.train(train_data, None, 5)

    logging.debug('finished training check')

def all_weights_updated_test(config, train_data, word2vec, id2word, word2id):
    """ run a single train step and ensure that each value in trainable 
            variables was updated
    """
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
        print("before: {}".format(b))
        print("after: {}".format(a))
        assert (b != a), "test found a weight that was not updated!"

    logging.debug('finished all weights updated test')

def save_load_test(config, train_data, word2vec, id2word, word2id):
    logging.debug('running save load test!')

    # init model
    tf.reset_default_graph()
    config.checkpoint_dir = './testing/test_dir'
    model = Model(config, word2vec, id2word, word2id)

    # get model weights before
    vars_before = test_util.get_trainable_variables_numpy()

    # get an example feedforward run
    sample_persona, sample_conversation, _ = get_sample(
            train_data, word2id, 0, 0)
    sample_persona = pad_persona(sample_persona)
    sample_persona = tf.expand_dims(sample_persona, 0)
    sample_conversation = tf.expand_dims(sample_conversation, 0)
    predictions_before, _ = model.call(sample_conversation, 
            personas=sample_persona)

    # save model
    model.save()

    # reload model
    model.load()

    # get feedforward run after load
    vars_after = test_util.get_trainable_variables_numpy()
    predictions_after, _ = model.call(sample_conversation,
            personas=sample_persona)

    # check everything is the same
    for i in range(len(vars_before)):
        b = vars_before[i]
        a = vars_after[i]
        assert (b == a), "test found a weight that was different!"

    # get rid of batch dimension
    predictions_before = predictions_before[0]
    predictions_after = predictions_after[0]

    for i in range(len(predictions_before)):
        pred_before = predictions_before[i]
        pred_after = predictions_after[i]

        difference = pred_after - pred_before
        mag = tf.norm(difference)

        assert mag < 0.001, "test found predictions were different!"

    logging.debug('finished save load test!')

def loss_nonzero_test():
    pass

def loss_at_init_test():
    pass
