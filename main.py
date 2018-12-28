
import logging
import sys
import os

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np

from util.load_util import load_word_embeddings, load_dataset
from util.data_util import get_data_info, convert_to_id

from models.seq2seq import Seq2SeqBot

#tf.enable_eager_execution()


FLAGS = tf.app.flags.FLAGS


# TODO default flag for metadata filepath

# dataset flags
tf.app.flags.DEFINE_boolean('load_dataset', False, 'load the dataset from pickle?')
tf.app.flags.DEFINE_string('dataset_file', './data/persona_data.txt',
        'file containing the dataset')
tf.app.flags.DEFINE_string('pickle_filepath', './data/dataset.pickle',
        'filepath to the saved dataset pickle')
tf.app.flags.DEFINE_integer('embedding_dim', '300', 
        'number of dimensions of word embeddings')
tf.app.flags.DEFINE_string('embedding_fname', 'data/glove.6B.300d.txt',
        'filepath of word embeddings')

# model flags
tf.app.flags.DEFINE_integer('hidden_size', 600, 'size of the hidden layers')

tf.app.flags.DEFINE_boolean('debug', False, 'run in debug mode?')

# runtime "flags"
tf.app.flags.DEFINE_integer('max_sentence_len', 0, 
        'the maximum length of any sentence in the dataset. calculated \
        at runtime')

# logging
#logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logging.basicConfig(stream=sys.stderr, level=logging.CRITICAL)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main(_):
    # load training data
    print('loading training data')
    dataset = load_dataset(FLAGS.dataset_file, FLAGS.pickle_filepath,
            FLAGS.load_dataset)
    logging.debug('dataset size: %i' % len(dataset))

    # load metadata
    print('loading metadata')
    word2id, id2word, max_sentence_len = get_data_info(dataset)
    FLAGS.max_sentence_len = max_sentence_len
    logging.debug('max sentence len: %i' % max_sentence_len)
    logging.debug('word2id size: %i' % len(word2id))
    logging.debug('id2word shape: %s' % str(id2word.shape))

    # load word vectors
    print('loading word vectors')
    word2vec = load_word_embeddings(FLAGS.embedding_fname,
            FLAGS.embedding_dim, word2id)
    logging.debug('word2vec type: %s' % type(word2vec))
    logging.debug('word2vec shape: %s' % str(word2vec.shape))

    # convert dataset to integer ids
    print('converting dataset to ids')
    dataset = convert_to_id(dataset, word2id)

    # split into train and test
    print('splitting dataset')
    train_size = int(len(dataset) * 0.9)

    # TODO revert after testing
    train_data = dataset[:train_size]
    test_data = dataset[train_size:] # test is remainder after training

    # run training
    print('training')

    sess = tf.Session()
    if FLAGS.debug == True:
        sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:6064')

    model = Seq2SeqBot(FLAGS, sess, word2vec, id2word)

    # perform parameter search
    """
    parameter_ranges = {}
    parameter_ranges["learning_rate"] = (-12, -2)
    parameter_ranges["hidden_size"] = (10, 1000)

    model.perform_parameter_search(parameter_ranges, train_data)
    """
    model.train(train_data, test_data)

    sess.close()

if __name__ == '__main__':
    tf.app.run()

