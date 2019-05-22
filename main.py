
import logging
import random
import sys
import os

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np

from util.load_util import load_word_embeddings, load_dataset
from util.data_util import get_data_info, convert_to_id
from inference import run_inference

from model import Model

from tools.parameter_search import perform_parameter_search


config = tf.app.flags.FLAGS


# TODO default flag for metadata filepath

# dataset flags
tf.app.flags.DEFINE_boolean('load_dataset', 
        False, 'load the dataset from pickle?')
tf.app.flags.DEFINE_string('dataset_file', 
        './data/persona_data.txt',
        'file containing the dataset')
tf.app.flags.DEFINE_string('pickle_filepath', 
        './data/dataset.pickle',
        'filepath to the saved dataset pickle')
tf.app.flags.DEFINE_integer('embedding_dim', '300', 
        'number of dimensions of word embeddings')
tf.app.flags.DEFINE_string('embedding_fname', 
        'data/glove.6B.300d.txt',
        'filepath of word embeddings')

# model flags
tf.app.flags.DEFINE_list('encoder_sizes', '500, 300',
        'size of each layer in the encoder')
tf.app.flags.DEFINE_list('persona_encoder_sizes', '500, 300',
        'size of each layer in the persona encoder')
tf.app.flags.DEFINE_integer('decoder_units', 
        500, 'size of the hidden layer in the decoder')
tf.app.flags.DEFINE_float('max_gradient_norm',
        3.0, 'max gradient norm to clip to during training')
tf.app.flags.DEFINE_float('learning_rate',
        0.021583475806248406, 'learning rate during training')
tf.app.flags.DEFINE_integer('train_steps',
        1000000, 'number of training steps to train for')
tf.app.flags.DEFINE_integer('batch_size',
        64, 'batch size')

# training flags
tf.app.flags.DEFINE_boolean('save_summary',
        True, 'controls whether summaries are saved during training.')
tf.app.flags.DEFINE_integer('save_frequency',
        100, 'number of epochs between summary saves')
tf.app.flags.DEFINE_boolean('print_training',
        True, 'controls whether training progress is printed')
tf.app.flags.DEFINE_integer('print_dot_interval',
        20, 'number of epochs between dot prints to screen')
tf.app.flags.DEFINE_integer('dots_per_line',
        45, 'number of dots printed between newlines')
tf.app.flags.DEFINE_integer('model_save_interval',
        1000, 'number of epochs between model saves')
tf.app.flags.DEFINE_boolean('save_model',
        True, 'whether to save the model or not')
tf.app.flags.DEFINE_string('checkpoint_dir',
        './train/model_save/', 'where to save the model')
tf.app.flags.DEFINE_string('logdir',
        './train/', 'where to save tensorboard summaries')
tf.app.flags.DEFINE_boolean('load_model',
        False, 
        'whether to load the model from file or not for training.')

# control flags
tf.app.flags.DEFINE_boolean('debug', 
        False, 'run in debug mode?')
tf.app.flags.DEFINE_boolean('run_inference',
        False, 'run inference instead of training?')
tf.app.flags.DEFINE_boolean('parameter_search',
        False, 'run parameter search instead of training?')


# runtime "flags"
# computed at runtime
tf.app.flags.DEFINE_integer('max_sentence_len', 0, 
        'the maximum length of any sentence in the dataset. calculated \
        at runtime')
tf.app.flags.DEFINE_integer('max_conversation_len', 0, 
        'the maximum length of any conversation in the dataset. calculated \
        at runtime')
tf.app.flags.DEFINE_integer('max_conversation_words', 0, 
        'the maximum number of words in any conversation in the dataset. calculated \
        at runtime')
tf.app.flags.DEFINE_integer('max_persona_len', 0, 
        'the maximum length of any persona in the dataset. calculated \
        at runtime')
tf.app.flags.DEFINE_integer('vocab_size', 0,
        'the number of words in our vocabulary. calculated \
            at runtime')

# logging
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
#logging.basicConfig(stream=sys.stderr, level=logging.CRITICAL)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.enable_eager_execution()


def main(_):
    # load training data
    print('loading training data')
    dataset = load_dataset(config.dataset_file, config.pickle_filepath,
            config.load_dataset)
    logging.debug('dataset size: %i' % len(dataset))

    # load metadata
    print('loading metadata')
    word2id, id2word, max_sentence_len, max_conversation_len, \
            max_conversation_words, max_persona_len = get_data_info(dataset)

    # load runtime "flags"
    config.max_sentence_len = max_sentence_len
    config.max_conversation_len = max_conversation_len
    config.max_conversation_words = max_conversation_words
    config.max_persona_len = max_persona_len
    config.vocab_size = len(word2id)

    logging.debug('max sentence len: %i' % max_sentence_len)
    logging.debug("max conversation words: {}".format(max_conversation_words))
    logging.debug('word2id size: %i' % len(word2id))
    logging.debug('id2word shape: %s' % str(id2word.shape))

    # load word vectors
    print('loading word vectors')
    word2vec = load_word_embeddings(config.embedding_fname,
            config.embedding_dim, word2id)
    logging.debug('word2vec type: %s' % type(word2vec))
    logging.debug('word2vec shape: %s' % str(word2vec.shape))

    # convert dataset to integer ids
    print('converting dataset to ids')
    dataset = convert_to_id(dataset, word2id)

    # split into train and test
    # TODO make split ratio into a parameter
    print('splitting dataset')
    random.shuffle(dataset)
    train_size = int(len(dataset) * 0.9)

    train_data = dataset[:train_size]
    # test is remainder after training
    test_data = dataset[train_size:] 

    # setup debugger
    if config.debug == True:
        sess = tf_debug.TensorBoardDebugWrapperSession(
                sess, 'localhost:6064')


    # TODO add flags and control flow for parameter search
    logging.debug('building model')
    #model = Seq2SeqBot(config, sess, word2vec, id2word)
    #model = ProfileMemoryBot(config, sess, word2vec, id2word)

    model = Model(config, word2vec, id2word, word2id)

    # load model
    # TODO add check to ensure the file exists
    if config.load_model == True or config.run_inference == True:
        print("loading model")
        model.load(config.checkpoint_dir)

    # perform parameter search
    if config.parameter_search == True:
        print("performing parameter search", flush=True)
        parameter_ranges = {}
        parameter_ranges["learning_rate"] = (-12, -2)
        parameter_ranges["hidden_size"] = (100, 1000)

        perform_parameter_search(Model, config,
                word2vec, id2word, word2id, parameter_ranges,
                train_data)
    # run inference
    elif config.run_inference == True:
        config.batch_size = 1
        run_inference(dataset, config, word2id, word2vec, id2word, 1)
    else:
        # train model
        logging.debug('training model')
        model.train(train_data, test_data, config.train_steps)


if __name__ == '__main__':
    tf.app.run()

