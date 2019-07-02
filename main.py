import logging 
import random
import sys
import os
import datetime

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np

from util.load_util import load_word_embeddings, load_dataset
from util.data_util import get_data_info, convert_to_id
from util.data_viz import look_at_data
from util.file_util import tee_output
from inference import run_inference

from model import Model

from parameter_search import perform_parameter_search



# set eager to allow growth
gpu_options = tf.GPUOptions(allow_growth=True)
tf_config = tf.ConfigProto(gpu_options=gpu_options)
tf.enable_eager_execution(config=tf_config)

# set environment variable to force allow growth
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


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
tf.app.flags.DEFINE_list('encoder_sizes', '205',
        'size of each layer in the encoder')
tf.app.flags.DEFINE_boolean('input_independant', False,
        'whether to train without input')
tf.app.flags.DEFINE_bool('use_persona_encoder', True,
        'whether to process persona information and feed to the decoder or not')
tf.app.flags.DEFINE_list('persona_encoder_sizes', '139, 160, 655, 117',
        'size of each layer in the persona encoder')
tf.app.flags.DEFINE_integer('decoder_units', 
        205, 'size of the hidden layer in the decoder')
tf.app.flags.DEFINE_float('max_gradient_norm',
        3.0, 'max gradient norm to clip to during training')
tf.app.flags.DEFINE_float('learning_rate',
        4*10**-3, 'learning rate during training')
tf.app.flags.DEFINE_bool('use_epochs', True,
        'whether to measure epochs when deciding to stop training rather than number of steps')
tf.app.flags.DEFINE_integer('epochs', -1,
        'number of epochs to train for. if -1 then train until interrupted')
tf.app.flags.DEFINE_integer('train_steps', -1, 
        'number of training steps to train for. if -1 then train until interrupted')
tf.app.flags.DEFINE_integer('batch_size',
        64, 'batch size')

# training flags
tf.app.flags.DEFINE_boolean('save_summary',
        True, 'controls whether summaries are saved during training.')
tf.app.flags.DEFINE_integer('save_frequency',
        200, 'frequency of summary saves')
tf.app.flags.DEFINE_integer('eval_frequency',
        1000, 'frequency of eval runs')
tf.app.flags.DEFINE_boolean('print_training',
        True, 'controls whether training progress is printed')
tf.app.flags.DEFINE_integer('model_save_interval',
        1000, 'number of epochs between model saves')
tf.app.flags.DEFINE_boolean('save_model',
        True, 'whether to save the model or not')
tf.app.flags.DEFINE_string('checkpoint_dir',
        'default', 'where to save and load the model. If default then set at runtime to logdir/model_save')
tf.app.flags.DEFINE_string('logdir',
        './train/parameter_search_original', 'where to save tensorboard summaries')
tf.app.flags.DEFINE_boolean('load_model',
        True, 
        'whether to load the model from file or not for training.')

# control flags
tf.app.flags.DEFINE_boolean('debug', 
        False, 'run in debug mode?')
tf.app.flags.DEFINE_boolean('run_inference',
        False, 'run inference instead of training?')
tf.app.flags.DEFINE_boolean('run_data_viz',
        False, 'run dataset visualization instead of anything else')
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

# set random seeds for testing
tf.set_random_seed(1) # for testing, remove after system has been verified
random.seed(1) # for testing, remove after the system has been verified


def main(_):
    # override checkpoint
    if config.checkpoint_dir == 'default':
        config.checkpoint_dir = config.logdir + "/model_save/"

    # copy stdout and stderr to checkpoint dir
    tee_output(config.logdir, "out")

    # print date and time for output records
    now = datetime.datetime.now()
    print("Current date and time: {}".format(now.strftime("%Y-%m-%d %H:%M:%S")))

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

    # output parameters to stdout
    print("\n\n")
    print("parameters:")
    for key, value in config.flag_values_dict().items():
        print("    ", end="")
        print("{} : {}".format(key, value))
    print("\n\n")

    # load word vectors
    print('loading word vectors')
    word2vec = load_word_embeddings(config.embedding_fname,
            config.embedding_dim, word2id)

    logging.debug('word2vec type: %s' % type(word2vec))
    logging.debug('word2vec shape: %s' % str(word2vec.shape))

    # convert dataset to integer ids
    print('converting dataset to ids')
    if config.run_data_viz is False:
        dataset = convert_to_id(dataset, word2id)

    # split into train and test
    # TODO make split ratio into a parameter
    print('splitting dataset')
    #random.shuffle(dataset)
    train_size = int(len(dataset) * 0.9)

    train_data = dataset[:train_size]
    #train_data = dataset[:1]
    #train_data = dataset
    test_data = dataset[train_size:]

    # setup debugger
    if config.debug == True:
        sess = tf_debug.TensorBoardDebugWrapperSession(
                sess, 'localhost:6064')


    logging.debug('building model')

    model = Model(config, word2vec, id2word, word2id)

    # load model
    # TODO add check to ensure the file exists
    if config.load_model == True or config.run_inference == True:
        # ensure load folder exists 
        if os.path.isdir(config.checkpoint_dir):
            print("loading model from: {}".format(config.checkpoint_dir))
            model.load(config.checkpoint_dir)
        else:
            print("no save folder exists. Continuing without loading model.")

    # perform parameter search
    if config.parameter_search == True:
        print("performing parameter search", flush=True)
        parameter_ranges = {}
        parameter_ranges["learning_rate"] = (-12, -2)
        parameter_ranges["hidden_size"] = (100, 700)
        parameter_ranges["num_layers"] = (1, 5)

        perform_parameter_search(Model, config,
                word2vec, id2word, word2id, parameter_ranges,
                train_data)
    # run inference
    elif config.run_inference == True:
        config.batch_size = 1
        run_inference(dataset, config, word2id, word2vec, id2word, 1)
    # run data visualization
    elif config.run_data_viz == True:
        look_at_data(train_data)
    # train model
    else:
        logging.debug('training model')
        model.train(train_data, test_data, config.train_steps)


if __name__ == '__main__':
    tf.app.run()

