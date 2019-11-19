import logging 
import random
import sys
import os
import datetime

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
from absl import app
from absl import flags

from util.load_util import load_word_embeddings
from util.load_util import load_dataset
from util.load_util import load_parameter_file
from util.data_util import get_data_info, convert_to_id
from util.data_viz import look_at_data
from util.file_util import tee_output
from inference import run_inference

from model import Model

from parameter_search import perform_parameter_search

from testing.test import run_all_tests




FLAGS = flags.FLAGS


# TODO default flag for metadata filepath

# dataset flags
flags.DEFINE_boolean('load_dataset', 
        False, 'load the dataset from pickle?')
flags.DEFINE_string('dataset_file', 
        './data/persona_data.txt',
        'file containing the dataset')
flags.DEFINE_string('pickle_filepath', 
        './data/dataset.pickle',
        'filepath to the saved dataset pickle')
flags.DEFINE_integer('embedding_dim', '300', 
        'number of dimensions of word embeddings')
flags.DEFINE_string('embedding_fname', 
        'data/glove.6B.300d.txt',
        'filepath of word embeddings')

# model flags
flags.DEFINE_list('encoder_sizes', '400, 400, 400',
        'size of each layer in the encoder')
flags.DEFINE_boolean('input_independant', False,
        'whether to train without input')
flags.DEFINE_bool('use_persona_encoder', True,
        'whether to process persona information and feed to the decoder or not')
flags.DEFINE_list('persona_encoder_sizes', '300, 300, 300',
        'size of each layer in the persona encoder')
flags.DEFINE_list('decoder_sizes', '800, 400, 400',
        'size of each layer in the decoder')
flags.DEFINE_float('max_gradient_norm',
        3.0, 'max gradient norm to clip to during training')
flags.DEFINE_float('learning_rate',
        3*10**-4, 'learning rate during training')
flags.DEFINE_integer('batch_size',
        32, 'batch size')

# parameter search flags
# TODO add flags for where to save files and parameter ranges
flags.DEFINE_integer('parameter_search_epochs', 5,
        'number of epochs to test each parameter for')

# training flags
flags.DEFINE_boolean('save_summary',
        True, 'controls whether summaries are saved during training.')
flags.DEFINE_integer('save_frequency',
        400, 'frequency of summary saves')
flags.DEFINE_boolean('run_eval',
        False, 'controls whether eval is run or not')
flags.DEFINE_integer('eval_frequency',
        1000, 'frequency of eval runs')
flags.DEFINE_boolean('print_training',
        True, 'controls whether training progress is printed')
flags.DEFINE_integer('model_save_interval',
        2000, 'number of epochs between model saves')
flags.DEFINE_boolean('save_model',
        True, 'whether to save the model or not')
flags.DEFINE_string('checkpoint_dir',
        'default', 'where to save and load the model. If default then set at runtime to logdir/model_save')
flags.DEFINE_boolean('load_model',
        True, 
        'whether to load the model from file or not for training.')
flags.DEFINE_string('logdir',
        './train/progressive_overfit/1024', 'where to save tensorboard summaries')
flags.DEFINE_integer('dataset_size', 1024, 
        'number of samples to put in the dataset. -1 indicates 90/10 train test split')
flags.DEFINE_bool('use_epochs', True,
        'whether to measure epochs when deciding to stop training rather than number of steps')
flags.DEFINE_integer('epochs', -1,
        'number of epochs to train for. if -1 then train until interrupted')
flags.DEFINE_integer('train_steps', -1, 
        'number of training steps to train for. if -1 then train until interrupted')

# control flags
flags.DEFINE_boolean('use_parameter_override_file',
        True, 'whether to override command line parameters with parameter file'
        + ' or not')
flags.DEFINE_string('parameter_override_filepath',
        'default', 'location of the parameter override file'
            + 'if default then will be changed to logdir/parameters.txt')
flags.DEFINE_boolean('allow_growth', False,
        'whether tensorflow should only allocate memory as needed')
flags.DEFINE_boolean('debug', 
        False, 'run in debug mode?')
flags.DEFINE_string('mode',
        'train', ('what mode to run in. Available modes are train, inference, '
            + 'data_viz, parameter_search, memtest, and unit_test'))

# runtime "flags"
# computed at runtime
flags.DEFINE_integer('max_sentence_len', 0, 
        'the maximum length of any sentence in the dataset. calculated \
        at runtime')
flags.DEFINE_integer('max_conversation_len', 0, 
        'the maximum length of any conversation in the dataset. calculated \
        at runtime')
flags.DEFINE_integer('max_conversation_words', 0, 
        'the maximum number of words in any conversation in the dataset. calculated \
        at runtime')
flags.DEFINE_integer('max_persona_len', 0, 
        'the maximum length of any persona in the dataset. calculated \
        at runtime')
flags.DEFINE_integer('vocab_size', 0,
        'the number of words in our vocabulary. calculated \
            at runtime')

# logging
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
#logging.basicConfig(stream=sys.stderr, level=logging.CRITICAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main(_):
    if FLAGS.allow_growth is True:
        raise Exception("allow growth is depricated")

    # override checkpoint
    if FLAGS.checkpoint_dir == 'default':
        FLAGS.checkpoint_dir = FLAGS.logdir + "/model_save/"
    # override parameter file
    if FLAGS.parameter_override_filepath == 'default':
        FLAGS.parameter_override_filepath = FLAGS.logdir + "/parameters.txt"

    # copy stdout and stderr to checkpoint dir
    tee_output(FLAGS.logdir, "out")

    # load parameter file
    if FLAGS.use_parameter_override_file is True:
        load_parameter_file(FLAGS.parameter_override_filepath, FLAGS)

    # print date and time for output records
    now = datetime.datetime.now()
    print("Current date and time: {}".format(now.strftime("%Y-%m-%d %H:%M:%S")))

    # load training data
    print('loading training data')
    dataset = load_dataset(FLAGS.dataset_file, FLAGS.pickle_filepath,
            FLAGS.load_dataset)
    logging.debug('dataset size: %i' % len(dataset))

    # load metadata
    print('loading metadata')
    word2id, id2word, max_sentence_len, max_conversation_len, \
            max_conversation_words, max_persona_len = get_data_info(dataset)

    # load runtime "flags"
    FLAGS.max_sentence_len = max_sentence_len
    FLAGS.max_conversation_len = max_conversation_len
    FLAGS.max_conversation_words = max_conversation_words
    FLAGS.max_persona_len = max_persona_len
    FLAGS.vocab_size = len(word2id)

    # output parameters to stdout
    print("\n\n")
    print("parameters:")
    for key, value in FLAGS.flag_values_dict().items():
        print("    ", end="")
        print("{} : {}".format(key, value))
    print("\n\n")

    # load word vectors
    print('loading word vectors')
    word2vec = load_word_embeddings(FLAGS.embedding_fname,
            FLAGS.embedding_dim, word2id)

    logging.debug('word2vec type: %s' % type(word2vec))
    logging.debug('word2vec shape: %s' % str(word2vec.shape))

    # convert dataset to integer ids
    print('converting dataset to ids')
    if FLAGS.mode != "data_viz":
        dataset = convert_to_id(dataset, word2id)

    # split into train and test
    # TODO make split ratio into a parameter
    print('splitting dataset')
    #random.shuffle(dataset)

    if FLAGS.dataset_size < 0:
        train_size = int(len(dataset) * 0.9)
    else:
        train_size = FLAGS.dataset_size
    
    train_data = dataset[:train_size]
    test_data = dataset[train_size:]

    # setup debugger
    if FLAGS.debug == True:
        sess = tf_debug.TensorBoardDebugWrapperSession(
                sess, 'localhost:6064')


    logging.debug('building model')

    model = Model(FLAGS, word2vec, id2word, word2id)

    # load model
    if FLAGS.load_model == True or FLAGS.mode == 'inference':
        # ensure load folder exists 
        if os.path.isdir(FLAGS.checkpoint_dir):
            print("loading model from: {}".format(FLAGS.checkpoint_dir))
            model.load()
        else:
            print("no save folder exists. Continuing without loading model.")

    # ensure print outs in main get printed out before further logging debugs
    # TODO change all logging type printouts to logging.debug calls
    print("", flush=True) 

    # perform parameter search
    if FLAGS.mode == "parameter_search":
        logging.debug("performing parameter search")
        parameter_ranges = {}
        parameter_ranges["learning_rate"] = (-5, -2)
        parameter_ranges["hidden_size"] = (250, 950)
        parameter_ranges["num_layers"] = (2, 4)

        perform_parameter_search(Model, FLAGS,
                word2vec, id2word, word2id, parameter_ranges,
                train_data, 
                num_epochs_per_parameter=FLAGS.parameter_search_epochs)
    # run inference
    elif FLAGS.mode == "inference":
        FLAGS.batch_size = 1
        run_inference(dataset, FLAGS, word2id, word2vec, id2word, 1)
    # run data visualization
    elif FLAGS.mode == "data_viz":
        look_at_data(train_data)
    # train model
    elif FLAGS.mode == "train":
        logging.debug('training model')
        model.train(train_data, test_data, FLAGS.train_steps,
                num_epochs=FLAGS.epochs)
    elif FLAGS.mode == "unit_test":
        logging.debug('running unit tests')
        run_all_tests(FLAGS, train_data, word2vec, id2word, word2id)
    elif FLAGS.mode == "memtest":
        # TODO investigate potential memory leak
        # memory usage seems to increase inverse exponentially
        logging.debug('running memtest')
        FLAGS.use_epochs = True
        model.train(train_data, test_data, memtest=True, num_epochs=2000)
    else:
        print("invalid mode! Exiting.")


if __name__ == '__main__':
    print(tf.__version__)
    app.run(main)
