
import logging

import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core

import sys
import random
import math


class Chatbot(object):
    def __init__(self, config, sess, word2vec, id2word):
        self.max_sentence_len = config.max_sentence_len

        self.sess = sess
        self.vocab_size = len(word2vec)
        self.vocab_dim = word2vec.shape[1]
        self.word2vec = word2vec
        self.id2word = id2word

        # hyperparams
        self.n_hidden = config.hidden_size
        self.max_gradient_norm = 3.0
        self.learning_rate = 4.7 * 10**-3
        self.train_steps = 1000000
        self.batch_size = 1

        # build model
        self.build_model()

        # file writer
        self.writer = tf.summary.FileWriter('./train', sess.graph)

        # summaries
        self.summaries = tf.summary.merge_all()

        # global variables initialize
        self.sess.run(tf.global_variables_initializer())

        # initialize embeddings
        self.sess.run(self.embedding_init, feed_dict={self.embedding_placeholder: self.word2vec})

    def build_model(self):
        raise Exception("virtual method! Implement in subclass.")

    def train(self):
        raise Exception("virtual method! Implement in subclass.")

    def perform_parameter_search(self, parameter_ranges, training_data, 
            num_epochs_per_parameter=2500, result_filepath="parameter_search_results.txt"):
        # TODO modify this so it works properly with arbitrary models
        # WARNING: not modified to work with models other than seq2seq
        """ perform random parameter search

        runs random parameter searches in the valid ranges until termination
        (receive SIG-int, CTRL-C)

        current searchable parameters:
            learning rate
            hidden size

        input:
            parameter_ranges:
                dictionary of parameter_name -> (low_value, high_value)
            training_data - training data, see load util.load_dataset
            num_epochs_per_parameter - number of epochs to run each parameter
                configuration for before returning the output
        output: 
            returns: None
            prints: parameter configurations and their scores
            saves parameter configurations and their scores to file
        """
        # open result file
        result_file = open(result_filepath, "a")
        
        # get parameter ranges
        learning_rate_range = None if "learning_rate" not in parameter_ranges \
                else parameter_ranges["learning_rate"]
        hidden_size_range = None if "hidden_size" not in parameter_ranges \
                else parameter_ranges["hidden_size"]

        def get_loguniform(low_exponent, high_exponent):
            value = random.random() * 10
            exponent = random.randint(low_exponent, high_exponent)

            return value * 10**exponent


        def generate_parameter_config():
            config = {}
            config["learning_rate"] = get_loguniform(learning_rate_range[0], 
                    learning_rate_range[1])
            config["hidden_size"] = random.randint(hidden_size_range[0],
                    hidden_size_range[1])

            return config

        def apply_parameter_config(config):
            self.learning_rate = config["learning_rate"]
            self.n_hidden = config["hidden_size"]

            tf.reset_default_graph()
            self.sess.close()
            self.sess = tf.Session()

            self.build_model()

        best_loss_config = None
        best_loss = None

        # generate a parameter config and test
        while True:
            config = generate_parameter_config()
            apply_parameter_config(config)

            # train
            try:
                loss, perplexity = self.train(training_data, None, 
                        num_epochs_per_parameter, save_summary=False,
                        print_training=True)
            except:
                loss, perplexity = math.inf, math.inf

            # output results
            print()
            results_text = str(config) + " loss: " + str(loss) + "\n"
            result_file.write(results_text)
            print(results_text, end='')

            # calculate best loss and output
            if best_loss == None or best_loss > loss:
                best_loss_config = config
                best_loss = loss

            # output best loss
            print("best loss: " + str(best_loss_config) + " loss: " + str(best_loss))
            result_file.write("best loss: " + str(best_loss_config) + " loss: " + str(best_loss) + "\n\n")
            result_file.flush()


