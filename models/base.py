
import logging

import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core

import sys
import random
import math


class Chatbot(object):
    def __init__(self, config, sess, word2vec, id2word):
        # load parameters
        self._load_params(config, sess, word2vec, id2word)

        # build model
        self.build_model()

        # saver
        self.saver = tf.train.Saver()

        # file writer
        if config.save_summary == True:
            self.writer = tf.summary.FileWriter('./train', sess.graph)
        else:
            self.writer = None

        # summaries
        self.summaries = tf.summary.merge_all()

        # global variables initialize
        self.sess.run(tf.global_variables_initializer())

        # initialize embeddings
        self.sess.run(self.embedding_init, feed_dict={self.embedding_placeholder: self.word2vec})

    def _load_params(self, config, sess, word2vec, id2word):
        self.max_sentence_len = config.max_sentence_len

        self.sess = sess
        self.vocab_size = len(word2vec)
        self.vocab_dim = word2vec.shape[1]
        self.word2vec = word2vec
        self.id2word = id2word

        # hyperparams
        self.n_hidden = config.hidden_size
        self.max_gradient_norm = config.max_gradient_norm
        self.learning_rate = config.learning_rate
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size

        # trianing params
        self.save_frequency = config.save_frequency
        self.save_summary = config.save_summary
        self.print_training = config.print_training
        self.print_dot_interval = config.print_dot_interval
        self.dots_per_line = config.dots_per_line
        self.model_save_interval = config.model_save_interval
        self.save_model = config.save_model
        self.model_save_filepath = config.model_save_filepath

    def load_model(self):
        self.saver.restore(self.sess, self.model_save_filepath)

    def build_model(self):
        raise Exception("virtual method! Implement in subclass.")

    def train(self):
        raise Exception("virtual method! Implement in subclass.")


