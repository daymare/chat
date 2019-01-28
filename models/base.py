
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


