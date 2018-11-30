
import logging

import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core

from data_util import get_sample


class Seq2SeqBot(object):
    def __init__(self, config, sess, word2vec):
        self.n_hidden = config.hidden_size
        self.max_sentence_len = config.max_sentence_len

        self.sess = sess
        self.vocab_size = len(word2vec)
        self.vocab_dim = word2vec.shape[1]
        self.word2vec = word2vec

        # hyperparams
        self.max_gradient_norm = 0.1 # TODO set this to something reasonable
        self.learning_rate = 0.1 # TODO set this to something reasonable
        self.train_steps = 10000
        self.batch_size = 32

    def build_model(self):
        # placeholders for input
        sentences = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_sentence_len))
        responses = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_sentence_len))
        sentence_lens = tf.placeholder(tf.int64, shape=(self.batch_size))
        response_lens = tf.placeholder(tf.int64, shape=(self.batch_size))

        # convert everything to the proper shapes and types
        #response = tf.reshape(response, (self.batch_size, self.max_sentence_len))
        #sentence_lens = tf.reshape(sentence_lens, (self.batch_size))
        sentence_lens = tf.cast(sentence_lens, tf.int32)
        response_lens = tf.cast(response_lens, tf.int32)

        # start to build the thing
        logging.debug("word2vec shape: " + str(self.word2vec.shape))
        logging.debug("sentences shape: " + str(sentences.shape))
        logging.debug("response: " + str(responses))
        logging.debug("sentence_lens: " + str(sentence_lens))
        #logging.debug("response_lens: " + str(response_lens))

        encoder_embedding_input = tf.nn.embedding_lookup(
                self.word2vec, sentences)
        encoder_embedding_input = tf.cast(encoder_embedding_input, 
                tf.float32)
        encoder_embedding_input = tf.reshape(encoder_embedding_input, [self.max_sentence_len, self.batch_size, self.word2vec.shape[1]])
        logging.debug("encoder embedding input: " + str(encoder_embedding_input))

        with tf.name_scope('encoder'):
            # encoder cell
            encoder_cell = tf.contrib.rnn.LSTMCell(
                    self.n_hidden,
                    initializer = tf.orthogonal_initializer()
                    )
            logging.debug("encoder cell: " + str(encoder_cell))

            # encoder
            encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
                    encoder_cell,
                    encoder_embedding_input,
                    sequence_length = sentence_lens,
                    time_major=True,
                    dtype = tf.float32,
                    scope = "encoder"
                    )
            logging.debug("encoder outputs: " + str(encoder_outputs))
            logging.debug("encoder final state: " + str(encoder_final_state))

        # decoder
        with tf.name_scope('decoder'):
            # decoder cell
            decoder_cell = tf.contrib.rnn.LSTMCell(
                    self.n_hidden,
                    initializer = tf.orthogonal_initializer())
            logging.debug("decoder cell: " + str(decoder_cell))

            # helper
            helper = tf.contrib.seq2seq.TrainingHelper(
                    encoder_outputs,
                    response_lens, time_major=True)
            logging.debug("helper: " + str(helper))

            # decoder
            projection_layer = layers_core.Dense(self.vocab_size,
                    use_bias=False)
            logging.debug("projection layer: " + str(projection_layer))
            decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell,
                    helper,
                    encoder_final_state,
                    output_layer=projection_layer)
            logging.debug("decoder: " + str(decoder))

            # dynamic decoding
            outputs, final_state, final_sequence_lengths = \
                    tf.contrib.seq2seq.dynamic_decode(decoder)
            logits = outputs.rnn_output
            logging.debug("outputs: " + str(outputs))
            logging.debug("final state: " + str(final_state))
            logging.debug("final sequence lengths: " + str(final_sequence_lengths))
            logging.debug("logits: " + str(logits))

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=responses, logits=logits)
        logging.debug("loss: " + str(loss))

        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
                gradients, self.max_gradient_norm)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = train_op = optimizer.apply_gradients(
                zip(clipped_gradients, params), global_step=global_step)

    def run_eager(self, training_data, test_data):
        # setup inputs
        sentences = []
        responses = []
        sentence_lens = []
        response_lens = []

        # build batch data
        batch_size = self.batch_size
        for i in range(batch_size):
            sentence, response, sentence_len, response_len = \
                get_sample(training_data, self.max_sentence_len)

            sentences.append(sentence)
            responses.append(response)
            sentence_lens.append(sentence_len)
            response_lens.append(response_len)

        sentences = np.array(sentences)
        responses = np.array(responses)
        sentence_lens = np.array(sentence_lens)
        response_lens = np.array(response_lens)

        logging.debug("sentences shape: " + str(sentences.shape))
        logging.debug("responses shape: " + str(responses.shape))
        logging.debug("sentence_lens shape: " + str(sentence_lens.shape))
        logging.debug("response_lens shape: " + str(response_lens.shape))

        logging.debug("sentences type: " + str(sentences.dtype))
        logging.debug("responses type: " + str(responses.dtype))
        logging.debug("sentence_lens type: " + str(sentence_lens.dtype))
        logging.debug("response_lens type: " + str(response_lens.dtype))
        
        # start to build the thing

        response = tf.convert_to_tensor(responses)
        response = tf.reshape(response, [batch_size, self.max_sentence_len])

        sentence_lens = tf.convert_to_tensor(sentence_lens)
        sentence_lens = tf.reshape(sentence_lens, [batch_size])
        sentence_lens = tf.cast(sentence_lens, tf.int32)

        response_lens = tf.convert_to_tensor(response_lens)
        response_lens = tf.cast(response_lens, tf.int32)

        logging.debug("word2vec shape: " + str(self.word2vec.shape))
        logging.debug("sentences shape: " + str(sentences.shape))
        logging.debug("response: " + str(responses))
        logging.debug("sentence_lens: " + str(sentence_lens))
        #logging.debug("response_lens: " + str(response_lens))

        encoder_embedding_input = tf.nn.embedding_lookup(
                self.word2vec, sentences)
        encoder_embedding_input = tf.cast(encoder_embedding_input, 
                tf.float32)
        encoder_embedding_input = tf.reshape(encoder_embedding_input, [self.max_sentence_len, batch_size, self.word2vec.shape[1]])
        logging.debug("encoder embedding input: " + str(encoder_embedding_input))


        with tf.name_scope('encoder'):
            # encoder cell
            encoder_cell = tf.contrib.rnn.LSTMCell(
                    self.n_hidden,
                    initializer = tf.orthogonal_initializer()
                    )
            logging.debug("encoder cell: " + str(encoder_cell))

            # encoder
            encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
                    encoder_cell,
                    encoder_embedding_input,
                    sequence_length = sentence_lens,
                    time_major=True,
                    dtype = tf.float32,
                    scope = "encoder"
                    )
            logging.debug("encoder outputs: " + str(encoder_outputs))
            logging.debug("encoder final state: " + str(encoder_final_state))

        # decoder
        with tf.name_scope('decoder'):
            # decoder cell
            decoder_cell = tf.contrib.rnn.LSTMCell(
                    self.n_hidden,
                    initializer = tf.orthogonal_initializer())
            logging.debug("decoder cell: " + str(decoder_cell))

            # helper
            helper = tf.contrib.seq2seq.TrainingHelper(
                    encoder_outputs,
                    response_lens, time_major=True)
            logging.debug("helper: " + str(helper))

            # decoder
            projection_layer = layers_core.Dense(self.vocab_size,
                    use_bias=False)
            logging.debug("projection layer: " + str(projection_layer))
            decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell,
                    helper,
                    encoder_final_state,
                    output_layer=projection_layer)
            logging.debug("decoder: " + str(decoder))

            # dynamic decoding
            outputs, final_state, final_sequence_lengths = \
                    tf.contrib.seq2seq.dynamic_decode(decoder)
            logits = outputs.rnn_output
            logging.debug("outputs: " + str(outputs))
            logging.debug("final state: " + str(final_state))
            logging.debug("final sequence lengths: " + str(final_sequence_lengths))
            logging.debug("logits: " + str(logits))

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=response, logits=logits)
        logging.debug("loss: " + str(loss))

        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
                gradients, self.max_gradient_norm)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = train_op = optimizer.apply_gradients(
                zip(clipped_gradients, params), global_step=global_step)



            



