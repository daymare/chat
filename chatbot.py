
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
        """ build the computation graph
            
            graph:
                inputs: 
                    two previous sentences
                outputs:
                    response sentence
        """
        # TODO add dropout
        # TODO add batch norm

        with tf.name_scope('inputs'):
            self.sentence = tf.placeholder(
                    tf.int32, [None, self.max_sentence_len],
                    name="sentence")
            self.sentence_lens = tf.placeholder(tf.int32, [None],
                    name="sentence_lens")
            self.response_sentence = tf.placeholder(
                    tf.int32, [None, self.max_sentence_len],
                    name="response_sentence")
            self.response_lens = tf.placeholder(tf.int32, shape=[None],
                    name="response_lens")

            # embedding input
            encoder_embedding_input = tf.nn.embedding_lookup(
                    self.word2vec, self.sentence)
            encoder_embedding_input = tf.cast(encoder_embedding_input, 
                    tf.float32)
            encoder_embedding_input = tf.reshape(
                    encoder_embedding_input, 
                    [self.max_sentence_len, self.batch_size, 
                        self.word2vec.shape[1]])

        # encoder (1 layer single directional lstm rnn)
        with tf.name_scope('encoder'):
            # encoder cell
            encoder_cell = tf.contrib.rnn.LSTMCell(
                    self.n_hidden,
                    initializer = tf.orthogonal_initializer()
                    )

            # encoder overall
            print("input shape: ", encoder_embedding_input.shape)
            encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
                    encoder_cell,
                    encoder_embedding_input,
                    sequence_length = self.sentence_lens,
                    time_major=True,
                    dtype = tf.float32,
                    scope = 'encoder'
                    )


        # decoder
        with tf.name_scope('decoder'):
            # decoder cell
            decoder_cell = tf.contrib.rnn.LSTMCell(
                    self.n_hidden,
                    initializer = tf.orthogonal_initializer()
                    )

            # helper
            helper = tf.contrib.seq2seq.TrainingHelper(
                    encoder_outputs,
                    self.response_lens, time_major=True)

            # decoder
            projection_layer = layers_core.Dense(self.vocab_size,
                    use_bias=False)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell,
                    helper,
                    encoder_final_state,
                    output_layer=projection_layer
                    )

            # dynamic decoding
            outputs, final_state, final_sequence_lengths = \
                tf.contrib.seq2seq.dynamic_decode(decoder)
            logits = outputs.rnn_output

        # loss
        self.loss = loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.response_sentence, logits=logits)

        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
                gradients, self.max_gradient_norm)

        # optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = train_op = optimizer.apply_gradients(
                zip(clipped_gradients, params), global_step=global_step)

    def train(self, training_data, test_data):
        """
        """
        for i in range(self.train_steps):
            # create next batch
            input_sentences = []
            sentence_lens = []
            responses = []
            response_lens = []

            for j in range(self.batch_size):
                next_sentence, next_response = get_sample(training_data,
                        self.max_sentence_len)
                sentence_lens.append(len(next_sentence))
                response_lens.append(len(responses))
                input_sentences.append(next_sentence)
                responses.append(next_response)

            # set up feed dict
            feed_dict = {
                    self.sentence: input_sentences,
                    self.sentence_lens: sentence_lens,
                    self.response_sentence: responses,
                    self.response_lens: response_lens
            }

            # run iteration
            _, _ = self.sess.run([self.train_op, self.loss], 
                                 feed_dict=feed_dict)

    def run_eager(self, training_data, test_data):
        # setup inputs
        sentence, response = get_sample(training_data, 
                self.max_sentence_len)
        sentence_len = [len(sentence)]
        response_len = [len(response)]
        batch_size = 1
        
        # start to build the thing
        logging.debug("word2vec shape: " + str(self.word2vec.shape))
        logging.debug("sentences shape: " + str(sentence.shape))
        response = tf.convert_to_tensor(response)
        response = tf.reshape(response, [batch_size, self.max_sentence_len])
        logging.debug("response: " + str(response))

        #logging.debug("sentences: " + str(sentence))
        encoder_embedding_input = tf.nn.embedding_lookup(
                self.word2vec, sentence)
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
                    sequence_length = sentence_len,
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
                    response_len, time_major=True)
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
                    labels = response, logits=logits)
            logging.debug("loss: " + str(loss))

            params = tf.trainable_variables()
            gradients = tf.gradients(loss, params)
            



