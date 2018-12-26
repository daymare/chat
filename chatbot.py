
import logging

import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core

import sys

from data_util import get_training_batch


class Seq2SeqBot(object):
    def __init__(self, config, sess, word2vec, id2word):
        self.n_hidden = config.hidden_size
        self.max_sentence_len = config.max_sentence_len

        self.sess = sess
        self.vocab_size = len(word2vec)
        self.vocab_dim = word2vec.shape[1]
        self.word2vec = word2vec
        self.id2word = id2word

        # hyperparams
        self.max_gradient_norm = 3.0
        self.learning_rate = 1.0 * 10**-4
        self.train_steps = 1000000
        self.batch_size = 32

        # build model
        self.build_model()

        # file writer
        self.writer = tf.summary.FileWriter('./train', sess.graph)


    def build_model(self):
        # placeholders for input
        sentences = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_sentence_len))
        responses = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_sentence_len))
        sentence_lens = tf.placeholder(tf.int64, shape=(self.batch_size))
        response_lens = tf.placeholder(tf.int64, shape=(self.batch_size))

        # convert everything to the proper shapes and types
        sentence_lens = tf.cast(sentence_lens, tf.int32)
        response_lens = tf.cast(response_lens, tf.int32)

        # put placeholders into self for feed dict
        self.sentences = sentences
        self.responses = responses
        self.sentence_lens = sentence_lens
        self.response_lens = response_lens

        # start to build the thing
        logging.debug("word2vec shape: " + str(self.word2vec.shape))
        logging.debug("sentences shape: " + str(sentences.shape))
        logging.debug("responses: " + str(responses))
        logging.debug("sentence_lens: " + str(sentence_lens))
        logging.debug("response_lens: " + str(response_lens))

        # embeddings
        embeddings = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.vocab_dim]),
                trainable=False, name="embeddings")
        embedding_placeholder = tf.placeholder(
                tf.float32,
                shape=[self.vocab_size, self.vocab_dim]
                )

        embedding_init = embeddings.assign(embedding_placeholder)

        # encoder embedding input
        encoder_embedding_input = tf.nn.embedding_lookup(
                embeddings, sentences)
        encoder_embedding_input = tf.cast(encoder_embedding_input, 
                tf.float32)

        # decoder embedding input
        responses_input = tf.pad(responses, [[0,0], [1,0]], "CONSTANT", constant_values=0)
        decoder_embedding_input = tf.nn.embedding_lookup(
                embeddings, responses_input)
        decoder_embedding_input = tf.cast(decoder_embedding_input, tf.float32)

        logging.debug("encoder embedding input: " + str(encoder_embedding_input))

        def lstm_cell():
            return tf.contrib.rnn.LSTMCell(self.n_hidden)

        with tf.name_scope('encoder'):
            # encoder cell
            encoder_cell = lstm_cell()

            logging.debug("encoder cell: " + str(encoder_cell))

            # encoder
            encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
                    encoder_cell,
                    encoder_embedding_input,
                    sequence_length = sentence_lens,
                    time_major=False,
                    dtype = tf.float32,
                    scope = "encoder"
                    )

            # set up summary histograms
            weights, biases = encoder_cell.variables
            tf.summary.histogram("encoder_cell_weights", weights)
            tf.summary.histogram("encoder_cell_biases", biases)

            logging.debug("encoder outputs: " + str(encoder_outputs))
            logging.debug("encoder final state: " + str(encoder_final_state))

        # decoder
        with tf.name_scope('decoder'):
            # decoder cell
            decoder_cell = lstm_cell()
            logging.debug("decoder cell: " + str(decoder_cell))

            decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
                    decoder_cell, decoder_embedding_input,
                    initial_state=encoder_final_state, 
                    dtype=tf.float32, time_major=False)

            logits = tf.layers.dense(
                    inputs=decoder_outputs, 
                    units=self.vocab_size, 
                    name="projection_layer")

            with tf.variable_scope("projection_layer", reuse=True):
                weights = tf.get_variable("kernel")
                bias = tf.get_variable("bias")
                tf.summary.histogram("projection_layer_weights", weights)
                tf.summary.histogram("projection_layer_bias", bias)

            # set up summary histograms
            weights, biases = decoder_cell.variables
            tf.summary.histogram("decoder_cell_weights", weights)
            tf.summary.histogram("decoder_cell_biases", biases)

            # 0 pad logits to match shape of labels
            logging.debug("final state: " + str(decoder_final_state))
            logging.debug("logits: " + str(logits))
            logging.debug("logits[:-1]: " + str(logits.shape[:-1]))


        # loss
        # repad responses so they are the same dims as logits
        responses = tf.pad(responses, [[0,0],[0,1]], "CONSTANT", constant_values=0)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(responses, depth=self.vocab_size, dtype=tf.float32),
                logits=logits)

        perplexity = tf.exp(cross_entropy)
        loss = tf.reduce_mean(cross_entropy)

        logging.debug("loss: " + str(loss))

        # summarize the loss
        averaged_loss = tf.reduce_mean(loss)
        tf.summary.histogram('loss', loss)
        tf.summary.scalar('averaged_loss', averaged_loss)

        # summarize the perplexity
        average_perplexity = tf.reduce_mean(perplexity)
        tf.summary.scalar('avg_perplexity', average_perplexity)

        # summarize the text input and output
        output_example = logits[0] # shape (max_sentence_len, dictionary_size)

        # convert logits to ids
        example_predictions = tf.argmax(output_example, 1) # shape (max_sentence_len) 

        # convert ids to sentence
        example_input_list = tf.nn.embedding_lookup(self.id2word, sentences[0])
        example_response_list = tf.nn.embedding_lookup(self.id2word, responses[0])
        example_text_list = tf.nn.embedding_lookup(self.id2word, example_predictions)

        logging.debug("example_input_list: " + str(example_input_list))
        logging.debug("example_response_list: " + str(example_response_list))
        logging.debug("example_text_list: " + str(example_text_list))

        example_sentence = tf.strings.reduce_join(example_input_list, separator=' ')
        example_response = tf.strings.reduce_join(example_response_list, separator=' ')
        example_text = tf.strings.reduce_join(example_text_list, separator=' ')

        example_output = tf.strings.join(
                ["input: ", example_sentence, "\nresponse: ", example_response,
                    "\nmodel response: ", example_text])
        tf.summary.text('example_output', example_output)

        # gradients
        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
                gradients, self.max_gradient_norm)

        # optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = train_op = optimizer.apply_gradients(
                zip(clipped_gradients, params), global_step=global_step)

        # summaries
        self.summaries = tf.summary.merge_all()

        # initialize embeddings
        self.sess.run(embedding_init, feed_dict={embedding_placeholder: self.word2vec})

        # global variables initializer
        self.sess.run(tf.global_variables_initializer())

    def train(self, training_data, test_data, num_epochs=1000000):
        print("entering train function")

        for i in range(num_epochs):
            if i % 50 == 0:
                print()
                print("epoch: ", i, " ", end='')

            # get training batch
            sentences, responses, sentence_lens, response_lens = \
                get_training_batch(training_data, self.batch_size, 
                        self.max_sentence_len)

            # feed into model
            feed_dict = {
                    self.sentences: sentences,
                    self.responses: responses,
                    self.sentence_lens: sentence_lens,
                    self.response_lens: response_lens
                    }

            if i % 100 == 0:
                _, summary = self.sess.run([self.train_op, self.summaries], feed_dict=feed_dict)
                self.writer.add_summary(summary, i)
            else:
                self.sess.run([self.train_op], feed_dict=feed_dict)


