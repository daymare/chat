
import numpy as np
import tensorflow as tf


class Seq2SeqBot(object):
    def __init__(self, config, sess, word2vec):
        self.n_hidden = config.hidden_size
        self.max_sentence_len = config.max_sentence_len

        self.sess = sess
        self.word2vec = word2vec
        self.target_sequence_length = config.target_sequence_length

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
            self.sentences = tf.placeholder(
                    tf.int32, [None, 2, self.max_sentence_len])
            self.sentence_lens = tf.placeholder(tf.int32, None)
            self.response_sentence = tf.placeholder(
                    tf.int32, [None, self.max_sentence_len])
            self.response_lens = tf.placeholder(tf.int32, None)

            # embedding input
            encoder_embedding_input = tf.nn.embedding_lookup(
                    self.word2vec, self.sentences)
            encoder_embedding_input = tf.cast(encoder_embedding_input, 
                    tf.float32)
            encoder_embedding_input = tf.reshape(
                    encoder_embedding_input, 
                    (-1, self.max_sentence_len*2, 300))

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
                    self.response_len, time_major=True)

            # decoder
            decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell,
                    helper,
                    encoder_final_state,
                    )

            # dynamic decoding
            outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)
            logits = outputs.rnn_output

        # loss
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=outputs, logits=logits)
        # TODO make target weights matrix
        train_loss = (tf.reduce_sum(crossent * target_weights) / batch_size)

        # calculate and clip gradients
        params = tf.trainable_variables()
        gradients = tf.gradients(train_loss, params)
        # TODO set max gradient norm
        clipped_gradients, _ = tf.clip_by_global_norm(
                gradients, max_gradient_norm)

        # optimizer
        # TODO set learning rate
        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_step = optimizer.apply_gradients(
                zip(clipped_gradients, params))

