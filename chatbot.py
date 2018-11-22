
import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core


class Seq2SeqBot(object):
    def __init__(self, config, sess, word2vec):
        self.n_hidden = config.hidden_size
        self.max_sentence_len = config.max_sentence_len

        self.sess = sess
        self.vocab_size = len(word2vec)
        self.word2vec = word2vec

        # hyperparams
        self.max_gradient_norm = 0.1 # TODO set this to something reasonable
        self.learning_rate = 0.1 # TODO set this to something reasonable

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
            self.response_lens = tf.placeholder(tf.int32, shape=[None])

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
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.response_sentence, logits=logits)

        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
                gradients, self.max_gradient_norm)

        # optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.apply_gradients(
                zip(clipped_gradients, params), global_step=global_step)


