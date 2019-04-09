

import os
import time
import sys
import logging

import tensorflow as tf
import numpy as np

from util.data_util import get_training_batch_full


def lstm(units):
    return tf.keras.layers.CuDNNLSTM(units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='orthogonal')

def gru(units):
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNGRU(
                units,
                return_sequences=True,
                return_state=True,
                recurrent_initializer='glorot_uniform')
    else:
        return tf.keras.layers.GRU(
                units,
                return_sequences=True,
                return_state=True,
                recurrent_activation='sigmoid',
                recurrent_initializer='glorot_uniform')

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units,
            batch_size):
        super(Encoder, self).__init__()

        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size,
                embedding_dim)
        self.gru = gru(self.enc_units)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim,
            dec_units, batch_size):
        super(Decoder, self).__init__()
        
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(
                vocab_size, embedding_dim)
        self.gru = gru(self.dec_units)
        self.projection_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden):
        # x shape after passing through embedding: 
        # (batch_szie, 1, embedding_dim)
        x = self.embedding(x)

        output, state = self.gru(x)

        # output shape: (batch_size, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape: (batch_size, vocab)
        x = self.projection_layer(output)

        return x, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.dec_units))


class Model(object):
    def __init__(self, config, word2vec, id2word, word2id):
        self.load_config(config, word2vec, id2word, word2id)
        
        self.encoder = Encoder(
                self.config.vocab_size, 
                self.config.embedding_dim,
                self.config.num_units, 
                self.config.batch_size)
        self.decoder = Decoder(
                self.config.vocab_size, 
                self.config.embedding_dim,
                self.config.num_units, 
                self.config.batch_size)

        # optimizer and loss function
        optimizer = tf.train.AdamOptimizer()

        # checkpoints
        checkpoint_dir = config.checkpoint_dir
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                encoder=self.encoder,
                decoder=self.decoder)

    def loss_function(self, real, pred):
        mask = 1 - np.equal(real, 0)
        loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=real, logits=pred) * mask
        return tf.reduce_mean(loss_)

    def load_config(self, config, word2vec, id2word, word2id):
        self.config = config

        self.word2vec = word2vec
        self.id2word = id2word
        self.word2id = word2id


    def load(self, checkpoint_dir):
        """ load the model from a save file """
        self.checkpoint.restore(
                tf.train.latest_checkpoint(checkpoint_dir))

    def train(self, train_data, test_data, num_epochs):
        # TODO fix training
        for epoch in range(num_epochs):
            start = time.time()

            hidden = self.encoder.initialize_hidden_state()
            loss = 0

            # get training batch
            personas, sentences, responses, persona_lens, \
                    sentence_lens, response_lens = \
                    get_training_batch_full(
                        train_data, self.config.batch_size,
                        self.config.max_sentence_len,
                        self.config.max_conversation_len,
                        self.config.max_conversation_words,
                        self.config.max_persona_len)
            
            with tf.GradientTape() as tape:
                enc_output, enc_hidden = self.encoder(sentences, hidden)
                dec_hidden = enc_hidden
                logging.debug("dec hidden: {}".format(dec_hidden.shape))

                dec_input = tf.expand_dims([self.word2id[
                    '<pad>']] * self.config.batch_size, 1)

                # Teacher forcing - feed the target as the next input
                for t in range(1, len(responses[0])):
                    logging.debug("output number: {}".format(t))
                    logging.debug("memory usage in GB: {}".format(tf.contrib.memory_stats.BytesInUse() / 1000000000))
                    # passing enc_output to the decoder
                    predictions, dec_hidden = self.decoder(
                            dec_input,
                            dec_hidden)

                    loss += self.loss_function(responses[:, t], predictions)

                    # using teacher forcing
                    dec_input = tf.expand_dims(responses[:, t], 1)


            batch_loss = (loss / len(responses[0]))
            variables = encoder.variables + decoder.variables
            gradients = tape.gradient(loss, variables)

            self.optimizer.apply_gradients(zip(gradients, variables))

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss: {:.4f}'.format(
                    epoch + 1,
                    batch,
                    batch_loss.numpy()))

            # save the model every x batches
            # TODO make this use the parameter
            if (epoch + 1) % 10000 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)

            print('Time taken for 1 epoch {} sec\n'.format(
                time.time() - start))

    def call(self, inputs):
        pass




