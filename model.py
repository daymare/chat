

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


class PersonaEncoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units,
            batch_size):
        super(PersonaEncoder, self).__init__()

        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size,
                embedding_dim)

        self.gru = gru(self.enc_units)

    def call(self, personas):
        """
            personas - np array 
                (batch_size, max_persona_sentences, max_sentence_len)
            hidden - previous hidden vector
        """

        outputs = []

        # reshape personas to fit the gru
        personas = np.transpose(personas, (1, 0, 2))

        for persona in personas:
            hidden = self.initialize_hidden_state()
            persona = self.embedding(persona)
            output, _ = self.gru(persona, initial_state=hidden)
            last_output = output[:,-1,:]
            outputs.append(last_output)

        # reshape outputs to be what we expect
        outputs = tf.convert_to_tensor(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])

        return outputs

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))


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

        # attention stuff
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, x, persona_embeddings, hidden):
        # attention calculations
        # persona embeddings shape: (batch_size, max_persona_sentences, hidden_size)

        # hidden shape (batch size, hidden size)
        # hidden with time axis shape (batch size, 1, hidden size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape (batch size, max persona sentences, 1)
        W1_hidden = self.W1(hidden_with_time_axis)
        W2_persona = self.W2(persona_embeddings)
        score = self.V(W1_hidden + W2_persona)

        # attention weights shape (batch size, max persona sentences, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context vector shape after sum (batch size, hidden size)
        context_vector = attention_weights * persona_embeddings
        context_vector = tf.reduce_sum(context_vector, axis=1)

        # x shape after passing through embedding: 
        # (batch_szie, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation (batch size, 1, embedding dim + hidden size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

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
        
        self.persona_encoder = PersonaEncoder(
                self.config.vocab_size, 
                self.config.embedding_dim,
                self.config.num_units, 
                self.config.batch_size)
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
        self.optimizer = optimizer = tf.train.AdamOptimizer()

        # checkpoints
        checkpoint_dir = config.checkpoint_dir
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
                optimizer=optimizer,
                persona_encoder=self.persona_encoder,
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

                persona_embeddings = self.persona_encoder(personas)

                dec_input = tf.expand_dims([self.word2id[
                    '<pad>']] * self.config.batch_size, 1)

                # TODO implement attention over persona embeddings

                # Teacher forcing - feed the target as the next input
                for t in range(1, len(responses[0])):
                    # passing enc_output to the decoder
                    predictions, dec_hidden = self.decoder(
                            dec_input,
                            persona_embeddings,
                            dec_hidden)

                    loss += self.loss_function(responses[:, t], predictions)

                    # using teacher forcing
                    dec_input = tf.expand_dims(responses[:, t], 1)


            batch_loss = (loss / len(responses[0]))
            variables = self.encoder.variables + self.decoder.variables
            gradients = tape.gradient(loss, variables)

            self.optimizer.apply_gradients(zip(gradients, variables))

            # TODO make this use the parameters
            if epoch % 1 == 0:
                logging.debug('Epoch {} Loss: {:.4f}'.format(
                    epoch + 1,
                    batch_loss.numpy()))

            # save the model every x batches
            # TODO make this use the parameter
            if (epoch + 1) % 10000 == 0:
                self.checkpoint.save(file_prefix = checkpoint_prefix)

            """
            logging.debug('Time taken for 1 epoch {} sec\n'.format(
                time.time() - start))
            """

    def call(self, inputs):
        pass




