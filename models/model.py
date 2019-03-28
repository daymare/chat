
import tensorflow as tf

import numpy as np


def lstm(units):
    return tf.keras.layers.cuDNNLSTM(units)

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units,
            batch_size):
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size,
                embedding_dim)
        self.lstm = lstm(self.enc_units)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.lstm(x, initial_state=hidden)
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
        self.lstm = lstm(self.dec_units)
        self.projection_layer = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        # used for performing addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying
        # tanh(FC(EO) + FC(H))
        score = self.V(tf.nn.tanh(self.W1(enc_output) + \
                self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)

        # x shape after passing through embedding: 
        # (batch_szie, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation: 
        # (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x],
                axis=-1)

        # passing the concatenated vector to the LSTM
        output, state = self.lstm(x)

        # output shape: (batch_size, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape: (batch_size, vocab)
        x = self.projection_layer(output)

        return x, state, attention_weights

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.dec_units))


class Model(object):
    def __init__(self, config):
        vocab_inp_size = config.vocab_size
        embedding_dim = config.embedding_dim
        units = config.num_units
        batch_size = config.batch_size

        self.encoder = Encoder(vocab_inp_size, embedding_dim,
                units, batch_size)
        self.decoder = Decoder(vocab_tar_size, embedding_dim,
                units, batch_size)

        # optimizer and loss function
        optimizer = tf.train.AdamOptimizer()

        def loss_function(real, pred):
            mask = 1 - np.equal(real, 0)
            loss_ = tf.nn.sparse_softramx_cross_entropy_with_logits(
                    labels=real, logits=pred) * mask
            return tf.reduce_mean(loss_)

        # checkpoints
        checkpoint_dir = config.checkpoint_dir
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                encoder=encoder,
                decoder=decoder)

    def load_config(self, config):
        pass

    def load(self, checkpoint_dir):
        """ load the model from a save file """
        self.checkpoint.restore(
                tf.train.latest_checkpoint(checkpoint_dir))

    def train(self, dataset, num_epochs):
        for epoch in range(num_epochs):
            start = time.time()

            hidden = encoder.initialize_hidden_state()
            total_loss = 0
            
            for (batch, (inp, targ)) in enumerate(dataset):
                loss = 0

                with tf.GradientTape() as tape:
                    enc_output, enc_hidden = encoder(inp, hidden)
                    dec_hidden = enc_hidden

                    # Teacher forcing - feed the target as the next input
                    for t in range(1, targ.shape[1]):
                        # passing enc_output to the decoder
                        predictions, dec_hidden, _ = decoder(
                                dec_input,
                                dec_hidden,
                                enc_output)

                        loss += loss_function(targ[:, t], predictions)

                        # using teacher forcing
                        dec_input = tf.expand_dims(targ[:, t], 1)

                batch_loss = (loss / int(targ.shape[1]))
                total_loss += batch_loss
                variables = encoder.variables + decoder.variables
                gradients = tape.gradient(loss, variables)

                self.optimizer.apply_gradients(zip(gradients, variables))

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss: {:.4f}'.format(
                        epoch + 1,
                        batch,
                        batch_loss.numpy()))

            # save the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
            
            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                total_loss / N_BATCH))

            print('Time taken for 1 epoch {} sec\n'.format(
                time.time() - start)





    def call(self, inputs):
        pass




