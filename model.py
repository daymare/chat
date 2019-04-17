

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
    def __init__(self, enc_units, batch_size, embedding):
        super(PersonaEncoder, self).__init__()

        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = embedding

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
    def __init__(self, enc_units, batch_size, embedding):
        super(Encoder, self).__init__()

        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = embedding
        self.gru = gru(self.enc_units)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))

class Decoder(tf.keras.Model):
    def __init__(self, dec_units, vocab_size, batch_size, embedding):
        super(Decoder, self).__init__()
        
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = embedding
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

        embedding = tf.keras.layers.Embedding(
                input_dim=self.config.vocab_size,
                output_dim=self.config.embedding_dim,
                weights=word2vec,
                trainable=False)
        
        self.persona_encoder = PersonaEncoder(
                self.config.num_units, 
                self.config.batch_size,
                embedding)
        self.encoder = Encoder(
                self.config.num_units, 
                self.config.batch_size,
                embedding)
        self.decoder = Decoder(
                self.config.num_units, 
                self.config.vocab_size,
                self.config.batch_size,
                embedding)

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
        perplexity = tf.exp(loss_)
        return tf.reduce_mean(loss_), tf.reduce_mean(perplexity)

    def load_config(self, config, word2vec, id2word, word2id):
        self.config = config

        self.word2vec = word2vec
        self.id2word = id2word
        self.word2id = word2id


    def load(self, checkpoint_dir):
        """ load the model from a save file """
        self.checkpoint.restore(
                tf.train.latest_checkpoint(checkpoint_dir))

    def train(self, train_data, test_data, num_steps):
        global_step = tf.train.get_or_create_global_step()

        # tensorboard setup
        if self.config.save_summary == True:
            logdir = self.config.logdir

            summary_writer = tf.contrib.summary.create_file_writer(logdir)
            summary_writer.set_as_default()

        # train loop
        for step in range(num_steps):
            global_step.assign_add(1)

            start = time.time()

            hidden = self.encoder.initialize_hidden_state()
            loss = 0.0
            ppl = 0.0

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
                # TODO double check padding isn't screwing up the encoder training.
                # May want to restructure how we are doing input
                enc_output, enc_hidden = self.encoder(sentences, hidden)
                dec_hidden = enc_hidden

                persona_embeddings = self.persona_encoder(personas)

                dec_input = tf.expand_dims([self.word2id[
                    '<pad>']] * self.config.batch_size, 1)

                # Teacher forcing - feed the target as the next input
                for t in range(1, len(responses[0])):
                    # passing enc_output to the decoder
                    predictions, dec_hidden = self.decoder(
                            dec_input,
                            persona_embeddings,
                            dec_hidden)

                    sample_loss, sample_ppl = \
                        self.loss_function(responses[:, t], predictions)

                    loss += sample_loss
                    ppl += sample_ppl

                    # using teacher forcing
                    dec_input = tf.expand_dims(responses[:, t], 1)


            batch_loss = (loss / self.config.batch_size)
            batch_ppl = (ppl / self.config.batch_size)
            variables = self.encoder.variables + self.decoder.variables
            gradients = tape.gradient(loss, variables)
            gradients, _ = tf.clip_by_global_norm(gradients,
                    self.config.max_gradient_norm)

            self.optimizer.apply_gradients(zip(gradients, variables))

            # record summaries
            if self.config.save_summary == True:
                # record eval loss
                # TODO make parameter for how often to run eval
                if step % 10 == 0:
                    with (tf.contrib.summary.
                            always_record_summaries()):
                        # run eval
                        eval_loss = self.eval(test_data)

                        # record eval performance
                        tf.contrib.summary.scalar('loss', eval_loss)

                with (tf.contrib.summary.
                        record_summaries_every_n_global_steps(
                            self.config.save_frequency)):
                    tf.contrib.summary.scalar('loss', batch_loss)
                    tf.contrib.summary.scalar('perplexity', batch_ppl)

            # print out progress
            print('\n')
            print('Batch {}'.format(step + 1))
            print('Loss: {:.4f}'.format(batch_loss.numpy()))
            print('Perplexity: {:.4f}'.format(batch_ppl.numpy()))
            print('Time taken for 1 step {} sec'.format(
                time.time() - start), flush=True)

            # save the model every x batches
            if ((step + 1) % self.config.model_save_interval == 0
                    and self.config.save_model == True):
                logging.debug('Saving model to: {}'.format(
                    self.config.checkpoint_dir))
                self.checkpoint.save(
                    file_prefix = self.config.checkpoint_dir)

    def eval(self, test_data):
        """ get loss on eval set
        """
        total_loss = 0.0
        num_samples = 0

        for persona, partner_sentences, agent_responses in (
                get_eval_iterator(test_data,
                    self.config.max_sentence_len)):

            # process persona information
            persona = tf.expand_dims(persona, 0)
            persona_embeddings = self.persona_encoder(persona)

            # initialize encoder hidden state
            enc_hidden = self.encoder.initialize_hidden_state()

            # process each sentence in the conversation
            for i in range(len(partner_sentences)):
                num_samples += 1

                # encode partner sentence
                partner_sentence = partner_sentences[i]
                partner_sentence.append(self.word2id['<pad>'])
                _, enc_hidden = self.encoder(partner_sentence, enc_hidden)

                # run decoder
                dec_input = tf.expand_dims([self.word2id[
                    '<pad>']], 0)

                for t in range(len(responses[0])):
                    predictions, dec_hidden = self.decoder(
                            dec_input,
                            persona_embeddings,
                            dec_hidden)

                    total_loss += self.loss_function(agent_responses[t], predictions)
                    predicted_id = tf.argmax(predictions[0]).numpy()

                    # check if decoder is done
                    if self.id2word[predicted_id] == '<pad>':
                        continue

                    # feed predicted id back to decoder
                    dec_input = tf.expand_dims([predicted_id], 0)

                # encode dataset response
                agent_sentence = agent_responses[i]
                agent_sentence.append(self.word2id['<pad>'])
                _, enc_hidden = self.encoder(agent_sentence, enc_hidden)

            # return average eval loss over the whole set
            return total_loss / num_samples

    def call(self, inputs, reset=False):
        """ perform inference on inputs
        if reset=True then forget all past conversation.
        Otherwise cache conversation for future reference.

        inputs - conversation up to point of inference
            or next sentence to respond to.
            encoded as word ids

        output - response encoded as word ids
        """
        pass




