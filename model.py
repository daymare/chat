

import os
import time
import sys
import logging
import pdb

import tensorflow as tf
import numpy as np

from util.data_util import get_training_batch_full
from util.data_util import get_batch_iterator


def lstm(units, name=None):
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNLSTM(units,
                return_sequences=True,
                return_state=True,
                trainable=True,
                name=name)
    else:
        return tf.keras.layers.LSTM(
                units,
                return_sequences=True,
                return_state=True,
                trainable=True,
                name=name)


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
    def __init__(self, layer_sizes, batch_size, embedding):
        super(PersonaEncoder, self).__init__()

        self.batch_size = batch_size
        self.layer_sizes = layer_sizes
        self.embedding = embedding

        # initialize cells
        self.cells = []
        for i in range(len(layer_sizes)):
            name = "PersonaEncoder_Layer{}".format(i)
            size = layer_sizes[i]
            self.cells.append(lstm(size, name))

    def call(self, personas):
        """
            personas - np array 
                (batch_size, max_persona_sentences, max_sentence_len)
            hidden - previous hidden vector
        """

        outputs = []

        # reshape personas to fit the lstm
        personas = np.transpose(personas, (1, 0, 2))

        for persona in personas:
            hidden = self.initialize_hidden_state()
            persona = self.embedding(persona)

            x = persona

            for layer in range(len(self.cells)):
                cell = self.cells[layer]
                layer_hidden = hidden[layer]

                output, hidden1, hidden2 = cell(x, layer_hidden)
                layer_hidden = [hidden1, hidden2]

                x = output

            outputs.append(layer_hidden)

        # reshape outputs to be what we expect
        outputs = tf.convert_to_tensor(outputs)
        print("outputs: {}".format(outputs.shape))
        outputs = tf.transpose(outputs, [2, 0, 1, 3])

        return outputs

    def initialize_hidden_state(self):
        hidden = []
        for layer in range(len(self.cells)):
            layer_size = self.layer_sizes[layer]
            layer_hidden = [tf.zeros((self.batch_size, layer_size)),
                    tf.zeros((self.batch_size, layer_size))]
            hidden.append(layer_hidden)

        return hidden

class Encoder(tf.keras.Model):
    def __init__(self, layer_sizes, batch_size, embedding):
        super(Encoder, self).__init__()

        self.batch_size = batch_size
        self.layer_sizes = layer_sizes
        self.embedding = embedding

        # initialize cells
        self.cells = []
        for i in range(len(layer_sizes)):
            name = "Encoder_Layer{}".format(i)
            size = layer_sizes[i]
            self.cells.append(lstm(size, name))

    def call(self, x, hidden):
        x = self.embedding(x)

        for layer in range(len(self.cells)):
            cell = self.cells[layer]
            layer_hidden = hidden[layer]

            output, hidden1, hidden2 = cell(x, layer_hidden)
            layer_hidden = [hidden1, hidden2]

            x = output

        # return outputs from the last layer 
        # and hidden state from the last layer last timestep
        return output, layer_hidden

    def initialize_hidden_state(self):
        hidden = []
        for layer in range(len(self.cells)):
            layer_size = self.layer_sizes[layer]
            layer_hidden = [tf.zeros((self.batch_size, layer_size)),
                tf.zeros((self.batch_size, layer_size))]
            hidden.append(layer_hidden)

        return hidden

class Decoder(tf.keras.Model):
    def __init__(self, dec_units, vocab_size, batch_size, embedding):
        super(Decoder, self).__init__()
        
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = embedding
        self.cell = lstm(dec_units, "Decoder")
        self.projection_layer = tf.keras.layers.Dense(vocab_size,
                name="projection")

        # attention stuff
        self.W1 = tf.keras.layers.Dense(self.dec_units, name="W1")
        self.W2 = tf.keras.layers.Dense(self.dec_units, name="w2")
        self.V = tf.keras.layers.Dense(1, name="V")

    def call(self, x, persona_embeddings, hidden):
        # attention calculations
        # persona embeddings shape: (batch_size, max_persona_sentences, hidden_size)

        # hidden shape (batch size, hidden size)
        # hidden with time axis shape (batch size, 1, hidden size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape (batch size, max persona sentences, 1)
        """
        W1_hidden = self.W1(hidden_with_time_axis)
        W2_persona = self.W2(persona_embeddings)
        score = self.V(W1_hidden + W2_persona)

        # attention weights shape (batch size, max persona sentences, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context vector shape after sum (batch size, hidden size)
        context_vector = attention_weights * persona_embeddings
        context_vector = tf.reduce_sum(context_vector, axis=1)
        """

        # x shape after passing through embedding: 
        # (batch_szie, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation (batch size, 1, embedding dim + hidden size)
        #x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, dec_hidden1, dec_hidden2 = self.cell(x, hidden)
        dec_hidden = [dec_hidden1, dec_hidden2]

        # output shape: (batch_size, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape: (batch_size, vocab)
        x = self.projection_layer(output)

        return x, dec_hidden

    def initialize_hidden_state(self):
        return [tf.zeros((self.batch_size, self.dec_units)),
                tf.zeros((self.batch_size, self.dec_units))]


class Model(object):
    def __init__(self, config, word2vec, id2word, word2id):
        self.load_config(config, word2vec, id2word, word2id)

        embedding = tf.keras.layers.Embedding(
                input_dim=self.config.vocab_size,
                output_dim=self.config.embedding_dim,
                weights=word2vec,
                trainable=True)

        persona_encoder_sizes = [int(s_val) for s_val in config.persona_encoder_sizes]
        encoder_sizes = [int(s_val) for s_val in config.encoder_sizes]
        
        self.persona_encoder = PersonaEncoder(
                persona_encoder_sizes, 
                self.config.batch_size,
                embedding)

        self.encoder = Encoder(
                encoder_sizes, 
                self.config.batch_size,
                embedding)
        self.decoder = Decoder(
                self.config.decoder_units, 
                self.config.vocab_size,
                self.config.batch_size,
                embedding)

        # layer for translating from encoder hidden
        # to decoder hidden
        self.enc_dec_layer = tf.keras.layers.Dense(self.config.decoder_units)

        # optimizer and loss function
        self.optimizer = optimizer = \
            tf.train.AdamOptimizer(learning_rate=config.learning_rate)

        # checkpoints
        checkpoint_dir = config.checkpoint_dir
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
                embedding=embedding,
                optimizer=optimizer,
                persona_encoder=self.persona_encoder,
                encoder=self.encoder,
                decoder=self.decoder)

    def loss_function(self, real, pred):
        loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=real, logits=pred)
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

    def train(self, train_data, test_data, num_steps,
            parameter_search=False):
        global_step = tf.train.get_or_create_global_step()

        # keep track of average loss for parameter search
        if parameter_search == True:
            loss_history = []
            ppl_history = []

            def update_history(history, value, max=100):
                if len(history) >= max:
                    history.pop(0)
                history.append(value)

        # tensorboard setup
        if self.config.save_summary == True:
            logdir = self.config.logdir

            summary_writer = tf.contrib.summary.create_file_writer(logdir)
            summary_writer.set_as_default()

        last_enc_hidden = None

        # train loop
        for step in range(num_steps):
            global_step.assign_add(1)

            start = time.time()

            with tf.GradientTape() as tape:
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
                            self.config.max_persona_len,
                            self.word2id)

                tape.watch(sentences)
                tape.watch(personas)
                tape.watch(responses)
                tape.watch(hidden)

                _, enc_hidden = self.encoder(sentences, hidden)

                # calculate cosine similarity between last two enc_hidden outputs
                # this is to check that input is being processed meaningfully
                if last_enc_hidden is not None:
                    a = enc_hidden[1][0]
                    b = last_enc_hidden[1][0]
                    normalize_a = tf.nn.l2_normalize(a, 0)
                    normalize_b = tf.nn.l2_normalize(b, 0)
                    enc_hidden_cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b))
                else:
                    enc_hidden_cos_similarity = 0.0
                last_enc_hidden = enc_hidden

                # note that enc_hidden must be the same dim as decoder units
                dec_hidden = enc_hidden

                # disable persona for testing
                #persona_embeddings = self.persona_encoder(personas)
                persona_embeddings = None

                dec_input = tf.expand_dims([self.word2id[
                    '<start>']] * self.config.batch_size, 1)

                # Teacher forcing - feed the target as the next input
                model_response = [] # model response on index 0 for summary
                for t in range(0, len(responses[0])):
                    # passing enc_output to the decoder
                    # TODO re enable persona encoder
                    predictions, dec_hidden = self.decoder(
                            dec_input,
                            persona_embeddings,
                            dec_hidden)

                    sample_loss, sample_ppl = \
                        self.loss_function(responses[:, t], predictions)

                    # get model response
                    predicted_id = tf.argmax(predictions[0]).numpy()
                    model_response.append(predicted_id)

                    loss += sample_loss
                    ppl += sample_ppl

                    # using teacher forcing
                    dec_input = tf.expand_dims(responses[:, t], 1)


            batch_loss = loss
            batch_ppl = ppl

            # calculate gradient and apply
            # TODO ensure persona encoder variables have a gradient when it is re-enabled.
            variables = (
                    self.persona_encoder.variables 
                    + self.encoder.variables 
                    + self.decoder.variables
                    )
            gradients = tape.gradient(loss, variables)

            gradients, _ = tf.clip_by_global_norm(gradients,
                    self.config.max_gradient_norm)

            self.optimizer.apply_gradients(zip(gradients, variables))

            # record history for parameter search
            if parameter_search == True:
                update_history(loss_history, batch_loss)
                update_history(ppl_history, batch_ppl)

            # record summaries
            if self.config.save_summary == True:
                # record eval loss
                # TODO re-enable eval after validation
                if step % self.config.eval_frequency == 0 and False:
                    with (tf.contrib.summary.
                            always_record_summaries()):
                        # run eval
                        print('Running Eval')
                        eval_loss, eval_ppl = self.eval(test_data)

                        # record eval performance
                        print('Eval loss: {:.4f}'.format(eval_loss.numpy()))
                        print('Eval perplexity: {:.4f}'.format(eval_ppl.numpy()))
                        tf.contrib.summary.scalar('eval_loss', eval_loss)
                        tf.contrib.summary.scalar('eval_ppl', eval_ppl)

                # record all other summaries
                with (tf.contrib.summary.record_summaries_every_n_global_steps(
                            self.config.save_frequency)):
                    # loss and perplexity
                    tf.contrib.summary.scalar('loss', batch_loss)
                    tf.contrib.summary.scalar('perplexity', batch_ppl)

                    # enc hidden norm
                    tf.contrib.summary.scalar('enc_hidden_0_norm', tf.norm(enc_hidden[0]))
                    tf.contrib.summary.scalar('enc_hidden_1_norm', tf.norm(enc_hidden[1]))

                    # enc hidden cosine similarity
                    tf.contrib.summary.scalar('enc_hidden_cos_sim', enc_hidden_cos_similarity)

                    # text output
                    text_meta = tf.SummaryMetadata()
                    text_meta.plugin_data.plugin_name = "text"

                    # always take index 0 as our example output
                    ## persona
                    # personas shape: (batch size, max_persona_sentences, max_persona_sentence_len)
                    persona = personas[0] # (max_persona_sentences, max_persona_sentence_len)
                    persona_words = []
                    for sentence in persona:
                        for word in sentence:
                            if word.numpy() == 0:
                                break
                            else:
                                persona_words.append(self.id2word[word])
                    persona_text = tf.convert_to_tensor(" ".join(persona_words))
                    tf.contrib.summary.generic('persona', persona_text, metadata=text_meta)

                    ## sentence
                    # sentences shape: (batch_size, max_conversation_words)
                    conversation = sentences[0]
                    conversation_words = []
                    for i in range(len(conversation)):
                        word = conversation[i]
                        next_word = conversation[i+1]

                        if i == len(conversation) - 2:
                            break
                        elif word.numpy() == 0 and next_word.numpy() == 0:
                            break
                        else:
                            conversation_words.append(self.id2word[word])
                    conversation_text = tf.convert_to_tensor(" ".join(conversation_words))
                    tf.contrib.summary.generic('conversation', conversation_text, metadata=text_meta)

                    ## response
                    # response shape: (batch_size, max_sentence_len)
                    response = responses[0]
                    response_words = []
                    for word in response:
                        if word.numpy() == 0:
                            break
                        else:
                            response_words.append(self.id2word[word])
                    response_text = tf.convert_to_tensor(" ".join(response_words))
                    tf.contrib.summary.generic('response', response_text, metadata=text_meta)

                    ## model response
                    model_words = []
                    for word in model_response:
                        if word == 0:
                            break
                        else:
                            model_words.append(self.id2word[word])
                    model_text = tf.convert_to_tensor(" ".join(model_words))
                    tf.contrib.summary.generic('model_response', model_text, metadata=text_meta)

                    # model histograms
                    # encoder
                    def record_histograms(cells, name):
                        for i in range(len(cells)):
                            cell = cells[i]
                            kernel, recurrent_kernel, bias = cell.variables
                            tf.contrib.summary.histogram(name + "layer" + str(i+1) + "_Kernel", kernel)
                            tf.contrib.summary.histogram(name + "layer" + str(i+1) + "_ReccurentKernel", recurrent_kernel)
                            tf.contrib.summary.histogram(name + "layer" + str(i+1) + "_Bias", bias)
                    #record_histograms(self.persona_encoder.cells, "PersonaEncoder")
                    record_histograms(self.encoder.cells, "Encoder")

                    tf.contrib.summary.histogram("encoder_final_hidden", enc_hidden)

                    ## decoder histograms
                    projection_kernel, projection_bias = self.decoder.projection_layer.variables
                    """
                    w1_kernel, w1_bias = self.decoder.W1.variables
                    w2_kernel, w2_bias = self.decoder.W2.variables
                    v_kernel, v_bias = self.decoder.V.variables

                    tf.contrib.summary.histogram("decoder_w1_kernel", w1_kernel)
                    tf.contrib.summary.histogram("decoder_w1_bias", w1_bias)
                    tf.contrib.summary.histogram("decoder_w2_kernel", w2_kernel)
                    tf.contrib.summary.histogram("decoder_w2_bias", w2_bias)
                    tf.contrib.summary.histogram("decoder_v_kernel", v_kernel)
                    tf.contrib.summary.histogram("decoder_v_bias", v_bias)
                    """
                    tf.contrib.summary.histogram("decoder_projection_kernel", projection_kernel)
                    tf.contrib.summary.histogram("decoder_projection_bias", projection_bias)

                    kernel, recurrent_kernel, bias = self.decoder.cell.variables
                    tf.contrib.summary.histogram("decoder_kernel", kernel)
                    tf.contrib.summary.histogram("decoder_recurrentkernel", recurrent_kernel)
                    tf.contrib.summary.histogram("decoder_bias", bias)

                    # gradient histograms
                    for i in range(len(variables)):
                        variable = variables[i]
                        gradient = gradients[i]

                        try:
                            tf.contrib.summary.histogram("{}_gradient".format(variable.name[:-2]), gradient)
                            tf.contrib.summary.scalar("{}_gradient_mag".format(variable.name[:-2]), tf.norm(gradient))
                        except Exception as e:
                            pass

            # print out progress
            print('\n')
            print('Batch {}'.format(step + 1))
            print('Memory usage (MB): {}'.format(tf.contrib.memory_stats.BytesInUse() / 1000000))
            print('Max memory usage (MB): {}'.format(tf.contrib.memory_stats.MaxBytesInUse() / 1000000))
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

        if parameter_search == True:
            recent_avg_loss = sum(loss_history) / len(loss_history)
            recent_avg_ppl = sum(ppl_history) / len(ppl_history)

            return recent_avg_loss, recent_avg_ppl

    def eval(self, test_data):
        """ get loss on eval set
        """
        total_loss = 0.0
        total_ppl = 0.0 # perplexity
        num_samples = 0

        for batch in get_batch_iterator(
                test_data,
                self.config.batch_size,
                self.config.max_sentence_len,
                self.config.max_conversation_len,
                self.config.max_conversation_words,
                self.config.max_persona_len):
            num_samples += self.config.batch_size

            # split out batch
            personas, sentences, responses, persona_lens, \
                sentence_lens, response_lens = batch

            # run encoder
            hidden = self.encoder.initialize_hidden_state()
            enc_output, enc_hidden = self.encoder(sentences, hidden)
            
            dec_hidden = self.enc_dec_layer(enc_hidden)

            persona_embeddings = self.persona_encoder(personas)

            dec_input = tf.expand_dims([self.word2id['<pad>']]
                    * self.config.batch_size, 1)

            # run decoder with teacher forcing
            for t in range(1, len(responses[0])):
                predictions, dec_hidden = self.decoder(
                        dec_input,
                        persona_embeddings,
                        dec_hidden)

                sample_loss, sample_ppl = \
                    self.loss_function(responses[:, t], predictions)

                total_loss += sample_loss
                total_ppl += sample_ppl

                # using teacher forcing
                dec_input = tf.expand_dims(responses[:, t], 1)

            avg_loss = total_loss / num_samples
            avg_ppl = total_ppl / num_samples
            
            return avg_loss, avg_ppl

    def __call__(self, inputs, persona=None, reset=False):
        """ perform inference on inputs
        if reset=True then forget all past conversation.
        Otherwise cache conversation for future reference.

        inputs - conversation up to point of inference
            or next sentence to respond to.
            encoded as word ids

        persona - persona to use. Must be passed if reset 
            is true

        output - response encoded as word ids
        """

        # setup encoder hidden state
        if reset == True or self.encoder_cache is None:
            enc_hidden = self.encoder.initialize_hidden_state()
        else:
            enc_hidden = self.encoder_cache

        # encode persona
        if persona is not None:
            persona = tf.expand_dims(persona, 0)
            self.persona_embeddings = self.persona_encoder(persona)

        if self.persona_embeddings is None:
            raise Exception("need persona embeddings to run model!")

        # run encoder
        enc_output, enc_hidden = self.encoder(inputs, enc_hidden)
        dec_hidden = enc_hidden

        # run decoder
        dec_input = tf.expand_dims([self.word2id[
            '<pad>']] * 1, 1)

        str_result = ""
        id_result = []

        for t in range(1, self.config.max_sentence_len):
            predictions, dec_hidden = self.decoder(
                    dec_input,
                    self.persona_embeddings,
                    dec_hidden)
            
            predicted_id = tf.argmax(predictions[0]).numpy()
            predicted_word = self.id2word[predicted_id]

            str_result += predicted_word + " "
            id_result.append(predicted_id)

            if predicted_word == '<pad>':
                break

            dec_input = tf.expand_dims([predicted_id], 0)

        # run encoder on our output
        enc_input = tf.expand_dims(id_result, 0)
        _, enc_hidden = self.encoder(enc_input, enc_hidden)
        self.encoder_cache = enc_hidden

        # return result
        return str_result[:-1], id_result






