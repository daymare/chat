

import time
import logging

import tensorflow as tf
import numpy as np

from util.train_util import get_batch_iterator
from util.train_util import get_loss
from util.train_util import calculate_hidden_cos_similarity

from util.model_util import lstm
from util.model_util import gru
from util.model_util import initialize_multilayer_hidden_state


class PersonaEncoder(tf.keras.Model):
    def __init__(self, layer_sizes, batch_size, embedding, gru_over_lstm):
        super(PersonaEncoder, self).__init__()

        self.batch_size = batch_size
        self.layer_sizes = layer_sizes
        self.embedding = embedding
        self.gru_over_lstm = gru_over_lstm

        # initialize cells
        self.cells = []
        for i in range(len(layer_sizes)):
            name = "PersonaEncoder_Layer{}".format(i)
            size = layer_sizes[i]
            if gru_over_lstm is True:
                self.cells.append(gru(size, name))
            else:
                self.cells.append(lstm(size, name))

    def call(self, personas):
        """
            personas - np array 
                (batch_size, max_persona_sentences, max_sentence_len)
            hidden - previous hidden vector
        """
        outputs = []

        # reshape personas to fit the model
        personas = np.transpose(personas, (1, 0, 2))

        for persona in personas:
            hidden = self.initialize_hidden_state()
            persona = self.embedding(persona)

            x = persona

            for layer in range(len(self.cells)):
                cell = self.cells[layer]
                layer_hidden = hidden[layer]

                if self.gru_over_lstm is True:
                    output, layer_hidden = cell(x, layer_hidden)
                else:
                    output, hidden1, hidden2 = cell(x, layer_hidden)
                    layer_hidden = [hidden1, hidden2]

                x = output

            outputs.append(layer_hidden)

        # reshape outputs to be what we expect
        outputs = tf.convert_to_tensor(outputs)
        if self.gru_over_lstm is True:
            outputs = tf.transpose(outputs, [1, 0, 2])
        else:
            outputs = tf.transpose(outputs, [2, 0, 1, 3])


        return outputs

    def initialize_hidden_state(self):
        return initialize_multilayer_hidden_state(self.layer_sizes, 
                self.batch_size, self.gru_over_lstm)

class Encoder(tf.keras.Model):
    def __init__(self, layer_sizes, batch_size, embedding, gru_over_lstm):
        super(Encoder, self).__init__()

        self.batch_size = batch_size
        self.layer_sizes = layer_sizes
        self.embedding = embedding
        self.gru_over_lstm = gru_over_lstm

        # initialize cells
        self.fw_cells = []
        self.bw_cells = []
        for i in range(len(layer_sizes)):
            name = "Encoder_Layer{}".format(i)
            size = layer_sizes[i]
            if gru_over_lstm is True:
                self.fw_cells.append(gru(size, name))
                self.bw_cells.append(gru(size, name))
            else:
                self.fw_cells.append(lstm(size, name))
                self.bw_cells.append(lstm(size, name))

    def call(self, x, hidden):
        """ run encoder on input and output results
            inputs:
                x - input conversation encoded as ids 
                    (batch_size, max_batch_sentence_len)
                hidden - hidden state to use
                    for GRU:
                        List(num_layers, (batch_size, units))
                    for LSTM:
                        List(num_layers, [(batch_size, units), 
                                            (batch_size, units)]

            outputs:
                output - results of running through network
                        (batch_size, max_batch_sentence_len, units)
                new_hidden - hidden states at the end of running through network
                        same types as hidden inputs
        """
        x = self.embedding(x)

        new_hidden = []
        for layer in range(len(self.fw_cells)):
            cell_fw = self.fw_cells[layer]
            cell_bw = self.bw_cells[layer]
            fw_hidden, bw_hidden = hidden[layer]

            if self.gru_over_lstm is True:
                fw_output, fw_hidden = cell_fw(x, fw_hidden)
                bw_output, bw_hidden = cell_bw(tf.reverse(x, [1]), bw_hidden)

                output = tf.concat([fw_output, bw_output], 2)
                layer_hidden = [fw_hidden, bw_hidden]
            else:
                # TODO update lstm to be bidirectional
                output, hidden1, hidden2 = cell(x, layer_hidden)
                layer_hidden = [hidden1, hidden2]

            new_hidden.append(layer_hidden)
            x = output

        # return outputs from the last layer 
        # and hidden state from the last layer last timestep
        return output, new_hidden

    def initialize_hidden_state(self):
        fw_hiddens = initialize_multilayer_hidden_state(self.layer_sizes, 
                        self.batch_size, self.gru_over_lstm)
        bw_hiddens = initialize_multilayer_hidden_state(self.layer_sizes, 
                        self.batch_size, self.gru_over_lstm)
        out_hidden = []
        for i in range(len(fw_hiddens)):
            hiddens = [fw_hiddens[i], bw_hiddens[i]]
            out_hidden.append(hiddens)
        return out_hidden


class Decoder(tf.keras.Model):
    def __init__(self, layer_sizes, vocab_size, batch_size, embedding, gru_over_lstm):
        super(Decoder, self).__init__()
        
        self.batch_size = batch_size
        self.layer_sizes = layer_sizes
        self.embedding = embedding
        self.gru_over_lstm = gru_over_lstm
        self.projection_layer = tf.keras.layers.Dense(vocab_size,
                name="projection")

        # initialize cells
        self.cells = []
        for i in range(len(layer_sizes)):
            name = "Decoder_Layer{}".format(i)
            size = layer_sizes[i]
            if gru_over_lstm is True:
                self.cells.append(gru(size, name))
            else:
                self.cells.append(lstm(size, name))

        # attention stuff
        attention_units = self.layer_sizes[0]
        self.W1 = tf.keras.layers.Dense(attention_units, name="W1")
        self.W2 = tf.keras.layers.Dense(attention_units, name="W2")
        self.V = tf.keras.layers.Dense(1, name="V")

    def call(self, x, persona_embeddings, hidden, use_persona_encoder=False):
        # x shape after passing through embedding: 
        # (batch_szie, 1, embedding_dim)
        x = self.embedding(x)

        if use_persona_encoder is True:
            # concatenate the two hidden states for hidden and persona embeddings
            # TODO really look at this are and make sure nothing is screwed up
            if self.gru_over_lstm is False:
                persona_embedding_list = [persona_embeddings[:,:,i,:] for i in range(persona_embeddings.shape[2])]
                persona_embeddings = tf.concat(persona_embedding_list, 2)

            # add time dimension to hidden state
            layer_hidden = hidden[0]
            if self.gru_over_lstm is False:
                # concatenate the two hidden states provided by lstm
                layer_hidden = tf.concat(layer_hidden, 1)
            hidden_w_time_axis = tf.expand_dims(layer_hidden, 1)

            # get scores
            w1_hidden = self.W1(hidden_w_time_axis)
            w2_persona = self.W2(persona_embeddings)
            score = self.V(w1_hidden + w2_persona)

            # get attention weights
            attention_weights = tf.nn.softmax(score, axis=1)

            # get context vector
            context_vector = attention_weights * persona_embeddings
            context_vector = tf.reduce_sum(context_vector, axis=1)
            context_vector = tf.expand_dims(context_vector, 1)

            x = tf.concat([context_vector, x], axis=-1)


        # call decoder cells
        new_hidden = []
        for layer in range(len(self.cells)):
            cell = self.cells[layer]
            layer_hidden = hidden[layer]

            if self.gru_over_lstm is True:
                output, layer_hidden = cell(x, layer_hidden)
            else:
                output, hidden1, hidden2 = cell(x, layer_hidden)
                layer_hidden = [hidden1, hidden2]

            new_hidden.append(layer_hidden)

            x = output

        dec_hidden = new_hidden

        # output shape: (batch_size, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape: (batch_size, vocab)
        x = self.projection_layer(output)

        return x, dec_hidden

    def initialize_hidden_state(self):
        return initialize_multilayer_hidden_state(self.layer_sizes, 
                self.batch_size, self.gru_over_lstm)


class Model(object):
    def __init__(self, config, word2vec, id2word, word2id):
        self.load_config(config, word2vec, id2word, word2id)

        # word embeddings
        embedding = tf.keras.layers.Embedding(
                input_dim=self.config.vocab_size,
                output_dim=self.config.embedding_dim,
                weights=word2vec,
                trainable=True)

        # get model sizes
        persona_encoder_sizes = [int(s_val) for s_val in config.persona_encoder_sizes]
        encoder_sizes = [int(s_val) for s_val in config.encoder_sizes]
        decoder_sizes = [int(s_val) for s_val in config.decoder_sizes]

        # ensure encoder and decoder are compatible
        assert encoder_sizes[-1] * 2 == decoder_sizes[0], \
                "encoder is not compatible with decoder"
        
        # persona encoder
        if self.config.use_persona_encoder is True:
            self.persona_encoder = PersonaEncoder(
                    persona_encoder_sizes, 
                    self.config.batch_size,
                    embedding,
                    self.config.gru_over_lstm)
        else:
            self.persona_encoder = None

        # encoder
        self.encoder = Encoder(
                encoder_sizes, 
                self.config.batch_size,
                embedding,
                self.config.gru_over_lstm)

        # decoder
        self.decoder = Decoder(
                decoder_sizes, 
                self.config.vocab_size,
                self.config.batch_size,
                embedding,
                self.config.gru_over_lstm)

        # optimizer and loss function
        self.optimizer = optimizer = \
            tf.train.AdamOptimizer(learning_rate=config.learning_rate)

        # global step and epoch
        self.global_step = tf.train.get_or_create_global_step()
        self.epoch = tf.Variable(0)

        # checkpoints
        checkpoint_dir = config.checkpoint_dir
        if self.config.use_persona_encoder is True:
            self.checkpoint = tf.train.Checkpoint(
                    embedding=embedding,
                    optimizer=optimizer,
                    persona_encoder=self.persona_encoder,
                    encoder=self.encoder,
                    decoder=self.decoder,
                    global_step = self.global_step,
                    epoch = self.epoch)
        else:
            self.checkpoint = tf.train.Checkpoint(
                    embedding=embedding,
                    optimizer=optimizer,
                    encoder=self.encoder,
                    decoder=self.decoder,
                    global_step = self.global_step,
                    epoch = self.epoch)

        self.checkpoint_manager = tf.contrib.checkpoint.CheckpointManager(
                self.checkpoint, directory=self.config.checkpoint_dir, max_to_keep=1)
        
        # inference
        self.encoder_cache = None
        self.persona_embeddings = None

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

    def save(self):
        self.checkpoint_manager.save()

    def load(self):
        """ load the model from a save file """
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        print("global step after load: {}".format(self.global_step.numpy()))

    def train(self, train_data, test_data, num_steps=-1, 
            num_epochs=-1, parameter_search=False):

        # keep track of average loss and ppl
        loss_history = []
        ppl_history = []
        hundred_batch_time = 0.0

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
        quit = False
        while quit is False:
            # iterate through one epoch
            for batch in get_batch_iterator(
                train_data,
                self.config.batch_size,
                self.word2id):

                self.global_step.assign_add(1)
                start = time.time()

                with tf.GradientTape() as tape:
                    # split out batch
                    personas, sentences, responses = batch

                    if self.config.input_independant is True:
                        personas = tf.zeros_like(personas)
                        sentences = tf.zeros_like(sentences)

                    tape.watch(sentences)
                    tape.watch(personas)
                    tape.watch(responses)

                    # predictions shape (predicted_words, batch_size, vocab_len)
                    predictions, logging_info = self.call(sentences, responses, personas)

                    # calculate cosine similarity between last two enc_hidden outputs
                    # this is to check that input is being processed meaningfully

                    if self.config.gru_over_lstm is True:
                        enc_hidden = logging_info["last_enc_hidden"][0]
                    else:
                        enc_hidden = logging_info["last_enc_hidden"][1][0]
                        print("last enc hidden: {}".format(logging_info["last_enc_hidden"]))

                    logging_info["enc_hidden_cos_similarity"] = \
                            calculate_hidden_cos_similarity(enc_hidden, 
                                    last_enc_hidden, self.config.gru_over_lstm)
                    last_enc_hidden = enc_hidden

                    # calculate loss and ppl
                    loss, ppl = get_loss(predictions, responses, self.loss_function)

                    # get model responses
                    model_response = []
                    for t in range(len(predictions)):
                        predicted_id = tf.argmax(predictions[t][0]).numpy()
                        model_response.append(predicted_id)

                batch_loss = loss
                batch_ppl = ppl

                # calculate gradient and apply
                if self.config.use_persona_encoder is True:
                    variables = (
                            self.persona_encoder.variables 
                            + self.encoder.variables 
                            + self.decoder.variables)
                else:
                    variables = (
                            self.encoder.variables
                            + self.decoder.variables)
                gradients = tape.gradient(loss, variables)

                gradients, _ = tf.clip_by_global_norm(gradients,
                        self.config.max_gradient_norm)

                self.optimizer.apply_gradients(zip(gradients, variables))

                # record average recent loss and ppl
                update_history(loss_history, batch_loss)
                update_history(ppl_history, batch_ppl)

                # run eval
                if self.config.run_eval is True and \
                        self.global_step.numpy() % self.config.eval_frequency == 0:
                    eval_loss, eval_ppl = self.eval(test_data)
                else:
                    eval_loss = -1.0
                    eval_ppl = -1.0

                # record summaries
                logging_info["eval_loss"] = eval_loss
                logging_info["eval_ppl"] = eval_ppl
                logging_info["batch_loss"] = batch_loss
                logging_info["batch_ppl"] = batch_ppl
                logging_info["personas"] = personas
                logging_info["sentences"] = sentences
                logging_info["responses"] = responses
                logging_info["model_response"] = model_response
                logging_info["variables"] = variables
                logging_info["gradients"] = gradients

                if self.config.save_summary == True:
                    self.record_summaries(logging_info)

                # print out progress
                hundred_batch_time += time.time() - start
                if self.config.print_training == True:
                    print('.', end='')
                    if self.global_step.numpy() % 50 == 0:
                        print()
                    if self.global_step.numpy() % 100 == 0:
                        recent_avg_loss = sum(loss_history) / len(loss_history)
                        recent_avg_ppl = sum(ppl_history) / len(ppl_history)
                        print("max memory usage (GB): {}".format(
                            tf.contrib.memory_stats.MaxBytesInUse() / 10**9))
                        print("recent average loss: {}".format(recent_avg_loss))
                        print("recent average ppl: {}".format(recent_avg_ppl))
                        print("Epoch {}".format(self.epoch.numpy()))
                        print("Batch {}".format(self.global_step.numpy()))
                        print("minutes for 100 batches: {}".format(
                            hundred_batch_time / 60))
                        hundred_batch_time = 0.0
                    print('', end='', flush=True)

                # save the model every x batches
                if ((self.global_step.numpy() + 1) % self.config.model_save_interval == 0
                        and self.config.save_model == True):
                    logging.debug('Saving model to: {}'.format(
                        self.config.checkpoint_dir))
                    self.save()

                # quit if we have done the correct number of steps
                if self.global_step.numpy() >= num_steps and num_steps > 0 and self.config.use_epochs is False:
                    quit = True
                    break

            self.epoch.assign_add(1)
            
            # quit if we have done the correct number of epochs
            if self.config.use_epochs is True and self.epoch.numpy() >= num_epochs and num_epochs > 0:
                quit = True

        print("max memory usage (GB): {}".format(
            tf.contrib.memory_stats.MaxBytesInUse() / 10**9))

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
                self.word2id):
            num_samples += self.config.batch_size

            # split out batch
            personas, sentences, responses = batch

            # run encoder
            hidden = self.encoder.initialize_hidden_state()
            enc_output, enc_hidden = self.encoder(sentences, hidden)
            
            dec_hidden = self.enc_dec_layer(enc_hidden)

            persona_embeddings = self.persona_encoder(personas)

            dec_input = tf.expand_dims([self.word2id['<pad>']]
                    * self.config.batch_size, 1)

            # run decoder with teacher forcing
            for t in range(0, len(responses[0])):
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

    def call(self, inputs, expected_outputs=None, personas=None, reset=True,
            cache=False):
        """ model call for all purposes.
            Supports inference, eval, and training.
        """
        logging_info = {}

        # encode persona
        if self.config.use_persona_encoder is True and personas is not None:
            persona_embeddings = self.persona_encoder(personas)
            self.persona_embeddings = persona_embeddings
        else:
            assert self.persona_embeddings is not None, \
                    "personas must be provided for initial call"
            persona_embeddings = self.persona_embeddings

        # run encoder
        if reset is True or self.encoder_cache is None:
            enc_hidden = self.encoder.initialize_hidden_state()
        else:
            enc_hidden = self.encoder_cache

        _, enc_hidden = self.encoder(inputs, enc_hidden)

        # note encoder hidden shape is:
        # List(num_layers, (batch_size, units)), for GRU
        # List(num_layers, [(batch_size, units), (batch_size, units)], for LSTM
        logging_info["last_enc_hidden"] = enc_hidden[-1]

        # setup decoder hidden state
        # note that enc_hidden must be the same dimension as the first layer 
        # of the decoder
        dec_hidden = self.decoder.initialize_hidden_state()
        dec_hidden[0] = tf.concat(enc_hidden[-1], 1)

        # run decoder
        # if outputs are available teacher forcing will be used
        if expected_outputs is not None:
            decoder_limit = len(expected_outputs[0]) - 1
        else:
            decoder_limit = self.config.max_sentence_len - 1

        dec_input = tf.expand_dims(
                [self.word2id['<start>']] * self.config.batch_size,
                1)

        dec_predictions = []
        if expected_outputs is None:
            dec_out_ids = []

        # TODO double check there isn't an off by one error here somewhere
        for t in range(0, decoder_limit):
            # process a word
            predictions, dec_hidden = self.decoder(
                    dec_input,
                    persona_embeddings,
                    dec_hidden,
                    self.config.use_persona_encoder)
            
            # predictions shape (batch_size, dict_size)
            # output shape (decoder_limit, batch_size, dict_size)
            dec_predictions.append(predictions)

            # teacher forcing
            if expected_outputs is not None:
                dec_input = tf.expand_dims(expected_outputs[:, t], 1)
            else:
                # for inference
                dec_input = tf.argmax(predictions, axis=1)
                dec_input = tf.expand_dims(dec_input, 0)

            # if inference check for <end> token
            if expected_outputs is None:
                # TODO eventually make this work for multibatch
                chosen_word = tf.argmax(predictions, axis=1)[0]
                dec_out_ids.append(chosen_word)
                if self.id2word[chosen_word] == "<end>":
                    break

        if expected_outputs is None:
            # run encoder on our output and cache the result
            dec_out_ids = tf.expand_dims(dec_out_ids, 0)
            _, enc_hidden = self.encoder(dec_out_ids, enc_hidden)
            self.encoder_cache = enc_hidden

        # return results
        return dec_predictions, logging_info

    def __call__(self, inputs, persona=None, reset=False):
        """ perform inference on inputs
        if reset=True then forget all past conversation.
        Otherwise cache conversation for future reference.

        inputs - conversation up to point of inference
            or next sentence to respond to.
            encoded as word ids

        persona - persona to use. Must be passed if reset 
            is true

        output - 
            response_str - response as string
            response_ids - response as ids
        """
        inputs = tf.expand_dims(inputs, 0)
        if persona is not None:
            persona = tf.expand_dims(persona, 0)

        predictions, _ = self.call(inputs, personas=persona, reset=reset)

        out_ids = []
        out_string = []
        for prediction in predictions:
            cur_id = tf.argmax(prediction[0])
            out_ids.append(cur_id)
            out_string.append(self.id2word[cur_id])

        out_string = " ".join(out_string)
        out_ids = np.array(out_ids)

        return out_string, out_ids
    
    def record_summaries(self, logging_info):
        li = logging_info

        # record eval loss
        if li["eval_loss"] > 0:
            with (tf.contrib.summary.always_record_summaries()):
                # record eval performance
                print('Eval loss: {:.4f}'.format(li["eval_loss"].numpy()))
                print('Eval perplexity: {:.4f}'.format(li["eval_ppl"].numpy()))
                tf.contrib.summary.scalar('eval_loss', li["eval_loss"])
                tf.contrib.summary.scalar('eval_ppl', li["eval_ppl"])

        # record all other summaries
        with (tf.contrib.summary.record_summaries_every_n_global_steps(
                    self.config.save_frequency)):
            # loss and perplexity
            tf.contrib.summary.scalar('loss', li["batch_loss"])
            tf.contrib.summary.scalar('perplexity', li["batch_ppl"])

            # enc hidden norm
            if self.config.gru_over_lstm is True:
                tf.contrib.summary.scalar('enc_hidden_norm', tf.norm(li["last_enc_hidden"][0]))
            else:
                tf.contrib.summary.scalar('enc_hidden_0_norm', tf.norm(li["last_enc_hidden"][0][0]))
                tf.contrib.summary.scalar('enc_hidden_1_norm', tf.norm(li["last_enc_hidden"][1][0]))


            # enc hidden cosine similarity
            tf.contrib.summary.scalar('enc_hidden_cos_sim', li["enc_hidden_cos_similarity"])

            # text output
            text_meta = tf.SummaryMetadata()
            text_meta.plugin_data.plugin_name = "text"

            # always take index 0 as our example output
            ## persona
            # personas shape: (batch size, max_persona_sentences, max_persona_sentence_len)
            persona = li["personas"][0] # (max_persona_sentences, max_persona_sentence_len)
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
            conversation = li["sentences"][0]
            conversation_words = []
            for i in range(len(conversation)):
                word = conversation[i]
                conversation_words.append(self.id2word[word])

            conversation_text = tf.convert_to_tensor(" ".join(conversation_words))
            tf.contrib.summary.generic('conversation', conversation_text, metadata=text_meta)

            ## response
            # response shape: (batch_size, max_sentence_len)
            response = li["responses"][0]
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
            for word in li["model_response"]:
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
            if self.config.use_persona_encoder is True:
                record_histograms(self.persona_encoder.cells, "PersonaEncoder")
            record_histograms(self.encoder.fw_cells, "Encoder_fw")
            record_histograms(self.encoder.bw_cells, "Encoder_bw")
            record_histograms(self.decoder.cells, "Decoder")

            tf.contrib.summary.histogram("encoder_final_hidden",
                    li["last_enc_hidden"])

            ## decoder histograms
            projection_kernel, projection_bias = self.decoder.projection_layer.variables
            if self.config.use_persona_encoder is True:
                w1_kernel, w1_bias = self.decoder.W1.variables
                w2_kernel, w2_bias = self.decoder.W2.variables
                v_kernel, v_bias = self.decoder.V.variables

                tf.contrib.summary.histogram("decoder_w1_kernel", w1_kernel)
                tf.contrib.summary.histogram("decoder_w1_bias", w1_bias)
                tf.contrib.summary.histogram("decoder_w2_kernel", w2_kernel)
                tf.contrib.summary.histogram("decoder_w2_bias", w2_bias)
                tf.contrib.summary.histogram("decoder_v_kernel", v_kernel)
                tf.contrib.summary.histogram("decoder_v_bias", v_bias)

            tf.contrib.summary.histogram("decoder_projection_kernel", projection_kernel)
            tf.contrib.summary.histogram("decoder_projection_bias", projection_bias)

            # gradient histograms
            for i in range(len(li["variables"])):
                variable = li["variables"][i]
                gradient = li["gradients"][i]

                try:
                    tf.contrib.summary.histogram("{}_gradient".format(variable.name[:-2]), gradient)
                    tf.contrib.summary.scalar("{}_gradient_mag".format(variable.name[:-2]), tf.norm(gradient))
                except Exception as e:
                    pass







