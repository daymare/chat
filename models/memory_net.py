

from models.base import Chatbot

from util.data_util import get_training_batch_full

import tensorflow as tf
import tensorflow_hub as hub

import sys
import logging



class ProfileMemoryBot(Chatbot):
    """ profile memory bot

    sequence to sequence bot augmented with memory to handle the profile.
    bi-directional LSTM encoder with single layer decoder.

    """
    def __init__(self, config, sess, word2vec, id2word):
        self.max_persona_sentences = config.max_persona_len
        self.max_conversation_len = config.max_conversation_len

        Chatbot.__init__(self, config, sess, word2vec, id2word)


    def build_model(self):
        logging.debug('setup input')
        self.setup_input()
        logging.debug('setup embeddings')
        self.setup_embeddings()

        logging.debug('setup encoder')
        self.setup_encoder()
        logging.debug('setup profile memory')
        self.setup_profile_memory()
        logging.debug('setup decoder')
        self.setup_decoder()

        logging.debug('setup training')
        self.setup_training()

    def setup_training(self):
        # loss
        padded_response = tf.pad(self.response, [[0,0],[0,1]], "CONSTANT",
                constant_values=0)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels = tf.stop_gradient(tf.one_hot(
                    padded_response, depth=self.vocab_size,
                    dtype=tf.float32)),
                logits = self.logits)
        self.perplexity = tf.exp(cross_entropy)
        self.loss = tf.reduce_mean(cross_entropy)

        # summaries for loss
        tf.summary.histogram('loss', self.loss)

        # summaries for perplexity
        mean_perplexity, var_perplexity = tf.nn.moments(self.perplexity, 1)
        mean_perplexity = tf.reduce_mean(mean_perplexity)
        var_perplexity = tf.reduce_mean(var_perplexity)

        # summarize the text input and output
        output_example = self.logits[0] # shape (max_sentence_len, dictionary_size)
        # convert logits to ids
        example_predictions = tf.argmax(output_example, 1) # shape (max_sentence_len)
        # convert ids to sentence
        example_input_list = tf.nn.embedding_lookup(self.id2word, self.context_sentences[0])
        example_response_list = tf.nn.embedding_lookup(self.id2word, self.response[0])
        example_text_list = tf.nn.embedding_lookup(self.id2word, example_predictions)
        # build sentences
        example_sentence = tf.strings.reduce_join(example_input_list, separator=' ')
        example_response = tf.strings.reduce_join(example_response_list, separator=' ')
        example_text = tf.strings.reduce_join(example_text_list, separator=' ')
        example_output = tf.strings.join(["input: ", example_sentence,
            "\nresponse: ", example_response, "\nmodel response: ", example_text])
        tf.summary.text('example_output', example_output)

        # gradients
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 
                self.max_gradient_norm)

        # optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = optimizer.apply_gradients(
                zip(clipped_gradients, params), global_step=global_step)

    def get_lstm_cell(self):
        return tf.contrib.rnn.LSTMCell(self.n_hidden)

    def setup_decoder(self):
        with tf.name_scope('decoder'):
            # build decoder
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    num_units = self.n_hidden,
                    memory = self.encoded_personas)
            decoder_cell = self.get_lstm_cell()
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    decoder_cell, attention_mechanism,
                    attention_layer_size=self.n_hidden)

            decoder_initial_state = decoder_cell.zero_state(self.batch_size, 
                    dtype=tf.float32).clone(cell_state=self.encoder_final_state)
            self.decoder_outputs, self.decoder_final_state = tf.nn.dynamic_rnn(
                    cell = decoder_cell,
                    inputs = self.decoder_embedding_input,
                    initial_state = decoder_initial_state,
                    dtype = tf.float32,
                    time_major = False,
                    scope = "decoder_rnn")

            # setup logits
            self.logits = tf.layers.dense(
                    inputs = self.decoder_outputs,
                    units = self.vocab_size,
                    name = "projection_layer")

            # summary histogram for decoder cell
            # TODO figure out whatever the heck the third value of variables
            # is
            weights, biases, _ = decoder_cell.variables
            tf.summary.histogram("decoder_cell_weights", weights)
            tf.summary.histogram("decoder_cell_biases", biases)

            # summary histogram for projection layer
            with tf.variable_scope("projection_layer", reuse=True):
                weights = tf.get_variable("kernel")
                bias = tf.get_variable("bias")
                tf.summary.histogram("projection_layer_weights", weights)
                tf.summary.histogram("projection_layer_bias", bias)
    
    def setup_profile_memory(self):
        # encode each persona with a bidirectional lstm
        with tf.name_scope('profile_memory_encoder'):
            profile_cell_fw = self.get_lstm_cell()
            profile_cell_bw = self.get_lstm_cell()

            """
            self.encoded_personas = tf.zeros(shape=(
                self.batch_size, 
                self.max_persona_sentences,
                self.n_hidden * 2))
            """

            encoding_array = tf.TensorArray(
                    dtype = tf.float32, 
                    size = self.max_persona_sentences,
                    element_shape = (
                        self.batch_size, self.n_hidden*2))

            encoding_array.write(0, tf.zeros(
                shape=(self.batch_size, self.n_hidden*2),
                dtype=tf.float32))

            i = tf.Variable(0)

            def body(i, current_encoding_array):
                # extract out the next persona sentence
                # across all batches
                current_p_sentence = self.persona_embedding_input[:, i, :]
                current_sentence_lens = self.persona_sentence_lens[:, i]

                # encode the current value
                outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw = profile_cell_fw,
                    cell_bw = profile_cell_bw,
                    inputs = current_p_sentence,
                    sequence_length = current_sentence_lens,
                    time_major = False,
                    dtype = tf.float32,
                    scope = "profile_memory_rnn")

                # set the current sentence of the embedding
                logging.debug("final states: " +
                        str(final_states[0].c.shape) + ", " + 
                        str(final_states[1].c.shape))
                concat_states = tf.concat(
                        [final_states[0].c, final_states[1].c],
                        axis = 1)
                logging.debug("state output: " +
                        str(concat_states.shape))

                current_encoding_array.write(i, concat_states)

                # increment one in the count
                return (i + 1, current_encoding_array)

            def condition(i, current_encoding_array):
                # if we have encoded all the things in the batch
                # then stop
                return i < self.max_persona_sentences

            tf.while_loop(
                    cond = condition, 
                    body = body, 
                    loop_vars = [i, encoding_array])

            self.encoded_personas = encoding_array.stack()

            logging.debug("encoded personas shape: " +
                    str(self.encoded_personas.shape))

            self.encoded_personas = tf.transpose(
                    self.encoded_personas, [1, 0, 2])
            logging.debug("shape after transpose: " +
                    str(self.encoded_personas.shape))



    def setup_encoder(self):
        with tf.name_scope('encoder'):
            # build encoder
            encoder_cell_fw = self.get_lstm_cell()
            encoder_cell_bw = self.get_lstm_cell()

            encoder_outputs, encoder_states = \
                    tf.nn.bidirectional_dynamic_rnn(
                    cell_fw = encoder_cell_fw,
                    cell_bw = encoder_cell_bw,
                    inputs = self.encoder_embedding_input,
                    sequence_length = self.context_sentence_lens,
                    time_major = False,
                    dtype = tf.float32,
                    scope = "encoder_rnn")

            self.encoder_outputs = encoder_outputs
            self.encoder_final_state = tf.concat(encoder_states, 2)
            
            # set up summary histograms
            weights, biases = encoder_cell_fw.variables
            tf.summary.histogram("encoder_cell_fw_weights", weights)
            tf.summary.histogram("encoder_cell_fw_biases", biases)
            weights, biases = encoder_cell_bw.variables
            tf.summary.histogram("encoder_cell_bw_weights", weights)
            tf.summary.histogram("encoder_cell_bw_biases", biases)

    def setup_embeddings(self):
        # embeddings
        self.embeddings = tf.Variable(
                tf.constant(0.0, shape=(self.vocab_size, self.vocab_dim)),
                trainable=False, name="word_embeddings")
        self.embedding_placeholder = tf.placeholder(
                dtype=tf.float32,
                shape=(self.vocab_size, self.vocab_dim))
        self.embedding_init = self.embeddings.assign(self.embedding_placeholder)

        # persona input
        self.persona_embedding_input = tf.nn.embedding_lookup(self.embeddings,
                self.persona_sentences)

        # embedding input
        self.encoder_embedding_input = tf.nn.embedding_lookup(self.embeddings,
                self.context_sentences)

        # add <pad> as the first word to decoder to have accurate input
        # during training as the decoder will receive pad to start
        # TODO placeholder may need to be something else since pad also
        # signifies the end of the sentence.
        
        # decoder response input
        decoder_response_input = tf.pad(self.response, [[0,0], [1,0]], 
                "CONSTANT", constant_values=0)
        self.decoder_embedding_input = tf.nn.embedding_lookup(self.embeddings,
                decoder_response_input)

    def setup_input(self):
        # input sentences
        self.persona_sentences = tf.placeholder(dtype=tf.int32, 
            shape=(self.batch_size, self.max_persona_sentences, 
                self.max_sentence_len))
        self.context_sentences = tf.placeholder(dtype=tf.int32,
            shape=(self.batch_size, 
                self.max_conversation_len * self.max_sentence_len))
        self.response = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_sentence_len))

        # input lengths
        self.persona_sentence_lens = tf.placeholder(dtype=tf.int32,
            shape=(self.batch_size, self.max_persona_sentences))
        self.context_sentence_lens = tf.placeholder(dtype=tf.int32,
            shape=(self.batch_size))
        self.response_sentence_len = tf.placeholder(dtype=tf.int32,
            shape=(self.batch_size))
        
    
    def train(self, training_data, test_data, num_epochs=1000000,
            save_summary=True, print_training=True):
        # TODO move save summary and print training to parameters
        # TODO move num epochs to parameter
        # TODO move to parameters
        dot_interval = 30
        dots_per_line = 60

        for i in range(num_epochs):
            if print_training == True:
                if i % dot_interval == 0:
                    print('.', end='')
                    sys.stdout.flush()
                if i % (dot_interval * dots_per_line) == 0:
                    print()
                    print("epoch: " + str(i) + " ", end='')

            # get training batch
            personas, sentences, responses, persona_lens, sentence_lens, \
                response_lens = get_training_batch_full(
                training_data, self.batch_size, self.max_sentence_len, 
                self.max_conversation_len, self.max_persona_sentences)

            logging.debug("personas: " + str(personas))
            logging.debug("sentences: " + str(sentences))
            logging.debug("responses: " + str(responses))
            logging.debug("persona_lens: " + str(persona_lens))
            logging.debug("sentence_lens: " + str(sentence_lens))
            logging.debug("response_lens: " + str(response_lens))

            logging.debug(type(personas))
            logging.debug(type(sentences))
            logging.debug(type(responses))
            logging.debug(type(persona_lens))
            logging.debug(type(sentence_lens))
            logging.debug(type(response_lens))

            logging.debug(personas.shape)
            logging.debug(sentences.shape)
            logging.debug(responses.shape)
            logging.debug(persona_lens.shape)
            logging.debug(sentence_lens.shape)
            logging.debug(response_lens.shape)

            # feed into model
            feed_dict = {
                self.persona_sentences : personas,
                self.context_sentences : sentences,
                self.response : responses,
                self.persona_sentence_lens : persona_lens,
                self.context_sentence_lens : sentence_lens,
                self.response_sentence_len : response_lens
                }
            
            # TODO move to parameter
            save_frequency = 100
            if save_summary == True and i % 100 == 0:
                _, summary = self.sess.run(
                    [self.train_op, self.summaries],
                    feed_dict=feed_dict)
                self.writer.add_summary(summary, i)
            else:
                _ = self.sess.run(
                    [self.train_op], feed_dict=feed_dict)

        



