

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

        logging.debug('setup output')
        self.setup_output()

    def setup_output(self):
        ## get output text
        # shape (max_sentence_len, dictionary_size)
        output = self.logits[0] 
        # convert logits to ids
        # shape (max_sentence_len)
        self.output_predictions = predictions = \
                tf.argmax(output, 1) 
        output_text_list = tf.nn.embedding_lookup(
                self.id2word, predictions)
        self.text_output = tf.strings.reduce_join(
                output_text_list, separator=' ')

        ## get the dataset input and response text
        # convert ids to sentence
        context_list = tf.nn.embedding_lookup(
                self.id2word, self.context_sentences[0])
        response_list = tf.nn.embedding_lookup(
                self.id2word, self.response[0])
        # build sentences
        context_sentence = tf.strings.reduce_join(
                context_list, separator=' ')
        response_sentence = tf.strings.reduce_join(
                response_list, separator=' ')

        ## summarize
        tf.summary.text('output', self.text_output)
        tf.summary.text('context_sentences', context_sentence)
        tf.summary.text('dataset_response', response_sentence)

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
        tf.summary.scalar('loss', self.loss)

        # summaries for perplexity
        mean_perplexity, var_perplexity = tf.nn.moments(
                self.perplexity, 1)
        mean_perplexity = tf.reduce_mean(mean_perplexity)
        var_perplexity = tf.reduce_mean(var_perplexity)
        tf.summary.scalar('mean_perplexity', mean_perplexity)
        tf.summary.scalar('var_perplexity', var_perplexity)


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

    def get_lstm_cell(self, num_units=None, name=None):
        if num_units is None:
            num_units = self.n_hidden
        return tf.contrib.rnn.LSTMCell(num_units, name=name)

    def setup_decoder(self):
        with tf.name_scope('decoder'):
            self.logits = tf.no_op()

            # build decoder cell
            attention_mechanism = \
                    tf.contrib.seq2seq.LuongAttention(
                    num_units = self.n_hidden,
                    memory = self.encoded_personas)
            decoder_cell = self.get_lstm_cell(
                    name="decoder_cell")

            decoder_cell = \
                    tf.contrib.seq2seq.AttentionWrapper(
                    decoder_cell, attention_mechanism,
                    attention_layer_size=self.n_hidden,
                    name="decoder_cell_attn")

            decoder_initial_state = \
                    decoder_cell.zero_state(
                    self.batch_size, dtype=tf.float32)
            decoder_inital_state = \
                    decoder_initial_state.clone(
                    cell_state=self.encoder_final_state)

            # projection layer
            projection_layer = tf.layers.Dense(
                    units=self.vocab_size,
                    name="projection_layer")

            ## Train or Eval
            if self.mode != "inference":
                # training helper
                helper = tf.contrib.seq2seq.TrainingHelper(
                    self.decoder_embedding_input,
                    self.response_sentence_len,
                    time_major=False)

                # decoder
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell,
                    helper,
                    decoder_initial_state)

                # dynamic decode
                outputs, decoder_final_state, _ = \
                        tf.contrib.seq2seq.dynamic_decode(
                            decoder,
                            output_time_major=False)

                self.logits = projection_layer(
                        outputs.rnn_output)

                # summary histogram for decoder cell
                decoder_variables = decoder_cell.variables
                weights = decoder_variables[0]
                biases = decoder_variables[1]

                tf.summary.histogram("decoder_cell_weights", weights)
                tf.summary.histogram("decoder_cell_biases", biases)

            ## Inference
            else:
                start_tokens = tf.fill([self.batch_size],
                    0)
                end_token = 0
                
                # TODO beam search

                # greedy decode
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    self.embeddings, 
                    start_tokens,
                    end_token)

                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell,
                    helper,
                    decoder_initial_state,
                    output_layer=projection_layer)

                outputs, decoder_final_state, _ = \
                    tf.contrib.seq2seq.dynamic_decode(
                        decoder,
                        maximum_iterations = 100,
                        output_time_major=False,
                        swap_memory=True)

                self.logits = outputs.rnn_output
                sample_id = outputs.sample_id


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

            encoding_array = tf.TensorArray(
                    dtype = tf.float32, 
                    size = self.max_persona_sentences,
                    element_shape = (
                        self.batch_size, self.n_hidden*2))

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
                concat_states = tf.concat(
                        [final_states[0].c, final_states[1].c],
                        axis = 1)

                write_output = current_encoding_array.write(
                        i, concat_states)
                write_output.mark_used()


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

            self.encoded_personas = tf.transpose(
                    self.encoded_personas, [1, 0, 2])


    def setup_encoder(self):
        with tf.name_scope('encoder'):
            # build encoder
            encoder_cell = self.get_lstm_cell()

            encoder_outputs, self.encoder_final_state = \
                    tf.nn.dynamic_rnn(
                            cell=encoder_cell,
                            inputs=self.encoder_embedding_input,
                            sequence_length = self.context_sentence_lens,
                            time_major = False,
                            dtype = tf.float32,
                            scope = "encoder_rnn")

    def setup_embeddings(self):
        # embeddings

        # (vocab size, vocab dim) float32
        self.embeddings = tf.Variable(
                tf.constant(0.0, shape=(self.vocab_size, self.vocab_dim),
                    dtype=tf.float32),
                trainable=False, name="word_embeddings")
        # (vocab size, vocab dim) float32
        self.embedding_placeholder = tf.placeholder(
                dtype=tf.float32,
                shape=(self.vocab_size, self.vocab_dim))
        self.embedding_init = self.embeddings.assign(self.embedding_placeholder)

        # persona input
        # (?, max persona sentences, max sentence len, embedding dim) float32
        self.persona_embedding_input = tf.nn.embedding_lookup(self.embeddings,
                self.persona_sentences)

        # embedding input
        # (?, max conversationlen * max sentence len, embedding dim) int32
        self.encoder_embedding_input = tf.nn.embedding_lookup(self.embeddings,
                self.context_sentences)

        # add <pad> as the first word to decoder to have accurate input
        # during training as the decoder will receive pad to start
        # TODO placeholder may need to be something else since pad also
        # signifies the end of the sentence.
        # TODO experiment. See if shifting the response input
        # back by 1 causes the decoder to learn to just reproduce
        # it's input
        
        # decoder response input
        decoder_response_input = tf.pad(self.response, [[0,0], [1,0]], 
                "CONSTANT", constant_values=0)
        self.decoder_embedding_input = tf.nn.embedding_lookup(self.embeddings,
                decoder_response_input)

    def setup_input(self):
        # input sentences
        
        # (?, max persona sentences, max sentence len) int32
        self.persona_sentences = tf.placeholder(dtype=tf.int32, 
            shape=(self.batch_size, self.max_persona_sentences, 
                self.max_sentence_len),
            name="persona_sentences")
        # (?, max conversation len * max sentence len) int32
        self.context_sentences = tf.placeholder(dtype=tf.int32,
            shape=(self.batch_size, 
                self.max_conversation_len * self.max_sentence_len),
            name="context_sentences")
        # (?, max sentence len) int32
        self.response = tf.placeholder(tf.int32, 
                shape=(self.batch_size, self.max_sentence_len),
                name="response")

        # input lengths
        self.persona_sentence_lens = tf.placeholder(dtype=tf.int32,
            shape=(self.batch_size, self.max_persona_sentences),
            name="persona_sentence_lens")
        self.context_sentence_lens = tf.placeholder(dtype=tf.int32,
            shape=(self.batch_size),
            name="context_sentence_lens")
        self.response_sentence_len = tf.placeholder(dtype=tf.int32,
            shape=(self.batch_size),
            name="response_sentence_len")
        
    
    def train(self, training_data, test_data):
        recent_losses = []
        recent_perplexities = []

        def add_recent_value(recent_list, new_value, max_values=100):
            recent_list.append(new_value)
            if len(recent_list) > max_values:
                recent_list.pop(0)

        for i in range(self.num_epochs):
            # print training progress
            if self.print_training == True:
                if i % self.print_dot_interval == 0:
                    print('.', end='')
                    sys.stdout.flush()
                if i % (self.print_dot_interval * self.dots_per_line) == 0:
                    print()
                    print("epoch: " + str(i) + " ", end='')

            # get training batch
            personas, sentences, responses, persona_lens, sentence_lens, \
                response_lens = get_training_batch_full(
                training_data, self.batch_size, self.max_sentence_len, 
                self.max_conversation_len, self.max_persona_sentences)

            # build feed dict
            feed_dict = {
                self.persona_sentences : personas,
                self.context_sentences : sentences,
                self.response : responses,
                self.persona_sentence_lens : persona_lens,
                self.context_sentence_lens : sentence_lens,
                self.response_sentence_len : response_lens
                }

            loss = None
            perplexity = None
            
            # feed into model
            if self.save_summary == True and \
                    i % self.save_frequency == 0:
                _, loss, perplexity, summary = self.sess.run(
                    [self.train_op, self.loss, self.perplexity, 
                        self.summaries],
                    feed_dict=feed_dict)
                self.writer.add_summary(summary, i)
            else:
                _, loss, perplexity = self.sess.run(
                    [self.train_op, self.loss,
                        self.perplexity], feed_dict=feed_dict)

            add_recent_value(recent_losses, loss)
            add_recent_value(recent_perplexities, perplexity)

            # save the model
            if self.save_model and \
            i % self.model_save_interval == 0:
                print()
                print()
                print("saving model")
                print()

                self.saver.save(self.sess, 
                        self.model_save_filepath)


        # return final loss and perplexity
        average_recent_loss = sum(recent_losses) / \
                float(len(recent_losses))
        average_recent_perplexities = sum(recent_perplexities) / \
                float(len(recent_perplexities))

        return average_recent_loss, average_recent_perplexities



