

from models.base import Chatbot


class ProfileMemoryBot(Chatbot):
    """ profile memory bot

    sequence to sequence bot augmented with memory to handle the profile.
    bi-directional LSTM encoder with single layer decoder.

    """
    def __init__(self, config, sess, word2vec, id2word):
        super().__init__(config, sess, word2vec, id2word)

        # TODO make into parameters
        # TODO modify data info grabber to get the max persona sentences
        # TODO modify data info grabber to get the max conversation length
        self.max_persona_sentences = config.max_persona_sentences
        self.max_conversation_length = config.max_conversation_length

    def build_model(self):
        self.setup_input()
        self.setup_embeddings()

        self.setup_encoder()
        self.setup_decoder()

        self.setup_training()

    def setup_training(self):
        # loss
        padded_response = tf.pad(self.response, [[0,0],[0,1]], "CONSTANT",
                constant_values=0)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels = tf.one_hot(padded_response, depth=self.vocab_size,
                    dtype=tf.float32).
                logits = self.logits)
        self.perplexity = tf.exp(cross_entropy)
        self.loss = tf.reduce_mean(cross_entropy)

        # summaries for loss
        tf.summary.histogram('loss', self.loss)

        # summaries for perplexity
        mean_perplexity, var_perplexity = tf.nn.moments(perplexity, 1)
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
        gradients = tf.gradients(loss, params)
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
            decoder_cell = self.get_lstm_cell()
            self.decoder_outputs, self.decoder_final_state = tf.nn.dynamic_rnn(
                    cell = decoder_cell,
                    inputs = decoder_embedding_input,
                    initial_state = encoder_final_state,
                    dtype = tf.float32,
                    time_major = False,
                    scope = "decoder_rnn")

            # setup logits
            self.logits = tf.layers.dense(
                    inputs = decoder_outputs,
                    units = self.vocab_size,
                    name = "projection_layer")

            # summary histogram for decoder cell
            weights, biases = decoder_cell.variables
            tf.summary.histogram("decoder_cell_weights", weights)
            tf.summary.histogram("decoder_cell_biases", biases)

            # summary histogram for projection layer
            with tf.variable_scope("projection_layer", reuse=True):
                weights = tf.get_variable("kernel")
                bias = tf.get_variable("bias")
                tf.summary.histogram("projection_layer_weights", weights)
                tf.summary.histogram("projection_layer_bias", bias)



    def setup_encoder(self):
        with tf.name_scope('encoder'):
            # build encoder
            encoder_cell = self.get_lstm_cell()
            self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(
                    cell = encoder_cell,
                    inputs = encoder_embedding_input,
                    sequence_length = sentence_lens,
                    time_major = False,
                    dtype = tf.float32,
                    scope = "encoder_rnn")
            
            # set up summary histograms
            weights, biases = encoder_cell.variables
            tf.summary.histogram("encoder_cell_weights", weights)
            tf.summary.histogram("encoder_cell_biases", biases)

    def setup_embeddings(self):
        # embeddings
        self.embeddings = tf.Variable(
                tf.constant(0.0, shape=(self.vocab_size, self.vocab_dim)),
                trainable=False, name="word_embeddings")
        self.embedding_placeholder = tf.placeholder(
                dtype=tf.float32,
                shape=(self.vocab_size, self.vocab_dim))
        self.embedding_init = embeddings.assign(embedding_placeholder)

        # embedding input
        self.encoder_embedding_input = tf.nn.embedding_lookup(embeddings,
                self.context_sentences)

        # add <pad> as the first word to decoder to have accurate input
        # during training as the decoder will receive pad to start
        decoder_response_input = tf.pad(self.response, [[0,0], [1,0]], 
                "CONSTANT", constant_values=0)
        self.decoder_embedding_input = tf.nn.embedding_lookup(self.embeddings,
                decoder_response_input)

    def setup_input(self):
        # input sentences
        self.persona_sentences = tf.placeholder(dtype=tf.int32, 
            shape=(self.batch_size, self.max_persona_sentences, 
                self.max_sentence_len)
        self.context_sentences = tf.placeholder(dtype=tf.int32,
            shape=(self.batch_size, 
                self.max_conversation_length, self.max_sentence_len)
        self.response = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_sentence_len))

        # input lengths
        self.persona_sentence_lens = tf.placeholder(dtype=tf.int32,
            shape=(self.batch_size, self.max_persona_sentences, self.max_sentence_len))
        self.context_sentence_lens = tf.placeholder(dtype=tf.int32,
            shape=(self.batch_size, self.max_conversation_length))
        self.response_sentence_len = tf.placeholder(dtype=tf.int32,
            shape=(self.batch_size)
        
    
    def train(self, training_data, test_data, num_epochs=1000000):
        def get_training_batch():
            pass
