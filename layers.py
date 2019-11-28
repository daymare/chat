
import tensorflow as tf

from tf.keras.layers import Layer

from util.model_util import gru


class gru_layer(Layer):
    def __init__(self, output_dims, **kwargs):
        self.output_dim = output_dim
        self.name = kwargs["name"]
        super(gru_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_shape = input_shape
        self.batch_size = input_shape[0]
        
        # build hidden state
        self._reset_hidden_state()

        # build layer
        self.fw_cell = gru(self.output_dim, self.name)
        self.bw_cell = gru(self.output_dim, self.name)

        super(gru_layer, self).build(input_shape)

    def call(self, x, reset=False):
        if reset is True:
            self._reset_hidden_state()

        fw_output, fw_hidden = self.fw_cell(x, self.fw_hidden)
        bw_output, bw_hidden = self.bw_cell(x, self.bw_hidden)

        self.fw_hidden = fw_hidden
        self.bw_hidden = bw_hidden

        return fw_output, bw_output


    def compute_output_shape(self, input_shape):
        return (self.batch_size, self.output_dim)

    def get_hidden_state(self):
        return (self.fw_hidden, self.bw_hidden)

    def _reset_hidden_state(self):
        self.fw_hidden = tf.zeros([self.batch_size, self.output_dim])
        self.bw_hidden = tf.zeros([self.batch_size, self.output_dim])


class multi_rnn_layer(Layer):
    def __init__(self, output_dims, **kwargs):
        self.output_dim = output_dim        
        pass

