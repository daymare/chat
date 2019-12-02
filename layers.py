
import tensorflow as tf

from tf.keras.layers import Layer

from util.model_util import gru


class GRULayer(Layer):
    def __init__(self, output_dims, **kwargs):
        self.output_dim = output_dim
        self.name = kwargs["name"]
        super(GRULayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_shape = input_shape
        self.batch_size = input_shape[0]

        # build hidden state
        self._reset_hidden_state()

        # build layer
        self.cell = gru(self.output_dim, self.name)
        super(GRULayer, self).build(input_shape)

    def call(self, x, reset=False):
        if reset is True:
            self._reset_hidden_state()

        output, hidden = self.cell(x, self.hidden)
        self.hidden = hidden

        return output

    def compute_output_shape(self):
        return (self.batch_size, self.output_dim)

    def reset_hidden_state(self):
        self.hidden = tf.zeros([self.batch_size, self.output_dim])

class BiDirectionalLayer(Layer):
    def __init__(self, output_dims, cell_class, **kwargs):
        """
            output_dims - desired output dimensions (does not include batch size)
            cell_class - Layer class to use as the cell for the bidirectional layer
        """
        self.output_dim = output_dim
        self.name = kwargs["name"]
        self.cell_class = cell_class
        super(BiDirectionalLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_shape = input_shape
        self.batch_size = input_shape[0]
        
        # build hidden state
        self._reset_hidden_state()

        # build layer
        self.fw_cell = self.cell_class(self.output_dim, name=self.name + "fw")
        self.bw_cell = self.cell_class(self.output_dim, name=self.name + "bw")

        super(BiDirectionalLayer, self).build(input_shape)

    def call(self, x, reset=False):
        if reset is True:
            self._reset_hidden_state()

        fw_output = self.fw_cell(x)
        bw_output = self.bw_cell(x)

        return fw_output, bw_output


    def compute_output_shape(self, input_shape):
        return (self.batch_size, self.output_dim, self.output_dim)

    def get_hidden_state(self):
        return (self.fw_hidden, self.bw_hidden)

    def _reset_hidden_state(self):
        self.fw_cell.reset_hidden_state()
        self.bw_cell.reset_hidden_state()


class multi_rnn_layer(Layer):
    def __init__(self, output_dims, **kwargs):
        self.output_dim = output_dim
        self.name = kwargs["name"]
        pass

