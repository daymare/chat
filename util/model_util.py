


import tensorflow as tf


def gru(units, name=None, backwards=False):
    return tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            trainable=True,
            recurrent_initializer='glorot_uniform',
            go_backwards=backwards,
            name=name)

def initialize_multilayer_hidden_state(layer_sizes, batch_size):
    hidden = []
    for layer in range(len(layer_sizes)):
        layer_size = layer_sizes[layer]

        layer_hidden = tf.zeros([batch_size, layer_size])
        hidden.append(layer_hidden)

    return hidden



