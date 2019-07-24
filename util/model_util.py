


import tensorflow as tf


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

def initialize_multilayer_hidden_state(layer_sizes, batch_size):
    hidden = []
    for layer in range(len(layer_sizes)):
        layer_size = layer_sizes[layer]
        layer_hidden = [
            tf.zeros((batch_size, layer_size)),
            tf.zeros((batch_size, layer_size))
            ]
        hidden.append(layer_hidden)

    return hidden


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

