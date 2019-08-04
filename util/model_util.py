


import tensorflow as tf



# TODO test using glorot uniform recurrent initialization

def lstm(units, name=None):
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNLSTM(units,
                return_sequences=True,
                return_state=True,
                trainable=True,
                recurrent_initializer='glorot_uniform',
                name=name)
    else:
        return tf.keras.layers.LSTM(
                units,
                return_sequences=True,
                return_state=True,
                trainable=True,
                recurrent_initializer='glorot_uniform',
                name=name)

def gru(units, name=None):
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNGRU(
                units,
                return_sequences=True,
                return_state=True,
                trainable=True,
                recurrent_initializer='glorot_uniform',
                name=name)
    else:
        return tf.keras.layers.GRU(
                units,
                return_sequences=True,
                return_state=True,
                trainable=True,
                recurrent_initializer='glorot_uniform',
                name=name)

def initialize_multilayer_hidden_state(layer_sizes, batch_size,
        gru_over_lstm):
    hidden = []
    for layer in range(len(layer_sizes)):
        layer_size = layer_sizes[layer]

        if gru_over_lstm is True:
            layer_hidden = tf.zeros([batch_size, layer_size])
        else:
            layer_hidden = [
                tf.zeros([batch_size, layer_size]),
                tf.zeros([batch_size, layer_size])
                ]
        hidden.append(layer_hidden)

    return hidden



