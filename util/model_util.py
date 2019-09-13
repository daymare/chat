


import tensorflow as tf



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

def initialize_multilayer_hidden_state(layer_sizes, batch_size):
    hidden = []
    for layer in range(len(layer_sizes)):
        layer_size = layer_sizes[layer]

        layer_hidden = tf.zeros([batch_size, layer_size])
        hidden.append(layer_hidden)

    return hidden



