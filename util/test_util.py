
import tensorflow as tf
import numpy as np


def set_test_config(config):
    """ set given config parameters to the default test config parameters
    """
    config.encoder_sizes = [50, 100, 200]
    config.persona_encoder_sizes = [200, 100, 50]
    config.decoder_sizes = [200, 120, 240]
    config.use_persona_encoder = True
    config.input_independant = False
    config.learning_rate = 0.004
    config.save_summary = False
    config.run_eval = False
    config.use_epochs = False
    config.print_training = False
    config.save_model = False
    config.load_model = False
    config.batch_size = 1

def get_test_batch(train_data):
    pass

def get_trainable_variables_numpy():
    """ get the tf trainable variables as their numpy values
        Should not change with training
    """
    trainable_vars = tf.compat.v1.trainable_variables()
    np_vars = []
    for variable in trainable_vars:
        np_vars.append(variable.numpy())

    return np_vars
