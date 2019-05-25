import random
import math

import tensorflow as tf

from tensorflow.contrib.memory_stats import BytesInUse

def get_loguniform(low_exponent, high_exponent):
    value = random.random() * 10
    exponent = random.randint(low_exponent, high_exponent)

    return value * 10**exponent

def perform_parameter_search(model_class, flags,
        word2vec, id2word, word2id,
        parameter_ranges, training_data, 
        num_steps_per_parameter=1000, 
        result_filepath="parameter_search_results.txt"):
    """ perform random parameter search

    runs random parameter searches in the valid ranges until
    termination. (receive SIG-INT, CTRL-C)

    current searchable parameters:
        learning rate
        hidden layer size

    input:
        parameter_ranges:
            dictionary of parameter_name -> (low_value, high_value)
        training_data - training data, see load util.load_dataset
        num_epochs_per_parameter - number of epochs to run each parameter
            configuration for before returning the output
    output: 
        returns: None
        prints: parameter configurations and their scores
        saves parameter configurations and their scores to file
    """

    # open results file
    result_file = open(result_filepath, "a")

    # get parameter ranges
    learning_rate_range = None if "learning_rate" not in \
            parameter_ranges else \
            parameter_ranges["learning_rate"]

    hidden_size_range = None if "hidden_size" not in \
            parameter_ranges else parameter_ranges["hidden_size"]

    num_layers_range = None if "num_layers" not in \
            parameter_ranges else parameter_ranges["num_layers"]

    model = None

    flags.save_summary = False
    flags.print_training = True
    flags.debug = False
    flags.train_steps = num_steps_per_parameter

    def generate_parameter_config():
        config = {}
        config["learning_rate"] = get_loguniform(learning_rate_range[0], 
                learning_rate_range[1])

        def get_hidden_size():
            return random.randint(hidden_size_range[0],
                    hidden_size_range[1])
        def get_num_layers():
            return random.randint(num_layers_range[0],
                    num_layers_range[1])

        # generate persona encoder sizes
        persona_encoder_sizes = [str(get_hidden_size()) for i in range(get_num_layers())]
        encoder_sizes = [str(get_hidden_size()) for i in range(get_num_layers())]
        decoder_size = get_hidden_size()

        config["persona_encoder_sizes"] = persona_encoder_sizes
        config["encoder_sizes"] = encoder_sizes
        config["decoder_units"] = decoder_size

        return config

    def apply_parameter_config(config):
        print("applying config: {}".format(config), flush=True)
        tf.reset_default_graph()

        flags.learning_rate = config["learning_rate"]
        flags.persona_encoder_sizes = config["persona_encoder_sizes"]
        flags.encoder_sizes = config["encoder_sizes"]
        flags.decoder_units = config["decoder_units"]

        model = model_class(flags, word2vec, id2word, word2id)

        print("applied config")

        return model

    best_loss_config = None
    best_loss = None
    num_tests = 0
    tests_since_last_best = 0

    # test parameter configs
    while True:
        config = generate_parameter_config()
        model = apply_parameter_config(config)

        # train
        try:
            loss, perplexity = model.train(training_data,
                    None, num_steps_per_parameter, parameter_search=True)

            loss = loss.numpy()
            perplexity = perplexity.numpy()
        except:
            loss, perplexity = math.inf, math.inf

        num_tests += 1
        tests_since_last_best += 1

        # update best loss and output
        if best_loss == None or loss < best_loss:
            tests_since_last_best = 0
            best_loss_config = config
            best_loss = loss

        # output results
        def print_std_and_file(result_file, text):
            text += "\n"
            result_file.write(text)
            print(text, end='')

        print_std_and_file(result_file, "\n\n")
        print_std_and_file(result_file, "test_number: " + \
                str(num_tests))
        print_std_and_file(result_file, "memory usage (MB): {}"
                .format(BytesInUse().numpy() / 1000000))
        print_std_and_file(result_file, "tests since last best: " + \
                str(tests_since_last_best))
        print_std_and_file(result_file, "loss: " + str(loss) + \
                " " + str(config))
        print_std_and_file(result_file, "best loss: " + \
                str(best_loss) + " " + str(best_loss_config))

        result_file.flush()

