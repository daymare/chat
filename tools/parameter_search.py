import random
import math

import tensorflow as tf

def get_loguniform(low_exponent, high_exponent):
    value = random.random() * 10
    exponent = random.randint(low_exponent, high_exponent)

    return value * 10**exponent

def perform_parameter_search(model_class, tf_session, flags,
        word2vec, id2word,
        parameter_ranges, training_data, 
        num_epochs_per_parameter=2500, 
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

    model = None
    sess = tf_session

    flags.save_summary = False
    flags.print_training = True
    flags.debug = False
    flags.num_epochs = num_epochs_per_parameter

    def generate_parameter_config():
        config = {}
        config["learning_rate"] = get_loguniform(learning_rate_range[0], 
                learning_rate_range[1])
        config["hidden_size"] = random.randint(hidden_size_range[0],
                hidden_size_range[1])

        return config

    def apply_parameter_config(config, sess):
        tf.reset_default_graph()
        sess.close()
        sess = tf.Session()

        flags.learning_rate = config["learning_rate"]
        flags.hidden_size = config["hidden_size"]

        model = model_class(flags, sess, word2vec, id2word)

        return sess, model

    best_loss_config = None
    best_loss = None
    num_tests = 0
    tests_since_last_best = 0

    # test parameter configs
    while True:
        config = generate_parameter_config()
        sess, model = apply_parameter_config(config, sess)

        # train
        try:
            loss, perplexity = model.train(training_data,
                    None)
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
        print_std_and_file(result_file, "tests since last best: " + \
                str(tests_since_last_best))
        print_std_and_file(result_file, "loss: " + str(loss) + \
                " " + str(config))
        print_std_and_file(result_file, "best loss: " + \
                str(best_loss) + " " + str(best_loss_config))

        result_file.flush()
