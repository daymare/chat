""" interpret_search.py
    read the data file and print statistics 
"""
import csv
import ast

import numpy as np
import matplotlib.pyplot as plt

class DataPoint:
    def __init__(self, row):
        self.loss = float(row[0])
        self.learning_rate = float(row[1])
        self.persona_encoder_sizes = ast.literal_eval(row[2])
        self.encoder_sizes = ast.literal_eval(row[3])
        self.decoder_sizes = ast.literal_eval(row[4])

        # convert layer sizes to integers
        self.persona_encoder_sizes = ([int(a) 
                for a in self.persona_encoder_sizes])
        self.encoder_sizes = ([int(a)
                for a in self.encoder_sizes])
        self.decoder_sizes = ([int(a)
                for a in self.decoder_sizes])

    @staticmethod
    def check_row_valid(row):
        # ensure first element in row is a double
        # in the future perhaps check each element in the row is what we want
        element = row[0]
        try:
            float(element)
            return True
        except ValueError:
            return False

class LayerSandwichStatistics:
    """ because encoder, decoder, and persona encoder are basically
        sandwiches made out of layers.

        So we will use this thing to hold and calculate statistics
        on their layers.

        Don't judge me.
    """

    def __init__(self, best_layers):
        """ best_layers - List[List[int]] list of layer sizes of the best 
                            configurations
        """
        self.num_layers_mean = None
        self.num_layers_var = None
        self.layers = None

        # get lists of stuff
        self.num_layers = num_layers = []
        self.layers_sizes = layers_sizes = []

        for i in range(len(best_layers)):
            layer_sizes = best_layers[i]

            num_layers.append(len(layer_sizes))

            for i in range(len(layer_sizes)):
                if len(layers_sizes) < i+1:
                    layers_sizes.append([])
                layers_sizes[i].append(layer_sizes[i])

        # get statistics of each thing
        self.num_layers_mean = np.mean(num_layers)
        self.num_layers_var = np.var(num_layers)

        self.layers = []
        for layer_sizes in layers_sizes:
            layer_mean = np.mean(layer_sizes)
            layer_var = np.var(layer_sizes)
            self.layers.append((layer_mean, layer_var))

def display_layer_statistics(best_points, max_layers):
    def plot_layer(layer_name, layer_stats, layer_index, axis):
        # plot num layers
        axis[layer_index, 0].boxplot(layer_stats.num_layers)
        axis[layer_index, 0].set_title(layer_name + " num layers")

        # plot each layer
        layers_sizes = layer_stats.layers_sizes
        for i in range(len(layers_sizes)):
            axis[layer_index, 1+i].boxplot(layers_sizes[i])
            axis[layer_index, 1+i].set_title(layer_name + 
                    " layer {}".format(i))

    # get statistics
    pe_stats, encoder_stats, decoder_stats = \
            get_layer_statistics(best_points)

    # display layer statistics in a plot
    fig, axs = plt.subplots(3, max_layers+1)

    plot_layer("persona encoder", pe_stats, 0, axs)
    plot_layer("encoder", encoder_stats, 1, axs)
    plot_layer("decoder", decoder_stats, 2, axs)

    plt.savefig('./search_plots.png')


def get_layer_statistics(best_points):
    persona_encoder_layers = []
    encoder_layers = []
    decoder_layers = []

    for point in best_points:
        persona_encoder_layers.append(point.persona_encoder_sizes)
        encoder_layers.append(point.encoder_sizes)
        decoder_layers.append(point.decoder_sizes)

    persona_encoder_stats = LayerSandwichStatistics(persona_encoder_layers)
    encoder_stats = LayerSandwichStatistics(encoder_layers)
    decoder_stats = LayerSandwichStatistics(decoder_layers)

    return persona_encoder_stats, encoder_stats, decoder_stats

def interpret_search(data_filepath):
    data_file = open(data_filepath, "r")
    data_reader = csv.reader(data_file, delimiter=',', quotechar='"')

    ## load datapoints
    datapoints = []
    for row in data_reader:
        if DataPoint.check_row_valid(row) is True:
            datapoints.append(DataPoint(row))

    ## calculate statistics
    # grab out the best percentile percent of the losses
    # remember that lower loss is better
    percentile = 20
    losses = [point.loss for point in datapoints]
    percentile_value = np.percentile(losses, percentile)

    # get the list of the best configurations
    best_datapoints = []
    for point in datapoints:
        if point.loss < percentile_value:
            best_datapoints.append(point)

    # get the ranges for each parameter
    learning_rate_low = np.inf
    learning_rate_high = -np.inf
    hidden_size_low = np.inf
    hidden_size_high = -np.inf
    num_layers_low = np.inf
    num_layers_high = -np.inf

    for point in best_datapoints:
        # learning rate
        learning_rate_low = min(learning_rate_low, point.learning_rate)
        learning_rate_high = max(learning_rate_high, point.learning_rate)
        
        # num layers
        num_layers_values = [
                len(point.persona_encoder_sizes), 
                len(point.encoder_sizes),
                len(point.decoder_sizes)
                ]
        for num_layers in num_layers_values:
            num_layers_low = min(num_layers_low, num_layers)
            num_layers_high = max(num_layers_high, num_layers)

        # hidden size
        hidden_sizes = (
                point.persona_encoder_sizes +
                point.encoder_sizes +
                point.decoder_sizes)
        for hidden_size in hidden_sizes:
            hidden_size_low = min(hidden_size_low, hidden_size)
            hidden_size_high = max(hidden_size_high, hidden_size)

    # get statistics on the layers
    display_layer_statistics(best_datapoints, num_layers_high)

    ## print out results
    print("{}th percentile value: {}".format(percentile, percentile_value))
    print("{}th percentile range:".format(percentile))
    print("    learning rate low: {}".format(learning_rate_low))
    print("    learning rate high: {}".format(learning_rate_high))
    print("    hidden size low: {}".format(hidden_size_low))
    print("    hidden size high: {}".format(hidden_size_high))
    print("    num layers low: {}".format(num_layers_low))
    print("    num layers high: {}".format(num_layers_high))


if __name__ == "__main__":
    interpret_search("parameter_data.csv")
