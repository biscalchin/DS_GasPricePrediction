import matplotlib.pyplot as plt
import numpy as np


def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Draw a neural network cartoon using matplotlib.

    :param ax: matplotlib Axes object
    :param left: float, the center of the leftmost node(s) will be placed here
    :param right: float, the center of the rightmost node(s) will be placed here
    :param bottom: float, the center of the bottommost node(s) will be placed here
    :param top: float, the center of the topmost node(s) will be placed here
    :param layer_sizes: list of int, list containing the number of nodes in each layer
    '''
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)

    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size):
            circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), v_spacing / 4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
            # Annotation for the first layer
            if n == 0:
                plt.annotate(f'Input {m + 1}', xy=(n * h_spacing + left, layer_top - m * v_spacing), xytext=(-30, 0),
                             textcoords='offset points', ha='center', va='center',
                             arrowprops=dict(arrowstyle='->', lw=0.5))
            # Annotation for the output layer
            elif n == len(layer_sizes) - 1:
                plt.annotate('Output', xy=(n * h_spacing + left, layer_top - m * v_spacing), xytext=(30, 0),
                             textcoords='offset points', ha='center', va='center',
                             arrowprops=dict(arrowstyle='->', lw=0.5))
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                  [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c='k')
                ax.add_artist(line)


def draw_updated_neural_net(ax, left, right, bottom, top, input_size, hidden_sizes, output_size):
    '''
    Draw an updated neural network cartoon using matplotlib.

    :param ax: matplotlib Axes object
    :param left: float, the center of the leftmost node(s) will be placed here
    :param right: float, the center of the rightmost node(s) will be placed here
    :param bottom: float, the center of the bottommost node(s) will be placed here
    :param top: float, the center of the topmost node(s) will be placed here
    :param input_size: int, number of nodes in the input layer
    :param hidden_sizes: list of int, list containing the number of nodes in each hidden layer
    :param output_size: int, number of nodes in the output layer
    '''
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)

    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size):
            circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), v_spacing / 4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
            if n == len(layer_sizes) - 1:
                plt.annotate('Output', xy=(n * h_spacing + left, layer_top - m * v_spacing), xytext=(30, 0),
                             textcoords='offset points', ha='center', va='center',
                             arrowprops=dict(arrowstyle='->', lw=0.5))

    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                  [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c='k')
                ax.add_artist(line)


def designer():
    # Parameters for the neural network
    input_size = 1  # Assuming 3 input features
    hidden_sizes = [20, 10, 5]  # Sizes of hidden layers
    output_size = 1  # Output size for regression

    # Create the figure with updated input size
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()
    ax.axis('off')
    draw_updated_neural_net(ax, .1, .9, .1, .9, input_size, hidden_sizes, output_size)
    plt.show()


if __name__ == '__main__':
    designer()
