import matplotlib.pyplot as plt

def draw_neural_net(ax, left, right, bottom, top, layer_sizes, labels, param_counts):
    '''
    Draw a neural network cartoon using matplotlib.
    
    :param ax: matplotlib.axes.Axes, the axes on which to plot the cartoon (get e.g. by plt.gca())
    :param left: float, the center of the leftmost node(s) will be placed here
    :param right: float, the center of the rightmost node(s) will be placed here
    :param bottom: float, the center of the bottommost node(s) will be placed here
    :param top: float, the center of the topmost node(s) will be placed here
    :param layer_sizes: list of int, list containing the number of nodes in each layer
    :param labels: list of str, labels for the hidden layers
    :param param_counts: list of int, parameter counts for each layer
    '''
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)
    
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size):
            circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), v_spacing / 8.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
            if n == 0:
                if m == 4:
                    ax.text(left - h_spacing / 2., layer_top - m * v_spacing, 'Input 784', fontsize=10, ha='center', va='center')
                elif m == 3:
                    ax.text(left - h_spacing / 2., layer_top - m * v_spacing, '...', fontsize=10, ha='center', va='center')
                else:
                    ax.text(left - h_spacing / 2., layer_top - m * v_spacing, f'Input {m + 1}', fontsize=10, ha='center', va='center')
            elif n == len(layer_sizes) - 1:
                ax.text(right + h_spacing / 2., layer_top - m * v_spacing, f'Output {m + 1}', fontsize=10, ha='center', va='center')
            elif n == 1 or n == 2:  # Hidden layers
                if m == 4:
                    ax.text(n * h_spacing + left, layer_top - m * v_spacing - v_spacing / 2, '128', fontsize=10, ha='center', va='center')
        
        # Add labels for input, hidden, and output layers
        if n == 0:
            ax.text(n * h_spacing + left, layer_top + v_spacing * 1.5, 'Input Layer', fontsize=12, ha='center', va='center', color='blue')
            ax.text(n * h_spacing + left, layer_top + v_spacing, f'Neurons: 784', fontsize=10, ha='center', va='center')
        elif n == len(layer_sizes) - 1:
            ax.text(n * h_spacing + left, layer_top + v_spacing * 1.5, 'Output Layer', fontsize=12, ha='center', va='center', color='blue')
            ax.text(n * h_spacing + left, layer_top + v_spacing, f'Neurons: {layer_size}', fontsize=10, ha='center', va='center')
            ax.text(n * h_spacing + left, layer_top + v_spacing / 2, f'Params: {param_counts[n-1]}', fontsize=10, ha='center', va='center')
        elif 0 < n < len(layer_sizes) - 1:
            ax.text(n * h_spacing + left, layer_top + v_spacing * 1.5, labels[n-1], fontsize=12, ha='center', va='center', color='blue')
            ax.text(n * h_spacing + left, layer_top + v_spacing, f'Neurons: 128', fontsize=10, ha='center', va='center')
            if n == 2:  # Adjust Hidden Layer 2
                ax.text(n * h_spacing + left, layer_top + v_spacing / 1.8, f'Params: {param_counts[n-1]}', fontsize=10, ha='center', va='center')
            else:
                ax.text(n * h_spacing + left, layer_top + v_spacing / 2, f'Params: {param_counts[n-1]}', fontsize=10, ha='center', va='center')
    
    # Add "..." before the 5th input node
    ax.text(left - h_spacing / 2., (layer_top - 3 * v_spacing + layer_top - 4 * v_spacing) / 2, '...', fontsize=10, ha='center', va='center')

    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                  [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c='k')
                ax.add_artist(line)

fig = plt.figure(figsize=(12, 8))
ax = fig.gca()
ax.axis('off')
draw_neural_net(ax, .1, .9, .1, .9, [5, 5, 5, 10], ["Hidden Layer 1", "Hidden Layer 2"], [100480, 16512, 1290])
plt.show()
