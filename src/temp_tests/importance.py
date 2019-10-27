from src.model.regression.linear_model import LinearModel
from src.optimization.linear import LinearSGD
from src.functions.loss import LogCosh
from src.utils.data_manipulation import split
from src.preconditioning.normalization import MinMaxNormalizer
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm # This allows different color schemes
import os

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


if __name__ == "__main__":
    path = os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0]
    data = np.load(file=path + '\\resources\\' + 'train.npy')

    loss = LogCosh()
    model = LinearModel((3, 1))
    kwargs = {'batch_size': 25, 'loss': loss, 'lr': 10**-1, 'epochs': 1000, 'epoch_step': (100, .75)}
    optimizer = LinearSGD()
    n_models = 1

    for i in range(32, data.shape[1]):
        for j in range(i+1, data.shape[1]):
            if i in [8, 19, 28] or j in [8, 19, 28]:
                continue
            min_error, best = np.inf, None
            for it in range(n_models):
                model.set_param(np.random.uniform(-1, 1, (3, 1)))
                train, test = split(MinMaxNormalizer()(data[np.logical_and(data[:, i] != -999, data[:, j] != -999)]))

                x = np.hstack([train[:, [i, j]], np.ones((train.shape[0], 1))])
                y = np.reshape(train[:, 1], (-1, 1))
                input = np.hstack([test[:, [i, j]], np.ones((test.shape[0], 1))])

                optimizer(model, x, y, **kwargs)
                error = loss(model(input),
                             np.reshape(test[:, 2], (-1, 1)))
                if error < min_error:
                    min_error = error
                    best = model.get_params()

            np.save(arr=best, file='./saved/' + str(i) + 'and' + str(j) + '_' + str("%0.2f" % min_error) + '.npy')

    onlyfiles = [f for f in os.listdir('./saved/') if os.path.isfile(os.path.join('./saved/', f))]
    n_features = data.shape[1]-2
    error_matrix = np.ones((n_features, n_features))*0
    for f in onlyfiles:
        split_str = f.split('and')
        first = int(split_str[0])-2
        split_str = split_str[1].split('_')
        second = int(split_str[0])-2
        error = float(split_str[1].split('.npy')[0])
        error_matrix[first, second] = error
    fig, ax = plt.subplots()

    im, _ = heatmap(error_matrix, range(n_features), range(n_features),
                    cmap="PuOr", vmin=0, vmax=10000,
                    cbarlabel="Error")

    plt.legend(loc='upper left')
    plt.show()
    pass