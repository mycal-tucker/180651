import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from sklearn.decomposition import PCA


# Save the models and images of reconstructions, predictions, and prototypes.
def plot_single_img(img, ax=None, savepath=None):
    side_length = int(np.sqrt(img.shape[1]))
    assert side_length * side_length == img.shape[1]  # Make sure didn't truncate anything.
    new_base_fig = ax is None
    if new_base_fig:
        fig, ax = plt.subplots()
    figure = img.reshape(side_length, side_length)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.imshow(figure, cmap='Greys_r')
    if savepath is not None:
        plt.savefig(savepath)
        return
    if new_base_fig:
        plt.show()


def plot_rows_of_images(images, savepath, show=True):
    num_types_of_imgs = len(images)
    fig = plt.figure(figsize=(images[0].shape[0], num_types_of_imgs))
    gs = gridspec.GridSpec(num_types_of_imgs, images[0].shape[0])
    for i, type_of_img in enumerate(images):
        for j in range(type_of_img.shape[0]):
            new_ax = plt.subplot(gs[i, j])
            plot_single_img(np.reshape(type_of_img[j], (1, -1)), ax=new_ax)
    plt.savefig(savepath)
    if show:
        plt.show()
    plt.close('all')


def plot_multiple_runs(x_data, y_data, y_stdev, labels, x_axis, y_axis, window_size=500, top=11, bottom=0):
    assert len(x_data) == len(y_data) == len(labels)
    if y_stdev is None:
        y_stdev = [[0 for _ in y_data[i]] for i in range(len(y_data))]
    # Smooth stuff out
    plot_y = []
    plot_y_err = []
    plot_x = []
    for i in range(len(x_data)):
        for j in range(len(x_data[i]) - window_size):
            val = np.mean(y_data[i][j: j + window_size])
            y_data[i][j] = val
        # There's a tiny bit of trickiness here to include the last element in the window.
        plot_y.append(y_data[i][:-window_size])
        plot_y[-1].append(y_data[i][-window_size])
        plot_y_err.append(y_stdev[i][:-window_size])
        plot_y_err[-1].append(y_stdev[i][-window_size])
        plot_x.append(x_data[i][:-window_size])
        plot_x[-1].append(x_data[i][-window_size])
    fig, ax = plt.subplots()
    for run_idx in range(len(x_data)):
        markers, caps, bars = ax.errorbar(plot_x[run_idx], plot_y[run_idx], yerr=plot_y_err[run_idx], label=labels[run_idx])
        [bar.set_alpha(0.4) for bar in bars]  # Tune this alpha param depending on how opaque you want the error bars.
    plt.legend([str(label) for label in labels])
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.ylim(top=top, bottom=bottom)
    plt.show()


def plot_bar_chart(x_data, y_data, y_stdev, labels, x_axis, y_axis, top=1, bottom=0):
    assert len(x_data) == len(y_data) == len(labels)
    if y_stdev is None:
        y_stdev = [0 for _ in y_data]
    fig, ax = plt.subplots()
    width_increment = 1.0 / (len(x_data) + 1)
    for run_idx in range(len(x_data)):
        ax.bar(np.asarray(x_data[run_idx]) + width_increment * (run_idx - (len(x_data) - 1) / 2),
               y_data[run_idx],
               yerr=y_stdev[run_idx],
               width=width_increment)
    ax.legend([str(label) for label in labels])
    ax.set_xticks(x_data[0])
    if len(x_data[0]) == 3:
        # Train, Prune, Tune case
        ax.set_xticklabels(['Trained', 'Pruned', 'Tuned'])
    elif len(x_data[0]) == 10:  # Sigma case
        ax.set_xticklabels(['s' + str(i) for i in range(10)])
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    plt.ylim(top=top, bottom=bottom)
    plt.show()


def plot_encodings(encodings, coloring_labels=None, num_to_plot=500, ax=None, coloring_name='digit'):
    enc = encodings[-num_to_plot:]
    plot_in_color = coloring_labels is not None
    array_version = np.asarray(enc)
    pca = PCA(n_components=2)
    pca.fit(array_version)
    transformed = pca.transform(array_version)
    x = transformed[:, 0]
    y = transformed[:, 1]
    new_base_fig = ax is None
    if new_base_fig:
        fig, ax = plt.subplots()
    if plot_in_color:
        colors = coloring_labels[-num_to_plot:]
        num_labels = np.max(colors) - np.min(colors)
        color_map_name = 'coolwarm' if num_labels == 1 else 'RdBu'
        cmap = plt.get_cmap(color_map_name, num_labels + 1)
        pcm = ax.scatter(x, y, s=20, marker='o', c=colors, cmap=cmap, vmin=np.min(colors) - 0.5, vmax=np.max(colors) + 0.5)
        if new_base_fig:
            min_tick = 0
            max_tick = 10 if np.max(colors) > 2 else 2
            fig.colorbar(pcm, ax=ax, ticks=np.arange(min_tick, max_tick))
    else:
        pcm = ax.scatter(x, y, s=20, marker='o', c='gray')
    if new_base_fig:
        ax.set_title('Encodings colored by ' + coloring_name)
        plt.show()
    return pca, pcm


def plot_latent_prompt(encoding, labels=None, test_encodings=None, block=True, ax=None, classes=None, savepath=None, show=True):
    if ax is None:
        fig, (ax) = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_title("PCA of Encodings")
    if encoding is not None:
        pca, pcm = plot_encodings(encoding, labels, ax=ax, coloring_name='class')
    if test_encodings is not None:
        for enc in test_encodings:
            plot_gray_encoding(np.reshape(enc, (1, -1)), ax, pca)
    if labels is not None:
        cbar = fig.colorbar(pcm, ticks=np.arange(0, 10), ax=ax)  # Assumes that ax is None and have new figure.
        if classes is None:
            classes = [i for i in range(10)]  # Assume digits
        cbar.ax.set_yticklabels(classes)
    if block:
        if savepath is not None:
            plt.savefig(savepath)
        if show:
            plt.show()
    else:
        plt.draw()
        plt.pause(0.001)
        if savepath is not None:
            plt.savefig(savepath)


def plot_gray_encoding(encoding, ax, pca):
    array_version = np.asarray(encoding)
    transformed = pca.transform(array_version)
    x = transformed[:, 0]
    y = transformed[:, 1]
    new_base_fig = ax is None
    if new_base_fig:
        fig, ax = plt.subplots()
    ax.scatter(x, y, s=100, marker='x', c='black')
    ax.set(xlabel="Latent PCA 0")
    ax.set(ylabel="Latent PCA 1")
