import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec


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


def plot_rows_of_images(images, savepath):
    num_types_of_imgs = len(images)
    fig = plt.figure(figsize=(images[0].shape[0], num_types_of_imgs))
    gs = gridspec.GridSpec(num_types_of_imgs, images[0].shape[0])
    for i, type_of_img in enumerate(images):
        for j in range(type_of_img.shape[0]):
            new_ax = plt.subplot(gs[i, j])
            plot_single_img(np.reshape(type_of_img[j], (1, -1)), ax=new_ax)
    plt.savefig(savepath)
    plt.close('all')
