import numpy as np
import matplotlib.pyplot as plt


def mse(a, b):
    diffs = np.square(np.subtract(a, b))
    total_diff = np.sum(diffs)
    return np.divide(total_diff, (a.shape[0] * a.shape[1]))


def is_unique_by_mse(new_tile, prev_tiles):
    for old_tile in prev_tiles:
        err = mse(new_tile, old_tile)
        if err < 0.00001:
            return False

    return True


def unique_concat(previous, new):
    out = previous.copy()
    dupes = 0
    for potential in new:
        is_new = True
        for seen in previous:
            if np.array_equal(potential, seen):
                # print('ARR EQUAL')
                is_new = False
                dupes += 1
                break
        if is_new:
            # print('add new')
            # print(type(out), len(out))
            out.append(potential)
            # print(type(out), len(out))
    return out, dupes


def myint(torchscalar):
    return int(torchscalar.item())


def show_images(images, cols=1, titles=None):
    """
    src: https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
    Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure(figsize=(10, 10))
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        x = plt.imshow(image)
        # a.set_title(title)
        x.axes.get_xaxis().set_visible(False)
        x.axes.get_yaxis().set_visible(False)
    # fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.axis('off')
    # plt.show()
    return fig
