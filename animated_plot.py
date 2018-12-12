import matplotlib.pyplot as plt
import numpy as np


def show_animated(images,titles=None, pause=1e-7):
    try:
        for image, axes_image in zip(images, show_animated.axis_images):
            axes_image.set_array(image)
        plt.show()
        plt.pause(pause)
    except AttributeError:
        n_ims = len(images)
        if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
        plt.ion()
        show_animated.fig = plt.figure()
        show_animated.axis_images = []
        n = 1
        for image,title in zip(images,titles):
            ax = show_animated.fig.add_subplot(1,n_ims,n)
            if image.ndim == 2:
                plt.gray()
            im = plt.imshow(image, animated=True)
            show_animated.axis_images.append(im)
            ax.set_title(title)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            n += 1
        show_animated.fig.set_size_inches(np.array(show_animated.fig.get_size_inches()) * n_ims)
        plt.show()
