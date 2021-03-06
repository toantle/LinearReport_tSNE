from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox

def plot_embedding_MNIST(X, y, digits, title=None, axes=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    ax = plt.subplot(111) if axes==None else axes
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    shown_images = np.array([[1., 1.]])  # just something big
    for i in range(X.shape[0]):
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 4e-3:
            # don't show points that are too close
            continue
        shown_images = np.r_[shown_images, [X[i]]]
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
            X[i])
        ax.add_artist(imagebox)
        
    plt.xticks([]), plt.yticks([])
    if title is not None:
        ax.set_title(title)

def plot_embedding_W2V(X, y, title=None, axes=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    ax = plt.subplot(111) if axes==None else axes
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1], str(y[i]), fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        ax.set_title(title)