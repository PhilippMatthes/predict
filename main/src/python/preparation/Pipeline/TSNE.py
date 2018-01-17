from sklearn.manifold import TSNE as SKTSNE
import numpy as np
from random import randint
import matplotlib.cm as cm

from sklearn.datasets import fetch_mldata


class TSNE:
    def __init__(self, X):
        self.tsne = SKTSNE(n_components=3)
        print("Fitting t-SNE.")
        self.tsne.fit(X)

    def reduce(self, X):
        print("Reducing dimensionality of X")
        return self.tsne.fit_transform(X)


if __name__ == "__main__":
    print("Fetching MNIST Data")
    mnist = fetch_mldata('MNIST original')
    X = mnist["data"]
    y = mnist["target"]
    X_array = np.nan_to_num(np.asarray(X, dtype=np.double))

    visualizer = TSNE(X_array)
    X_visualized = visualizer.reduce(X_array)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = cm.rainbow(np.linspace(0, 1, len(X_visualized)))
    for t, c in zip(X_visualized, colors):
        plt.scatter(t[0], t[1], t[2], color=c, marker="o")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.savefig("t-SNE.png")
    plt.show()
