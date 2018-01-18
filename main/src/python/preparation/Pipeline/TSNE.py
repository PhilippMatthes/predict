from sklearn.manifold import TSNE as SKTSNE
import numpy as np
from random import randint


from sklearn.datasets import fetch_mldata


class TSNE:
    def __init__(self, X):
        self.tsne = SKTSNE(n_components=2, verbose=True)

    def reduce(self, X):
        print("Reducing dimensionality of X")
        return self.tsne.fit_transform(X)


if __name__ == "__main__":
    print("Fetching MNIST Data")
    mnist = fetch_mldata('MNIST original')
    X = mnist["data"][::10]
    y = mnist["target"][::10]
    X_array = np.nan_to_num(np.asarray(X, dtype=np.double))

    visualizer = TSNE(X_array)
    X_visualized = visualizer.reduce(X_array)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches as mpatches

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = {0: "black", 1: "red", 2: "orange", 3: "gold",
              4: "chartreuse", 5: "darkgreen", 6: "lightseagreen",
              7: "royalblue", 8: "darkorchid", 9: "pink"}
    handles = []
    for key in colors.keys():
        patch = mpatches.Patch(color=colors[key], label=str(key))
        handles.append(patch)
    plt.legend(handles=handles)

    for t, l in zip(X_visualized, y):
        ax.scatter(t[0], t[1], marker="o", label=l, color=colors[l])


    plt.savefig("t-SNE.png")
    plt.show()
