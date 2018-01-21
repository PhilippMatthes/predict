from sklearn.decomposition import IncrementalPCA as SKIncrementalPCA
from sklearn.decomposition import PCA as SKPCA
import numpy as np
from random import randint

from sklearn.datasets import fetch_mldata


class PCA:
    def __init__(self, X, n_components):
        self.pca = SKIncrementalPCA(n_components=n_components)
        print("Fitting incremental PCA with n_components of {}".format(n_components))
        self.pca.fit(X)


    @staticmethod
    def dimension_needed(X, variance=0.95):
        normal_pca = SKPCA(n_components=variance, random_state=42)
        print("Fitting PCA to perform dimension check")
        normal_pca.fit(X)
        n_components = len(normal_pca.components_)
        print("To hold a variance of {}, the PCA must use a dimension of {}".format(variance, n_components))
        return n_components

    def refit(self, X):
        print("Refitting incremental PCA.")
        self.pca.partial_fit(X)

    def reduce(self, X):
        X_r = self.pca.transform(X)
        print("Reducing X of Shape {} to Shape {}".format(X.shape, X_r.shape))
        return X_r

    def reproduce(self, X_reduced):
        print("Reproducing X")
        return self.pca.inverse_transform(X_reduced)


if __name__ == "__main__":
    print("Fetching MNIST Data")
    mnist = fetch_mldata('MNIST original')
    X = mnist["data"]
    y = mnist["target"]
    X_array = np.nan_to_num(np.asarray(X, dtype=np.double))

    pca = PCA(X_array, 50)
    X_reduced = pca.reduce(X_array)
    X_reproduced = pca.reproduce(X_reduced)

    import matplotlib.pyplot as plt

    random_index1 = randint(0, len(X_reproduced))
    random_index2 = randint(0, len(X_reproduced))

    reconstructed_sample1 = X_reproduced[random_index1].reshape(28, 28)
    reconstructed_sample2 = X_reproduced[random_index2].reshape(28, 28)

    sample1 = X_array[random_index1].reshape(28, 28)
    sample2 = X_array[random_index2].reshape(28, 28)

    f, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(sample1)
    ax[0, 1].imshow(reconstructed_sample1)
    ax[1, 0].imshow(sample2)
    ax[1, 1].imshow(reconstructed_sample2)

    print("Showing reconstructed image")
    plt.show()
