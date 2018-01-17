from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import numpy as np

from main.src.python.Config import pca_reduced_dimensions
from main.src.python.Config import pca_grid_search_space
from main.src.python.Config import pca_cross_validation_splitting_strategy
from main.src.python.Config import pca_verbose

from sklearn.datasets import fetch_mldata


class PCA:
    def __init__(self, X, y):
        print("Performing grid search to select the best Kernel PCA parameters")
        clf = Pipeline([
            ("kpca", KernelPCA(n_components=pca_reduced_dimensions)),
            ("log_reg", LogisticRegression(verbose=pca_verbose)),
        ])

        param_grid = [{
            "kpca__gamma": pca_grid_search_space,
            "kpca__kernel": ["rbf", "sigmoid"]
        }]

        grid_search = GridSearchCV(clf,
                                   param_grid=param_grid,
                                   cv=pca_cross_validation_splitting_strategy,
                                   verbose=pca_verbose)

        grid_search.fit(X, y)

        print("Found best parameters:", grid_search.best_params_)
        self.best_estimator = grid_search.best_estimator_

    def reduce(self, X):
        print("Reducing dimensionality of X")
        return self.best_estimator.fit_transform(X)

    def reproduce(self, X_reduced):
        print("Reproducting X")
        return self.best_estimator.inverse_transform(X_reduced)


if __name__ == "__main__":
    print("Fetching MNIST Data")
    mnist = fetch_mldata('MNIST original')
    X, y = mnist["data"], mnist["target"]
    pca = PCA(X, y)
    X_reduced = pca.reduce(X)
    X_reproduced = pca.reproduce(X_reduced)

    import matplotlib
    import matplotlib.pyplot as plt

    some_digit = X_reproduced[36000]
    some_digit_image = some_digit.reshape(28, 28)

    plt.imshow(some_digit_image, interpolation="nearest")

    print("Showing reconstructed image")
    plt.axis("off")
    plt.show()
