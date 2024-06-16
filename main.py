import keras
import numpy as np
from model import Model


def main():
    dataset: tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] = (
        keras.datasets.mnist.load_data()
    )

    x_train: np.ndarray = dataset[0][0]
    y_train: np.ndarray = dataset[0][1]
    x_test: np.ndarray = dataset[1][0]
    y_test: np.ndarray = dataset[1][1]

    m = Model()
    m.train(x_train, y_train)


if __name__ == "__main__":
    main()
