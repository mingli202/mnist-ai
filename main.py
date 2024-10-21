import json
import os

import keras
import numpy as np
from matplotlib import pyplot as plt

import notif
import screen
from model import Model

f = "./json/params7.json"


def main():
    if os.path.exists(f):
        draw()
    else:
        train()


def train():
    dataset: tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] = (
        keras.datasets.mnist.load_data()
    )

    x_train: np.ndarray = dataset[0][0]
    y_train: np.ndarray = dataset[0][1]
    x_test: np.ndarray = dataset[1][0]
    y_test: np.ndarray = dataset[1][1]

    m = Model()

    iteration = 200

    def epoch(i):
        m.train(x_train, y_train)
        a = m.test(x_test, y_test)

        print(f"{i} {a}")

        return a

    accuracies = [epoch(i) for i in range(iteration)]

    with open(f, "w") as file:
        d = {
            "param": {
                "alpha": m.alpha,
                "partition_size": m.n,
                "accuracy": accuracies[-1],
            },
            "weight": [m.w1.tolist(), m.w2.tolist(), m.w3.tolist()],
            "bias": [m.b1.tolist(), m.b2.tolist(), m.b3.tolist()],
        }
        file.write(json.dumps(d, indent=2))

    notif.send(
        f"Done!\nalpha: {m.alpha}\npartition_size: {m.n}\naccuracy: {accuracies[-1]}\nfile: {f}"
    )

    plt.plot(accuracies)
    plt.show()


def draw():
    with open(f, "r") as file:
        data = json.loads(file.read())

    w1, w2, w3 = data["weight"]
    b1, b2, b3 = data["bias"]

    m = Model(w1, w2, w3, b1, b2, b3)

    screen.main(m)


if __name__ == "__main__":
    main()
