import numpy as np
from matplotlib import pyplot as plt


class Model:
    def __init__(self):
        # 2 hidden layers, 1 output layer
        self.layers = [Layer(28 * 28, 15), Layer(15, 15), Layer(15, 10)]
        self.output = np.array([])
        self.alpha = 0.5
        self.costs = []

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        i = 1

        print(x_train.shape)
        return

        for x, y in zip(x_train, y_train):
            # print(i, end=": ")
            cost = self.forward_propagation(x, y)
            self.back_propagation(x, cost)

            if i == -1:
                plt.plot(
                    [
                        sum(self.costs[0 : i + 1]) / (i + 1)
                        for i in range(len(self.costs))
                    ],
                )
                plt.ylim(bottom=0)
                plt.show()
                return

            i += 1

    def forward_propagation(self, x: np.ndarray, y: int):
        # make all the values a percentage of brightness
        # to avoid having astronomically big values
        x = x.reshape(28 * 28, 1) / 255

        for i in range(len(self.layers)):
            # assert x.shape == (self.layers[i].in_size, 1)
            # apply activation function and weights/biases
            x = np.matmul(self.layers[i].weights, x) + self.layers[i].biases

            if i == len(self.layers) - 1:
                x = self.softmax(x)
            else:
                x = self.fn(x)

            self.layers[i].values = x

        ideal = np.zeros((10, 1))
        ideal[y] = 1

        # assert ideal.shape == self.layers[-1].values.shape

        cost = self.layers[-1].values - ideal
        n_cost = np.sum((self.layers[-1].values - ideal) ** 2)
        self.costs.append(n_cost)

        return cost

    # relu
    def fn(self, x: np.ndarray):
        # assert len(x.shape) == 2
        return np.maximum(x, 0)

    def softmax(self, vec: np.ndarray):
        e = np.exp(vec)
        return e / np.sum(e)

    def back_propagation(self, x: np.ndarray, cost: np.ndarray):
        # assert len(x.shape) == 2
        # assert len(cost.shape) == 2

        values = [x.reshape(28 * 28, 1) / 255] + [
            np.atleast_2d(layer.values) for layer in self.layers
        ]

        # assert all(v.shape[1] == 1 for v in values)

        for i in range(len(self.layers) - 1, 0, -1):
            d_W: np.ndarray = np.matmul(cost, values[i].T) / cost.size

            d_B: np.ndarray = np.full((cost.size, 1), np.sum(cost)) / cost.size

            d_A: np.ndarray = np.matmul(self.layers[i].weights.T, cost)
            d_Fn: np.ndarray = values[i] > 0

            # assert d_A.shape == d_Fn.shape
            # assert d_W.shape == self.layers[i].weights.shape
            # assert d_B.shape == self.layers[i].biases.shape

            cost = d_A * d_Fn
            self.layers[i].weights = self.layers[i].weights - self.alpha * d_W
            self.layers[i].biases = self.layers[i].biases - self.alpha * d_B

    def test(self, x_test, y_test):
        i = 1
        failed = 0
        for x, y in zip(x_test, y_test):
            # print(i, end=": ")
            if self.predict(x, y) != y:
                failed += 1

            i += 1

        print("accuracy:", 1 - failed / i)

    def predict(self, x: np.ndarray, y: int):
        self.forward_propagation(x, y)

        return np.argmax(self.layers[-1].values)


class Layer:
    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        self.values = np.zeros((out_size, 1))
        self.biases = np.zeros((out_size, 1))
        self.weights = np.random.rand(out_size, in_size) - 0.5
        print(self.weights)
