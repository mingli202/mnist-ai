import numpy as np


class Model:
    def __init__(self):
        self.hidden_layers = [Layer(28 * 28, 15), Layer(15, 15), Layer(15, 10)]
        self.output = np.array([])

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        for x, y in zip(x_train, y_train):
            cost = self.forward_propagation(x, y)

            for node in self.hidden_layers[2].vec:
                print(node.bias)

            self.back_propagation(cost)
            print("after")

            for node in self.hidden_layers[2].vec:
                print(node.bias)

            return

    def forward_propagation(self, x: np.ndarray, y: int):
        # make all the values a percentage of brightness
        # to avoid having astronomically big values
        x = np.array([i / 255 for i in x.flatten()])

        for i in range(len(self.hidden_layers)):
            # apply activation function and weights/biases
            x = np.array([self.fn(node, x) for node in self.hidden_layers[i].vec])
            self.hidden_layers[i].values = x

        ideal = np.zeros(10)
        ideal[y] = 1

        self.output = self.softmax(x)
        cost = np.array([(a - b) for a, b in zip(ideal, self.output)])

        return cost

    # relu
    def fn(self, node, x):
        val = sum(np.multiply(node.weight, x)) + node.bias
        if val < 0:
            val = 0

        return val

    def softmax(self, vec: np.ndarray):
        s = sum(np.exp(vec[i]) for i in range(vec.size))
        return np.array([np.exp(vec[i]) / s for i in range(vec.size)])

    def back_propagation(self, cost):
        for _i in range(len(self.hidden_layers)):
            i = len(self.hidden_layers) - _i - 1

            layer = self.hidden_layers[i]

            new_cost = []

            for k in range(len(layer.vec)):
                node = layer.vec[k]
                d_b = 0
                d_a_l_1 = 0

                print("cost", len(cost))
                for w in range(len(cost)):
                    c = cost[w]
                    weight = node.weight[w]

                    d_w = 2 * c * layer.values[k]
                    d_b += 2 * c
                    d_a_l_1 += 2 * c * weight

                    self.hidden_layers[i].vec[k].weight[w] -= d_w

                self.hidden_layers[i].vec[k].bias -= d_b
                new_cost.append(d_a_l_1)

            cost = new_cost


class Layer:
    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        self.vec = np.array([Node(in_size) for _ in range(out_size)])
        self.values = np.array([])


class Node:
    def __init__(self, size):
        rng = np.random.default_rng()
        self.weight = np.array([rng.random() - 0.5 for _ in range(size)])
        self.bias = 0
