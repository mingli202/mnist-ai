import numpy as np
from numpy.core.multiarray import ndarray


class Model:
    def __init__(self, w1=None, w2=None, w3=None, b1=None, b2=None, b3=None):
        # 2 hidden layers, 1 output layer
        if w1 and w2 and w3 and b1 and b2 and b3:
            self.w1 = np.array(w1)
            self.w2 = np.array(w2)
            self.w3 = np.array(w3)

            self.b1 = np.array(b1)
            self.b2 = np.array(b2)
            self.b3 = np.array(b3)

        else:
            self.w1 = np.random.rand(15, 784) - 0.5
            self.w2 = np.random.rand(15, 15) - 0.5
            self.w3 = np.random.rand(10, 15) - 0.5

            self.b1 = np.zeros((15, 1))
            self.b2 = np.zeros((15, 1))
            self.b3 = np.zeros((10, 1))

        self.alpha = 0.03  # learning rate
        self.n = 20  # batch size

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        i = 1
        dw3 = np.zeros((10, 15))
        db3 = 0
        dw2 = np.zeros((15, 15))
        db2 = 0
        dw1 = np.zeros((15, 784))
        db1 = 0

        for x, y in zip(x_train, y_train):
            # make all the values a percentage of brightness
            # to avoid having astronomically big values
            z0 = x.reshape(784, 1) / 255

            z1, z2, z3 = self.forward_propagation(z0)

            _dw3, _db3, _dw2, _db2, _dw1, _db1 = self.back_propagation(
                z1, z2, z3, z0, y
            )

            if i % self.n == 0:
                self.update_params(
                    dw3 / self.n,
                    db3 / self.n,
                    dw2 / self.n,
                    db2 / self.n,
                    dw1 / self.n,
                    db1 / self.n,
                )
                dw3 = _dw3
                db3 = _db3
                dw2 = _dw2
                db2 = _db2
                dw1 = _dw1
                db1 = _db1

            else:
                dw3 += _dw3
                db3 += _db3
                dw2 += _dw2
                db2 += _db2
                dw1 += _dw1
                db1 += _db1

            i += 1

    def forward_propagation(self, x: np.ndarray):
        z1 = self.relu(self.w1.dot(x) + self.b1)
        z2 = self.relu(self.w2.dot(z1) + self.b2)
        z3 = self.softmax(self.w3.dot(z2) + self.b3)

        # assert x.shape == (784, 1)
        # assert z1.shape == (15, 1)
        # assert z2.shape == (15, 1)
        # assert z3.shape == (10, 1)

        return z1, z2, z3

    # relu
    def relu(self, x: np.ndarray):
        # assert len(x.shape) == 2

        return np.maximum(x, 0)

    def softmax(self, vec: np.ndarray):
        max = np.max(vec)
        e = np.exp(vec - max)
        return e / np.sum(e)

    def back_propagation(
        self, z1: np.ndarray, z2: np.ndarray, z3: np.ndarray, x: ndarray, y: int
    ):
        desired = np.zeros((10, 1))
        desired[y] = 1

        dz3 = z3 - desired
        dw3 = dz3.dot(z2.T) / z3.size
        db3 = np.sum(dz3) / z3.size

        dz2 = self.w3.T.dot(dz3) * (z2 > 0)
        dw2 = dz2.dot(z1.T) / z2.size
        db2 = np.sum(dz2) / z2.size

        dz1 = self.w2.T.dot(dz2) * (z1 > 0)
        dw1 = dz1.dot(x.T) / z1.size
        db1 = np.sum(dz1) / z1.size

        return dw3, db3, dw2, db2, dw1, db1

    def update_params(self, dw3, db3, dw2, db2, dw1, db1):
        self.w3 -= self.alpha * dw3
        self.b3 -= self.alpha * db3
        self.w2 -= self.alpha * dw2
        self.b2 -= self.alpha * db2
        self.w1 -= self.alpha * dw1
        self.b1 -= self.alpha * db1

    def test(self, x_test, y_test):
        accuracy = np.array([self.predict(x) == y for x, y in zip(x_test, y_test)])
        a = np.count_nonzero(accuracy) / accuracy.size
        return a

    def predict(self, x: np.ndarray):
        z0 = x.reshape(784, 1) / 255
        _, _, z = self.forward_propagation(z0)

        return np.argmax(z)
