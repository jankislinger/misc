import random

import numpy as np

from expressions.classes import Pow, Variable, Sum


def regression(xs, ys):
    err_sqr = [Pow(Variable('a') + Variable('b') * x - y, 2) for x, y in zip(xs, ys)]
    loss = Sum(err_sqr) / len(xs)

    _, beta = loss.minimize({'a': 0, 'b': 0}, 0.1)
    return beta['a'], beta['b']


if __name__ == '__main__':
    a, b = 1, 3
    x_train = [random.random() for _ in range(1000)]
    y_train = [a + b * x + random.random() for x in x_train]
    print(np.polyfit(x_train, y_train, deg=1))
    # ys = [a + b * x for x in xs]
    a_hat, b_hat = regression(x_train, y_train)
    print(f"y = {round(a_hat, 3)} + {round(b_hat, 3)} * x")
