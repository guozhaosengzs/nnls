"""
    Title:          NNLS.py
    Author:         Zhaosen Guo
    Date:           2020-01-20
    Description:    Non-negative Least Squares

    Given a data set of two dimensional matrix, use gradient descent to find the
    non-negative least squares for a regression line.
"""

import matplotlib.pyplot as plt
import numpy as np


def grad_dec(A, y, iters, learn_rate):
    """ Given A as independent variables, y as dependent variable, initiate from a random vector b (between 0 - 1)and run 
        for set iterations and learning rate. """
    
    dim = A.shape[1]
    b = np.random.random(dim)

    min_func = []
    grad =[]

    for t in range(iters):
        g = (A.T @ A) @ b - A.T @ y
        g /= np.linalg.norm(g)

        b = b - learn_rate * g

        min_func.append(0.5 * np.linalg.norm(A @ b - y) ** 2)
        grad.append(np.linalg.norm(g))

    return {'b': b, 'min_func' : min_func, 'grad' : grad}

def main():
    """ Load data, prepare for gd operation and output result as graphs. """
    data = np.loadtxt('data.txt', dtype = np.float64)
    x = data[:,0]
    y = data[:,1]

    A = np.ones((len(x), 2))
    A[:,1] = x

    result = grad_dec(A, y, 100, 0.05)

    plot(x, y, result['b'], result['min_func'])

def plot(x, y, b, min_func):
    """ Pot the fitted result and the cost function graph."""

    canv, fig = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 10))

    fig[0].scatter(x, y, marker = 'x', color = "blue", label = "points")

    fig[0].plot(x, (b[0] + b[1] * x), color = 'red', linestyle = '-.', label = 'Fitted Line')

    fig[0].set_xticks(np.linspace(int(min(x)) - 1, int(max(x)) + 1, 5))
    fig[0].set_yticks(np.linspace(int(min(y)) - 1, int(max(y)) + 1, 5))
    fig[0].tick_params(axis = 'both', labelsize = 10)

    fig[0].set_xlabel("x" , fontsize = 12 )
    fig[0].set_ylabel("y" , fontsize = 12 )

    fig[0].set_title("Non-negative Least Squares", fontsize=14 )
    fig[0].legend(loc = 'upper left')
    fig[0].grid(b = True)

    fig[1].plot(min_func, color = 'green', label = 'Cost function curve')
    fig[1].tick_params(axis = 'both', labelsize = 10)
    fig[1].set_xlabel("iternation" , fontsize = 12 )
    fig[1].set_ylabel("residual" , fontsize = 12 )
    fig[1].set_title("Errors during iternations", fontsize=14 )
    fig[1].legend(loc = 'upper right')
    fig[1].grid(b = True)

    canv.savefig("result.png", dpi = 400)

if __name__ == '__main__':
    main()
