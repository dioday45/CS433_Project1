import numpy as np
import src.helpers as hlp
import sys

# *************************************
#      LEAST SQUARE REGRESSION
# *************************************
def mean_squared_error_gd(y, tx, initial_w, max_iters=1000, gamma=0.1):
    """linear regression with gradient descent

    Args:
        y : labels
        tx : features
        initial_w : weights
        max_iters : maximum number of iterations. Defaults to 1000.
        gamma : gradient descent parameter. Defaults to 0.1.

    Returns:
        weights and final loss
    """
    w = initial_w

    for epoch in range(int(max_iters)):
        # Compute MSE gradient
        sys.stdout.write("\r{:.2f}%".format((float(epoch) / max_iters) * 100))
        sys.stdout.flush()
        grad = hlp.MSE_gradient(tx, w, y)

        # Update weight
        w = w - gamma * grad

    # compute final loss
    final_loss = hlp.MSE_loss(tx, w, y)

    return w, final_loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters=10, gamma=0.01):
    """linear regression with stochastich gradient descent

    Args:
        y : labels
        tx : features
        initial_w : weights
        max_iters : maximum number of iterations. Defaults to 1000.
        gamma : gradient descent parameter. Defaults to 0.1.

    Returns:
        weights and final loss
    """
    w = initial_w
    batch_size = 1

    for epoch in range(int(max_iters)):
        for y_b, tx_b in hlp.batch_iter(y, tx, batch_size, num_batches=1):
            # Compute MSE gradient
            grad = hlp.MSE_gradient(tx_b, w, y_b)

            # Update weight
            w = w - gamma * grad

    # Compute final loss
    final_loss = hlp.MSE_loss(tx, w, y)

    return w, final_loss


def least_squares(y, tx):
    """Least square regression

    Args:
        y : labels
        tx : features

    Returns:
        updated weights and final loss
    """
    A = np.dot(tx.T, tx)
    b = np.dot(tx.T, y)
    w = np.linalg.solve(A, b)

    y_hat = np.dot(tx, w)

    loss = hlp.MSE_loss(tx, w, y)

    return w, loss


# *************************************
#       LOGISTIC REGRESSION
# *************************************


def logistic_regression(y, tx, initial_w, max_iters=100, gamma=0.1):
    """logistic regression with gradient descent

    Args:
        y : labels
        tx : features
        initial_w : weights
        max_iters : maximum number of iterations. Defaults to 1000.
        gamma : gradient descent parameter. Defaults to 0.1.

    Returns:
        weights and final loss
    """
    w = initial_w

    for epoch in range(max_iters):

        # Compute gradient
        grad = hlp.log_gradient(tx, w, y)

        # Update weights
        w -= gamma * grad

    y_hat = hlp.sigmoid(tx.dot(w))
    final_loss = hlp.log_loss(y, y_hat)

    return w, final_loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters=100, gamma=0.1):
    """Regularized logistic regression with gradient descent

    Args:
        y : labels
        tx : features
        initial_w : weights
        max_iters : maximum number of iterations. Defaults to 1000.
        gamma : gradient descent parameter. Defaults to 0.1.
    """
    w = initial_w
    for epoch in range(max_iters):
        y_hat = hlp.sigmoid(np.dot(tx, w))
        # Compute gradient
        grad = hlp.log_gradient(tx, w, y, reg=True, lambda_=lambda_)

        # Update weights
        w = w - gamma * grad

    y_hat = hlp.sigmoid(np.dot(tx, w))
    final_loss = hlp.log_loss(y, y_hat)

    return w, final_loss


# *************************************
#         RIDGE REGRESSION
# *************************************


def ridge_regression(y, tx, lambda_):
    """Ridge regression

    Args:
        y : labels
        tx : features
        lambda_ : ridge parameter

    Returns:
        updated weights and final loss
    """
    shape = tx.shape
    lambda_prime = 2 * shape[0] * lambda_
    A = np.dot(tx.T, tx) + lambda_prime * np.identity(shape[1])
    b = np.dot(tx.T, y)
    w = np.linalg.solve(A, b)

    loss = hlp.MSE_loss(tx, w, y)

    return w, loss
