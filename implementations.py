import numpy as np

# Helper functions

def compute_gradient(y, tx, w):
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(y)
    return grad

def compute_mse_loss(y, tx, w):
    err = y - tx.dot(w)
    loss = np.sum(err**2) / (2 * len(y))
    return loss

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def logistic_loss(y, tx, w):
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(-loss)

def logistic_gradient(y, tx, w):
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad

# Main functions

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for _ in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient
    loss = compute_mse_loss(y, tx, w)
    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    n = len(y)
    for _ in range(max_iters):
        i = np.random.randint(0, n)
        gradient = compute_gradient(y[i], tx[i], w)
        w = w - gamma * gradient
    loss = compute_mse_loss(y, tx, w)
    return w, loss

def least_squares(y, tx):
    A = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(A, b)
    loss = compute_mse_loss(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    A = tx.T.dot(tx) + lambda_ * np.eye(tx.shape[1])
    b = tx.T.dot(y)
    w = np.linalg.solve(A, b)
    loss = compute_mse_loss(y, tx, w)  # Loss without the penalty term
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for _ in range(max_iters):
        gradient = logistic_gradient(y, tx, w)
        w = w - gamma * gradient
    loss = logistic_loss(y, tx, w)
    return w, loss

def reg_logistic_gradient(y, tx, w, lambda_):
    grad = logistic_gradient(y, tx, w)
    reg_grad = grad + 2 * lambda_ * w
    return reg_grad

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    for _ in range(max_iters):
        gradient = reg_logistic_gradient(y, tx, w, lambda_)
        w = w - gamma * gradient
    loss = logistic_loss(y, tx, w)  # Loss without the penalty term
    return w, loss
