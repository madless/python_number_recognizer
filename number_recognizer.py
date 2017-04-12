import numpy as np

# Init data
X = np.array([
    [[1, 1, 1, 1],  # 0
     [1, 1, 0, 1],
     [1, 1, 0, 1],
     [1, 1, 0, 1],
     [1, 1, 1, 1]],
    [[1, 1, 1, 0],  # 1
     [1, 0, 1, 0],
     [1, 0, 1, 0],
     [1, 0, 1, 0],
     [1, 1, 1, 1]],
    [[1, 1, 1, 1],  # 2
     [1, 0, 0, 1],
     [1, 1, 1, 1],
     [1, 1, 0, 0],
     [1, 1, 1, 1]],
    [[1, 1, 1, 1],  # 3
     [1, 0, 0, 1],
     [1, 1, 1, 1],
     [1, 0, 0, 1],
     [1, 1, 1, 1]],
    [[1, 1, 0, 1],  # 4
     [1, 1, 0, 1],
     [1, 1, 1, 1],
     [1, 0, 0, 1],
     [1, 0, 0, 1]],
    [[1, 1, 1, 1],  # 5
     [1, 1, 0, 0],
     [1, 1, 1, 1],
     [1, 0, 0, 1],
     [1, 1, 1, 1]],
    [[1, 1, 1, 1],  # 6
     [1, 1, 0, 0],
     [1, 1, 1, 1],
     [1, 1, 0, 1],
     [1, 1, 1, 1]],
    [[1, 1, 1, 1],  # 7
     [1, 0, 0, 1],
     [1, 0, 0, 1],
     [1, 0, 0, 1],
     [1, 0, 0, 1]],
    [[1, 1, 1, 1],  # 8
     [1, 1, 0, 1],
     [1, 1, 1, 1],
     [1, 1, 0, 1],
     [1, 1, 1, 1]],
    [[1, 1, 1, 1],  # 9
     [1, 1, 0, 1],
     [1, 1, 1, 1],
     [1, 0, 0, 1],
     [1, 1, 1, 1]],
])

Y = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
])

test_sample = [
    [1, 1, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 1, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 1]
]


# Algorithm
def predict(sample, weights):
    return 1. if np.sum(weights * sample) > 0. else 0.


def recognize_number(sample):
    return [i for i in range(10) if predict(sample, weights[i])]


def train_weights(x, y, l_rate, n_epoch):
    weights = np.zeros((10, 5, 4))
    for epoch in range(n_epoch):
        for i in range(len(x)):
            for j in range(len(weights)):
                error = y[i][j] - predict(x[i], weights[j])
                weights[j] += (x[i] * l_rate * error) if i == j else (x[i] * l_rate ** 2 * error)
    print("Weights:\n", np.around(weights, 1))
    return weights


# Execution
l_rate = 0.1
n_epoch = 1000
weights = train_weights(X, Y, l_rate, n_epoch)
print("Recognized numbers: ", recognize_number(test_sample))
