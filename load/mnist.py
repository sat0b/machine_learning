from tensorflow.python.keras.datasets import mnist


def get_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # reshape N x 28 x 28 x 1
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    # normalize -1 to 1
    x_train = (x_train - 127.5) / 127.5
    x_test = (x_test - 127.5) / 127.5

    return (x_train, y_train), (x_test, y_test)
