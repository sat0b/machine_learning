from tensorflow.python.keras.datasets import mnist, cifar10
from tensorflow.python.keras.utils.np_utils import to_categorical


def get_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # reshape N x 28 x 28 x 1
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    # normalize -1 to 1
    x_train = (x_train - 127.5) / 127.5
    x_test = (x_test - 127.5) / 127.5

    return (x_train, y_train), (x_test, y_test)


def get_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = (x_train - 127.5) / 127.5
    x_test = (x_test - 127.5) / 127.5

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return (x_train, y_train), (x_test, y_test)
