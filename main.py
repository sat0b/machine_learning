from load.mnist import get_mnist
from model.gan import DCGan

(x_train, y_train), (x_test, y_test) = get_mnist()

model = DCGan()
model.train(x_train, 10, 64)
model.save("output/gan")
