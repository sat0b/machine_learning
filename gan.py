from utils.load import get_mnist
from gan import model

(x_train, y_train), (x_test, y_test) = get_mnist()

gan_model = model.DCGan()
gan_model.train(x_train, 10, 64)
gan_model.save("output/gan")

model.show_example(gan_model, 10)
