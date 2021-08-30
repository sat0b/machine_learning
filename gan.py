from utils.load import get_mnist
from models import gan

(x_train, y_train), (x_test, y_test) = get_mnist()

model = gan.DCGan()
model.train(x_train, 10, 64)
model.save("output/gan")

gan.show_example(model, 10)
