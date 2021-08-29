from uitls.load import get_cifar10
from model import resnet

(x_train, y_train), (x_test, y_test) = get_cifar10()

model = resnet.ResNet34(input_shape=(32, 32, 3), output_shape=10)
model.train(x_train, y_train, epochs=100, batch_size=64)
