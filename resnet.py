from utils.load import get_cifar10
from resnet import train, model

(x_train, y_train), (x_test, y_test) = get_cifar10()

model = model.ResNet34(input_shape=(32, 32, 3), output_shape=10)
train.run(model,
          x_train[:1000],
          y_train[:1000],
          validation_data=(x_test, y_test),
          epochs=3,
          batch_size=10)
