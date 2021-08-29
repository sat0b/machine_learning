from tensorflow.keras import layers
from tensorflow.keras import models


class ResNet34:
    def __init__(self, input_shape=(224, 224, 3), output_shape=10, verbose=True):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = self._build()
        self._compile()
        self.verbose = verbose

    def _res_block(self, x, filters=64, down_sampling=False, blocks=1):
        for block in range(blocks):
            if down_sampling and block == 0:
                strides = 2
                shortcut = layers.Conv2D(kernel_size=(1, 1), filters=filters, strides=strides)(x)
                shortcut = layers.BatchNormalization()(shortcut)
            else:
                strides = 1
                shortcut = x
            x = layers.Conv2D(kernel_size=(3, 3), filters=filters, strides=strides, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU()(x)
            x = layers.Conv2D(kernel_size=(3, 3), filters=filters, strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU()(x)
            x = layers.Add()([shortcut, x])
        return x

    def _build(self):
        input_layer = layers.Input(self.input_shape)
        x = layers.Conv2D(kernel_size=(7, 7), filters=64, strides=2, padding='same')(input_layer)
        x = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(x)
        x = self._res_block(x, filters=64, down_sampling=False, blocks=2)
        x = self._res_block(x, filters=128, down_sampling=True, blocks=4)
        x = self._res_block(x, filters=256, down_sampling=True, blocks=6)
        x = self._res_block(x, filters=512, down_sampling=True, blocks=3)
        x = layers.GlobalAvgPool2D()(x)
        x = layers.Flatten()(x)
        output_layer = layers.Dense(units=self.output_shape, activation='softmax')(x)
        return models.Model(input_layer, output_layer)

    def _compile(self):
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics="accuracy")

    def summary(self):
        self.model.summary()

    def train(self, x_train, y_train, epochs, batch_size=64):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=64, verbose=self.verbose)

    def predict(self, x):
        return self.model.predict(x)

