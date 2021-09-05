from .model import ResNet34


def run(model: ResNet34, x_train, y_train, validation_data=None, epochs=128, batch_size=64, verbose=True):
    # cp_callback = tf.keras.callbacks.ModelCheckpoint("checkpoints/resnet/ckpt",
    #                                                  save_weights_only=True,
    #                                                  verbose=True)
    # tensorboard_callback = TensorBoard(log_dir="logs",
    #                                    histogram_freq=0,
    #                                    write_graph=True,
    #                                    write_images=True)
    model.fit(x_train,
              y_train,
              epochs=epochs,
              validation_data=validation_data,
              batch_size=batch_size,
              verbose=verbose,
              shuffle=True,
              # callbacks=[tensorboard_callback, cp_callback]
              )
