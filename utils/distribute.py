import os

import tensorflow as tf


def colab_tpu_strategy():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
    return strategy


def gpu_strategy():
    strategy = tf.distribute.MirroredStrategy()
    return strategy
