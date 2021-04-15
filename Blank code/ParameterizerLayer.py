import tensorflow as tf
from tensorflow.keras import layers

class ParameterizerLayer(layers.Layer):

    def __init__(self, out_shape, dropout_rate):
        super(ParameterizerLayer, self).__init__()
        self.para_lin = layers.Dense(out_shape, activation='linear')
        self.para_drop = layers.Dropout(dropout_rate)
        self.para_relu = layers.Dense(out_shape, activation=tf.keras.layers.LeakyReLU(alpha=0.05))

    def call(self, input_tensor, training=False):
        x = self.para_lin(input_tensor)
        if training:
            x = self.para_drop(x, training=training)
        x = self.para_relu(x)
        return x

    # should minimize robustness loss