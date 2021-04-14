# A combination of layers, common in the parameterizer

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

#functional model def

#TODO no more static sizes
batch_size = 1
preprocessed_inputs_shape = 1924 #TODO why does this change???
dropout_rate=0.1
hidden_sizes = [200, 100, 50, 50]  # TODO fix this to actual size
out_shape = preprocessed_inputs_shape

input_shape = [batch_size, preprocessed_inputs_shape]
input_layer = layers.Input(batch_shape = input_shape)

x = input_layer
###Parameterizer###
x = ParameterizerLayer(hidden_sizes[0], dropout_rate)(x)
x = ParameterizerLayer(hidden_sizes[1], dropout_rate)(x)
x = ParameterizerLayer(hidden_sizes[2], dropout_rate)(x)
x = ParameterizerLayer(hidden_sizes[3], dropout_rate)(x)
x = layers.Dense(out_shape, activation='linear')(x)
relevances = layers.Dropout(rate=dropout_rate, name="relevances")(x)

###Conceptizer###
concepts = layers.Lambda(lambda t: t, name="concepts")(input_layer)

###Aggregator###

aggregated = layers.multiply([relevances, concepts])
aggregated = layers.Lambda(lambda t: tf.keras.backend.sum(t, axis=-1))(aggregated)
aggregated = layers.Lambda(lambda t: tf.keras.activations.sigmoid(t), name="output")(aggregated)

out_layer = [aggregated, concepts, relevances]

functional_model = tf.keras.Model(inputs=input_layer, outputs=out_layer)


# custom fun with more inputs

def get_custom_loss(some_other_argument):
    def custom_loss(y_true, y_pred):
        loss = 0
        loss = loss + some_other_argument
        loss = keras.losses.binary_crossentropy(y_true, y_pred)
        return loss

    return custom_loss

def zero_loss(y_true, y_pred):
    return 0

loss_dict = {
    "relevances": keras.losses.mean_absolute_error, #get_custom_loss(some_other_argument=1),
    "output": keras.losses.binary_crossentropy, #tf.nn.log_poisson_loss,
    #"concepts": zero_loss
}

functional_model.compile(
    optimizer="adam",
    loss=loss_dict,
    loss_weights=[1, 5, 0],
    metrics= ['accuracy']
)
functional_model.summary()