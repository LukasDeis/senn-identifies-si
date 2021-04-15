import tensorflow as tf
from tensorflow.keras import layers
import keras
from ParameterizerLayer import ParameterizerLayer
import keras.losses


# A combination of layers, common in the parameterizer


class SENN:
    def __init__(
            self,
            parameterizer_out_shape=preprocessed_inputs_shape,
            preprocessed_inputs_shape=1924,  # TODO why does this change??? depends on preprocessing-model
            parameterizer_hidden_sizes=[200, 100, 50, 50],  # TODO fix this to actual size
            dropout_rate=0.1,
            batch_size=1
    ):
        self.parameterizer_out_shape = parameterizer_out_shape
        self.preprocessed_inputs_shape = preprocessed_inputs_shape
        self.parameterizer_hidden_sizes = parameterizer_hidden_sizes
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.input_shape = [self.batch_size, self.preprocessed_inputs_shape]
        self.input_layer = layers.Input(batch_shape=self.input_shape)
        self.functional_model = self.construct()
        loss_dict = {
            "relevances": keras.losses.mean_absolute_error,  # get_custom_loss(some_other_argument=1),
            "output": keras.losses.binary_crossentropy,  # tf.nn.log_poisson_loss,
            # "concepts": zero_loss
        }

        self.functional_model.compile(
            optimizer="adam",
            loss=loss_dict,
            loss_weights=[1, 5, 0],
            metrics=['accuracy']
        )


    def get_parameterizer(self):
        hidden_sizes = self.parameterizer_hidden_sizes
        dropout_rate = self.dropout_rate
        out_shape = self.parameterizer_out_shape

        x = self.input_layer
        x = ParameterizerLayer(hidden_sizes[0], dropout_rate)(x)
        x = ParameterizerLayer(hidden_sizes[1], dropout_rate)(x)
        x = ParameterizerLayer(hidden_sizes[2], dropout_rate)(x)
        x = ParameterizerLayer(hidden_sizes[3], dropout_rate)(x)
        x = layers.Dense(out_shape, activation='linear')(x)
        relevances = layers.Dropout(rate=dropout_rate, name="relevances")(x)
        return relevances

    def get_conceptizer(self):
        concepts = layers.Lambda(lambda t: t, name="concepts")(self.input_layer)
        return concepts

    def get_aggregator(self, relevances, concepts):
        aggregated = layers.multiply([relevances, concepts])
        aggregated = layers.Lambda(lambda t: tf.keras.backend.sum(t, axis=-1))(aggregated)
        aggregated = layers.Lambda(lambda t: tf.keras.activations.sigmoid(t), name="output")(aggregated)
        return aggregated

    def construct(self):
        relevances = self.get_parameterizer()
        concepts = self.get_conceptizer()
        aggregated = self.get_aggregator(relevances, concepts)
        out_layer = [aggregated, concepts, relevances]
        functional_model = tf.keras.Model(inputs=self.input_layer, outputs=out_layer)
        return functional_model

    # custom fun with more inputs
    # while this function is pretty senseless rn, it would be easy to manipulate the loss here
    def get_custom_loss(self, some_other_argument):
        def custom_loss(y_true, y_pred):
            loss = 0
            loss = loss + some_other_argument
            loss = keras.losses.binary_crossentropy(y_true, y_pred)
            return loss

        return custom_loss

    def zero_loss(self, y_true, y_pred):
        return 0

