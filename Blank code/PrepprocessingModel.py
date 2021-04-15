import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers


class Preprocessor:

    def get_normalization_layer(self, name, dataset):
        # Create a Normalization layer for our feature.
        normalizer = preprocessing.Normalization()

        # Prepare a Dataset that only yields our feature.
        feature_ds = dataset.map(lambda x, y: x[name])

        # Learn the statistics of the data.
        normalizer.adapt(feature_ds)

        return normalizer

    def get_category_encoding_layer(self, name, dataset, dtype, max_tokens=None):
        # Create a StringLookup layer which will turn strings into integer indices
        if dtype == 'string':
            index = preprocessing.StringLookup(max_tokens=max_tokens)
        else:
            index = preprocessing.IntegerLookup(max_values=max_tokens)

        # Prepare a Dataset that only yields our feature
        feature_ds = dataset.map(lambda x, y: x[name])

        # Learn the set of possible values and assign them a fixed integer index.
        index.adapt(feature_ds)

        # Create a Discretization for our integer indices.
        encoder = preprocessing.CategoryEncoding(max_tokens=index.vocab_size())

        # Prepare a Dataset that only yields our feature.
        feature_ds = feature_ds.map(index)

        # Learn the space of possible indices.
        encoder.adapt(feature_ds)

        # Apply one-hot encoding to our indices. The lambda function captures the
        # layer so we can use them, or include them in the functional model later.
        return lambda feature: encoder(index(feature))

    # TODO batch_size other than 1 leads to errors! why?
    def get_model(
            self,
            data_prepper,
            train,
            val,
            test,
            numerical_features,
            categorical_int_features,
            categorical_cols,
            target,
            batch_size=1):
        df_to_dataset = data_prepper.df_to_dataset
        train_ds = df_to_dataset(train, batch_size=batch_size)
        val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
        test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

        all_inputs = []
        encoded_features = []

        # bookkeeping for interpretation
        output_sizes = {}

        # Numeric features.
        for header in numerical_features:  # TODO use all headers in UMC set minus the ones I know are something else
            numeric_col = tf.keras.Input(shape=(1,), name=header)
            normalization_layer = self.get_normalization_layer(header, train_ds)
            encoded_numeric_col = normalization_layer(numeric_col)
            all_inputs.append(numeric_col)
            encoded_features.append(encoded_numeric_col)
            output_sizes[header] = encoded_numeric_col.get_shape()[1]

            # Categorical features encoded as integers.

            # Numeric features.
            for header in categorical_int_features:
                print(header)
                num_col = tf.keras.Input(shape=(1,), name=header, dtype='int64')
                encoding_layer = self.get_category_encoding_layer(header, train_ds, dtype='int64',
                                                                  max_tokens=5)
                encoded_col = encoding_layer(num_col)
                all_inputs.append(num_col)
                encoded_features.append(encoded_col)
                output_sizes[header] = encoded_col.get_shape()[1]

        # Categorical features encoded as string.
        # TODO include progress-bar
        categorical_cols.remove(target)
        for header in categorical_cols:
            categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
            encoding_layer = self.get_category_encoding_layer(header, train_ds, dtype='string',
                                                              max_tokens=5)
            # TODO maybe, this line should be duplicated and slightly changed to accomodate for different max_tokens
            encoded_categorical_col = encoding_layer(categorical_col)
            all_inputs.append(categorical_col)
            encoded_features.append(encoded_categorical_col)
            output_sizes[header] = encoded_categorical_col.get_shape()[1]

        # TODO: for many numeric features (hundreds, or more) it would be more efficient to concatenate them first and use a single normalization layer.

        # The first step towards a working model
        # is our preprocessed input.
        # As that is a relative complex task, that is regarded it's owy model.

        preprocessed_layers = layers.Concatenate()(encoded_features)  # encoded_features
        preprocesessing_model = tf.keras.Model(all_inputs, preprocessed_layers)
        preprocesessing_model.summary()

        return preprocesessing_model
