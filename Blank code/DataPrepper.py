import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


class DataPrepper:
    def __init__(self):
        print("dataPrepper!")

    # A utility method to create a tf.data dataset from a Pandas Dataframe
    def df_to_dataset(self, dataframe, shuffle=True, batch_size=1):
        dataframe = dataframe.copy()
        labels = dataframe.pop('target')

        # tf.print(dataframe.dtypes) #[539 rows x 396 columns] when including dates
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))

        # ds = tf.data.Dataset.from_tensor_slices((values, labels.values))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(batch_size)
        return ds

    def prep(self, dataframe):
        train, test = train_test_split(dataframe, test_size=0.25)
        train, val = train_test_split(train, test_size=0.1)
        print(len(train), 'train examples')
        print(len(val), 'validation examples')
        print(len(test), 'test examples')

        # To tackle the class imbalance, the minority-class will be oversampled
        # separate the two classes
        # S_I = Suicidal Ideation
        do_experience_S_I = train[train.target == 1]
        do_not_experience_S_I = train[train.target == 0]

        tf.print("Number of samples that DO experience suicidal ideation", len(do_experience_S_I))
        tf.print("Number of samples that do NOT experience suicidal ideation", len(do_not_experience_S_I))

        upsampled_do_experience_S_I = resample(
            do_experience_S_I,
            replace=True,  # sampling WITH replacement
            n_samples=len(do_not_experience_S_I),  # so the amount of samples is the same for both classes
            random_state=35  # so the results are reproducible (like seed)
        )

        # overwrite the old dataframe with the new, balanced one
        balanced_frame = pd.concat([upsampled_do_experience_S_I, do_not_experience_S_I])
        train = balanced_frame
        print("after resampling, the number of targets is balanced:")
        tf.print(train.target.value_counts())

        # TODO note this in the report

        batch_size = 1
        train_ds = self.df_to_dataset(train, batch_size=batch_size)

        [(train_features, label_batch)] = train_ds.take(1)
        print('Every feature:', list(train_features.keys()), "\n")
        print('A batch of PTSDFinal:', train_features['PTSDFinal'], "\n")
        print('A batch of targets:', label_batch)
        return train_ds
