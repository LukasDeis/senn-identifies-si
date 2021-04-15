import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from DataSorter import DataSorter
from DataPrepper import DataPrepper
from PrepprocessingModel import Preprocessor


#IF YOUR PYTHON ENVIRONMENT IS NOT SET-UP YET, YOU COULD TAKE A LOOK AT setup.py


# to see if your computer can utilize its GPU to speed everything up, let's take a look at how many GPUs are available
# if you installed tensorflow-gpu and all necessary CUDA toolkits and drivers it should be at least one
# If you feel like you should see more then you are, try looking at the console in which Jupyter is running
# it might give you an information about which cuda DLLs are missing.
# Often they end on the version of CUDA you're missing
# Yes, sometimes them seem ancient.
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


data_file = 'C:/Users/Lukas.Deis/Documents/dataset/MIND_Set_Data_exported.csv'
# convert to csv
dataframe = pd.read_csv(data_file)

target = "OQ_8"
string_targets = dataframe[target]
severity_sorting = { #TODO do these values make sense?
    "Nooit": 0.0,
    "Zelden": 0.25,
    "Soms": 0.5,
    "Vaak": 0.75,
    "Bijna altijd": 1
}


target_float = string_targets.map(severity_sorting)
target_categorical = np.where(target_float > severity_sorting["Nooit"], 1, 0)
dataframe['target'] = target_categorical # TODO this is now simple classification (0 or 1) but it could be more defined, would that not be better?

# Drop un-used columns. (including our now target which can not be used for training)
unused_cols = [target]
dataframe = dataframe.drop(columns=unused_cols)

# Patients that did not answer the target question can not be evaluated and are thus removed.
dataframe = dataframe[dataframe['target'].notna()]


tf.print("targets:", dataframe['target'])
#TODO note this in the report

dataframe.head()
dataSorter = DataSorter()
dataframe, numerical_features, categorical_int_features, categorical_cols = dataSorter.sort(dataframe, target)

data_prepper = DataPrepper()
train_ds, train, val, test = data_prepper.df_to_dataset(dataframe)
# The dataset returns a dictionary of column names (from the dataframe) that map to column values from rows in the dataframe.

preprocessor = Preprocessor()
preprocesessing_model = preprocessor.get_model(
    data_prepper,
    train,
    val,
    test,
    numerical_features,
    categorical_int_features,
    categorical_cols,
    target,
    batch_size=1  # TODO needs to be 1 rn (buggy!) but should be changeable from interface
)



#TODO the SENN model is stored in senn.py

# do pre-processing of data separately
processed_train_ds = train_ds.map(
  lambda x, y: (
      tf.cast(preprocesessing_model(x), dtype=tf.float32), # TODO this breaks if the batch-size is anything but 1
      tf.cast(y, dtype=tf.float32)
  )
)


for d in processed_train_ds.enumerate():
    tf.print(d)



# Define the Keras TensorBoard callback, used for the animated, interactive tensorboard visualizatioon
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

#This should plot the exhaustive graph, but is a bit unreliable
tf.keras.utils.plot_model(functional_model, show_shapes=True, rankdir="LR")

#tf.print("processed_train_ds shape:", processed_train_ds.take(0))
functional_model.fit(processed_train_ds, epochs=5, callbacks=tensorboard_callback)

#TODO testing is done in tester.py

preprocesessing_model.save('preprocessing_model')
functional_model.save('suicidal_ideation_model')
reloaded_preprocessing = tf.keras.models.load_model('preprocessing_model')
reloaded_model = tf.keras.models.load_model('suicidal_ideation_model')

#TODO analysis is done in analyzer.py


# storing everything for later

#location
directory = "./saves/medium_high_risk"

# models:
preprocesessing_model.save(directory+'preprocessing')
functional_model.save(directory+'my_pet_classifier')

# results:
explanations_frame.to_pickle(directory+"explanations_frame")
low_prop_explanations_frame.to_pickle(directory+"low_prop_explanations_frame")
high_prop_explanations_frame.to_pickle(directory+"high_prop_explanations_frame")
output.to_pickle(directory+"output")

# evaluation
metrics.to_pickle(directory+"metrics")