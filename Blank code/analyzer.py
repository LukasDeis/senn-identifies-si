import numpy as np
import pandas as pd
import tensorflow as tf


# create a dataframe of all stored outputs from testing
stored_aggregates = y_pred
outputs = pd.DataFrame(list(zip(stored_aggregates, stored_concepts, stored_relevances)),
                      columns =['aggregated', 'concepts', 'relevances'])

# creating a legend that contains which output belongs to which input
out_legend = []

for header in output_sizes:
    size = output_sizes[header]
    for layer in range(0, size):
        out_legend.append(header)


# select which sample to look at

def investigate_sample(output):
    aggregated = output['aggregated']
    concept = output['concepts']
    relevance = output['relevances']

    relevance = np.array(relevance)

    # if I multiply 0 inputs with the relevances first, only relevant parts will be shown
    binary_concepts = [0 if concept == 0 else 1 for concept in concepts[0]]
    binary_concepts = np.array(binary_concepts)
    polarized_relevances = np.multiply(binary_concepts, relevance[0])

    filtered_output = [
        (name, relevance)
        for name, relevance in
        zip(out_legend, polarized_relevances[0])  # TODO make this a dictionary
        if not relevance == 0
    ]

    return aggregated, filtered_output


def normalize_frame(df):
    max_value = df.abs().max()
    return df / max_value


# if
# TODO filtered_output was a dictionary,
# TODO and output was a param to investigate_sample()
# one could loop through all outputs and make a dataframe with all the explanations

pd.set_option('display.max_rows', 1000)

all_features = numerical_features + categorical_int_features + categorical_cols

aggregates = []
explanations = []
low_prop_explanations = []
high_prop_explanations = []

for target_index in range(0, len(outputs.index)):
    output = outputs.loc[[target_index]]
    aggregated, explanation = investigate_sample(output)
    aggregates.append(aggregated)
    dict_exp = dict(explanation)
    explanations.append(dict_exp)

    if aggregated[target_index] > classification_thres:
        high_prop_explanations.append(dict_exp)
    else:
        low_prop_explanations.append(dict_exp)

    # make  dataframes from records during loop
explanations_frame = pd.DataFrame.from_records(explanations)
explanations_frame.columns = all_features

low_prop_explanations_frame = pd.DataFrame.from_records(low_prop_explanations)
low_prop_explanations_frame.columns = all_features

high_prop_explanations_frame = pd.DataFrame.from_records(high_prop_explanations)
high_prop_explanations_frame.columns = all_features

# calculate average values
average_relevance = explanations_frame.mean(axis=0)

average_relevance_low = low_prop_explanations_frame.mean(axis=0)

average_relevance_high = high_prop_explanations_frame.mean(axis=0)

tf.print("The following frames have all been independently normalized and then sorted descendingly.")
tf.print("normalized average_relevance\n",normalize_frame(average_relevance).sort_values(ascending=False), "\n")
normalized_average_relevance_low = normalize_frame(average_relevance_low)
tf.print("normalized low probabilities average_relevance\n",normalized_average_relevance_low.sort_values(ascending=False), "\n")
normalized_average_relevance_high = normalize_frame(average_relevance_high)
tf.print("normalized high probabilities average_relevance\n",normalized_average_relevance_high.sort_values(ascending=False))
differential_frame = normalized_average_relevance_high - normalized_average_relevance_low
normalized_diff_frame = normalize_frame(differential_frame)
sorted_normalized_diff_frame = normalized_diff_frame.sort_values(ascending=False)
tf.print("normalized differences between high and low probabilities\n", sorted_normalized_diff_frame)
# so the absence of a feature is actually not really taken into account explicitly.
# however: if a certain feature is absent, that shifts the relevances of other features.
# thus one can say that in absence of a specific feature, the other, shown features become relevant

# this does make the average values less informative - the complex relationships are ignored
# a decision tree would be more suited to look at those relationships
# to inspect a specific sample:
target_index = 80 # just enter the index of the sample
output = outputs.loc[[target_index]]
aggregated, explanation = investigate_sample(output)
print(
    "This particular person had a %.1f percent probability "
    "of experiencing suicidal ideation." % (100 * aggregated)
)
tf.print("The explanations are: \n", explanation)