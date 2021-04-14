
# do pre-processing of data separately
processed_test_ds = test_ds.map(
  lambda x, y: (
      tf.cast(preprocesessing_model(x), dtype=tf.float32),
      tf.cast(y, dtype=tf.float32)
  )
)


accuracy = functional_model.evaluate(processed_test_ds)
print("Accuracy", accuracy)

#divide x and y of test set
x = []
y_true = []

for x_var, y_var in processed_test_ds:
    x.append(x_var)
    y_true.append(y_var[0])


##imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


## basic metrics

#skip NaN values here and in analysis later? TODO

def get_true_pos(y, pred, th):
    pred_t = (pred > th)
    return np.sum((pred_t == True) & (y == 1))


def get_true_neg(y, pred, th):
    pred_t = (pred > th)
    return np.sum((pred_t == False) & (y == 0))


def get_false_neg(y, pred, th):
    pred_t = (pred > th)
    return np.sum((pred_t == False) & (y == 1))


def get_false_pos(y, pred, th):
    pred_t = (pred > th)
    return np.sum((pred_t == True) & (y == 0))

def get_acc(tp, tn, fp, fn):
    total = sum([tp, tn, fp, fn])
    correct = sum([tp, tn])
    return correct / total

def get_prevalence(tp, tn, fp, fn):
    return (tp + fn) / (tp + tn + fp + fn)

def get_specificity(tp, tn, fp, fn):
    return tn / (tn + fp)

def get_sensitivity(tp, tn, fp, fn):
    return tp / (tp + fn)

def get_PPV(tp, tn, fp, fn):
    return (tp / (tp + fp))

def get_NPV(tp, tn, fp, fn):
    return (tn / (fn + tn))


#### based on coursera util.py for metrics


def get_performance_metrics(y, pred, class_labels, threshold,
                            tp=get_true_pos,
                            tn=get_true_neg, fp=get_false_pos,
                            fn=get_false_neg,
                            acc=get_acc, prevalence=get_prevalence, spec=get_specificity,
                            sens=get_sensitivity, ppv=get_PPV, npv=get_NPV, auc=None, f1=None):
    columns = ["Label", "TP", "TN", "FP", "FN", "Accuracy", "Prevalence",
               "Sensitivity",
               "Specificity", "PPV", "NPV", "AUC", "F1", "Threshold"]
    df = pd.DataFrame(columns=columns)
    for i in range(len(class_labels)):  ## i is the concerning class in each iteration
        class_pred = pred[i]
        # for separate classes:
        # count_preds = len(class_pred) # the class was tried to predict as often as a prediction was made
        # class_y = np.repeat(i, count_preds) # we filter for one class only anyway -> all the same
        # for one class:
        class_y = y[i]
        ## get base metrics
        true_p = round(tp(class_y, class_pred, threshold), 3) if tp != None else "Not Defined"
        true_n = round(tn(class_y, class_pred, threshold), 3) if tn != None else "Not Defined"
        false_p = round(fp(class_y, class_pred, threshold), 3) if fp != None else "Not Defined"
        false_n = round(fn(class_y, class_pred, threshold), 3) if fn != None else "Not Defined"

        ## construct df for all data concerning class
        row_data = {
            "Label": class_labels[i],
            "TP": true_p,
            "TN": true_n,
            "FP": false_p,
            "FN": false_n,
            "Accuracy": round(acc(true_p, true_n, false_p, false_n), 3) if acc != None else "Not Defined",
            "Prevalence": round(prevalence(true_p, true_n, false_p, false_n),
                                3) if prevalence != None else "Not Defined",
            "Sensitivity": round(sens(true_p, true_n, false_p, false_n), 3) if sens != None else "Not Defined",
            "Specificity": round(spec(true_p, true_n, false_p, false_n), 3) if spec != None else "Not Defined",
            "PPV": round(ppv(true_p, true_n, false_p, false_n), 3) if ppv != None else "Not Defined",
            "NPV": round(npv(true_p, true_n, false_p, false_n), 3) if npv != None else "Not Defined",
            "AUC": round(auc(class_y, class_pred), 3) if auc != None else "Not Defined",
            "F1": round(f1(class_y, class_pred > threshold), 3) if f1 != None else "Not Defined",
            "Threshold": round(threshold, 3)
        }
        tf.print("One row of metrics:", row_data)
        df = df.append(row_data, ignore_index=True)
    return df


def print_confidence_intervals(class_labels, statistics):
    df = pd.DataFrame(columns=["Mean AUC (CI 5%-95%)"])
    for i in range(len(class_labels)):
        mean = statistics.mean(axis=1)[i]
        max_ = np.quantile(statistics, .95, axis=1)[i]
        min_ = np.quantile(statistics, .05, axis=1)[i]
        df.loc[class_labels[i]] = ["%.2f (%.2f-%.2f)" % (mean, min_, max_)]
    return df


def get_curve(gt, pred, target_names, curve='roc'):
    for i in range(len(target_names)):
        if curve == 'roc':
            curve_function = roc_curve
            auc_roc = roc_auc_score(gt[:, i], pred[:, i])
            label = target_names[i] + " AUC: %.3f " % auc_roc
            xlabel = "False positive rate"
            ylabel = "True positive rate"
            a, b, _ = curve_function(gt[:, i], pred[:, i])
            plt.figure(1, figsize=(7, 7))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(a, b, label=label)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

            plt.legend(loc='upper center', bbox_to_anchor=(1.3, 1),
                       fancybox=True, ncol=1)
        elif curve == 'prc':
            precision, recall, _ = precision_recall_curve(gt[:, i], pred[:, i])
            average_precision = average_precision_score(gt[:, i], pred[:, i])
            label = target_names[i] + " Avg.: %.3f " % average_precision
            plt.figure(1, figsize=(7, 7))
            plt.step(recall, precision, where='post', label=label)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.legend(loc='upper center', bbox_to_anchor=(1.3, 1),
                       fancybox=True, ncol=1)


#make predictions
y_pred = []
stored_concepts = []
stored_relevances = []

for x_var in x:
    aggregated, concepts, relevances = functional_model.predict(x_var)
    y_pred.append(aggregated[0])
    stored_concepts.append(concepts)
    stored_relevances.append(relevances)
y_pred = np.array(y_pred)


y_true_array = np.array(y_true)
class_labels = ['suicidal ideation']

classification_thres = 0.5
metrics = get_performance_metrics([y_true_array], [y_pred], class_labels, classification_thres)
metrics

functional_model.summary()

#visualize model in an interactive way
#sadly only works until the preprocessing layers are over
# tensorboard sometimes thinks there still is an instance running when it is not
# fix that by deleting the contents of this folder or your equivalent of it
# C:\Users\deisl\AppData\Local\Temp\.tensorboard-info


# TODO reactivate this when tensorboard has been installed:
"""
%reload_ext tensorboard
# rankdir='LR' is used to make the graph horizontal.
#tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
%tensorboard --logdir logs
"""

