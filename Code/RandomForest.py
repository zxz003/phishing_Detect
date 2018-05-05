import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from pandas_confusion import ConfusionMatrix
from sklearn.metrics import roc_curve, auc


def load_data():

    # Load the training data from the CSV file
    training_data = np.genfromtxt('dataset.csv', delimiter=',', dtype=np.int32)

    # Extract the inputs from the training data array (all columns but the last one)

    # inputs = training_data[:, [23,25,26,27]]
    # inputs = training_data[:, [0,2,3,4,5,8,9,10,11,12,16,17,19,20,21,22,23,24,26,27,29]]
    # inputs = training_data[:, [1,6,7,13,14,15,25,28]]
    # inputs = training_data[:,:-1]
    inputs = training_data[:, [ 0, 1, 5, 6, 7, 8, 12, 13, 14, 15, 23, 24, 25, 26, 27, 28]]                              #95.8 - 16


    # Extract the outputs from the training data array (last column)
    outputs = training_data[:, -1]

    # Separate the training and testing data
    training_inputs = inputs[:10000]
    training_outputs = outputs[:10000]

    # testing size is 1055
    testing_inputs = inputs[10000:]
    testing_outputs = outputs[10000:]

    # Return the four arrays
    return training_inputs, training_outputs, testing_inputs, testing_outputs

if __name__ == '__main__':
    print "Using Random Forest technique to detect phishing websites\n"

    # Load the training data
    train_inputs, train_outputs, test_inputs, test_outputs = load_data()
    print "Training data loaded."

    # Create a RF classifier
    classifier = RFC()
    print "Random Forest classifier created."


    print "Beginning model training."
    # Train the RF classifier
    RF = classifier.fit(train_inputs, train_outputs)
    test_probas = RF.predict_proba(test_inputs)

    print ('test_probas: \t{}'.format(test_probas))

    print "Model training completed."

    # Use the trained classifier to make predictions on the test data
    predictions = classifier.predict(test_inputs)

    print "Predictions on testing data computed."

    # Print the accuracy (percentage of phishing websites correctly predicted)
    accuracy = 100.0 * accuracy_score(test_outputs, predictions)
    print "\nThe Accuracy of Random Forest on testing data is: " + str(accuracy) + "\n"

    precision, recall, fscore, support = score(test_outputs, predictions)
    confusion_matrix = ConfusionMatrix(test_outputs, predictions)

    # print out confusion matrix
    print("Confusion matrix:\n%s" % confusion_matrix)
    print ('\n')

    print ('precision: \t{}'.format(precision))
    print ('recall: \t{}'.format(recall))
    print ('fscore: \t{}'.format(fscore))
    print ('support: \t{}'.format(support))

    #calculate the FPR, TPR and threshold for current feature set
    fpr, tpr, thresholds = roc_curve(predictions,test_outputs)
    roc_auc = metrics.auc(fpr, tpr)

    print ('FPR: \t{}'.format(fpr))
    print ('TPR: \t{}'.format(tpr))
    print ('Thresholds: \t{}'.format(thresholds))
    print ('AUC: \t{}'.format(roc_auc))

    #Show the feature importance for current set
    importance = classifier.feature_importances_
    print "\nFeature Importances: \n" + str(importance)

    #plot the RoC curve based on test outputs and test_probas
    skplt.metrics.plot_roc_curve(test_outputs,test_probas)
    plt.show()