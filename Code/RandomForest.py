import numpy as np
import matplotlib.pyplot as plt

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
    # inputs = training_data[:, [0,2,3,4,5,8,9,10,11,12,16,17,19,20,21,22,23,24,26,27,29]]    #Only 2
    # inputs = training_data[:, [1,6,7,13,14,15,25,28]]
    # inputs = training_data[:,:-1]
    inputs = training_data[:, [ 0, 1, 5, 6, 7, 8, 12, 13, 14, 15, 23, 24, 25, 26, 27, 28]] #95.8 - 16


    # Extract the outputs from the training data array (last column)
    outputs = training_data[:, -1]

    # Separate the training and testing data
    training_inputs = inputs[:10000]
    training_outputs = outputs[:10000]
    # testing size is 1054
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
    classifier.fit(train_inputs, train_outputs)
    print "Model training completed."

    # Use the trained classifier to make predictions on the test data
    predictions = classifier.predict(test_inputs)
    print "Predictions on testing data computed."

    # Print the accuracy (percentage of phishing websites correctly predicted)
    accuracy = 100.0 * accuracy_score(test_outputs, predictions)
    print "\nThe Accuracy of Random Forest on testing data is: " + str(accuracy)

    precision, recall, fscore, support = score(test_outputs, predictions)
    confusion_matrix = ConfusionMatrix(test_outputs, predictions)
    print ('\n')
    print("Confusion matrix:\n%s" % confusion_matrix)
    print ('\n')
    
    #in body
    #fpr, tpr, thresholds = metrics.roc_curve(test_outputs,predictions)
    #auc = metrics.auc(fpr, tpr)
    #print accuracy precision recall f1 here

    print ('precision: \t{}'.format(precision))
    print ('recall: \t{}'.format(recall))
    print ('fscore: \t{}'.format(fscore))
    print ('support: \t{}'.format(support))

    print ('\n')

    TPR = np.array([0.511931, 0.78308, 0.915401, 0.941432, 0.947939])
    FPR = np.array([0.530303, 0.242424, 0.060606, 0.03367, 0.021886])

    fpr, tpr,_ = roc_curve(test_outputs,predictions)
    roc_auc = auc(fpr, tpr)

    print ('FPR: \t{}'.format(fpr))
    print ('TPR: \t{}'.format(tpr))
    print ('AUC: \t{}'.format(roc_auc))

<<<<<<< HEAD
    print ('\n')

    plt.figure()
    lw = 3  # Line width
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

=======
>>>>>>> 3696df5ca9f3a1a0e948dd30b317d1cf737cc062
    importance = classifier.feature_importances_
    print "\nFeature Importances: \n" + str(importance)
