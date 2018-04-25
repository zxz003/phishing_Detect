from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score

import numpy as np



def load_data():

    # Load the training data from the CSV file
    training_data = np.genfromtxt('dataset.csv', delimiter=',', dtype=np.int32)

    # Extract the inputs from the training data array (all columns but the last one)
    #inputs = training_data[:, [0,1,3,8,14,16,18,22,23]]
    #inputs = training_data[:, [6,7,8,9,13,14,15,16,24,26,29]]
    inputs = training_data[:,:-1]
    # true_input = training_data[10001:10015,30]
    # Extract the outputs from the training data array (last column)
    outputs = training_data[:, -1]

    # Separate the training and testing data
    training_inputs = inputs[:10000]
    training_outputs = outputs[:10000]
    testing_inputs = inputs[10001:]
    testing_outputs = outputs[10001:]

    # Return the four arrays
    return training_inputs, training_outputs, testing_inputs, testing_outputs
        #, true_input



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
    
    
    #score = precision_score(true_input, predictions, average=None)
    #print "Score: "+ str(true_input)

    # Print the accuracy (percentage of phishing websites correctly predicted)
    accuracy = 100.0 * accuracy_score(test_outputs, predictions)
    print "\nThe Accuracy of Random Forest on testing data is: " + str(accuracy)

    precision, recall, fscore, support = score(test_outputs, predictions)

    print ('precision: \t{}'.format(precision))
    print ('recall: \t{}'.format(recall))
    print ('fscore: \t{}'.format(fscore))
    print ('support: \t{}'.format(support))

    '''
    Note:
    1. Precision: P = TP/TP + FP - all correct classified / all classified
    
    2. Recall: R = TP/TP + FN - all correct classified / should be classified
    
    3. F1 weight the Precision and Recall, in other word choose which one is more important
        Default is P = R 
    '''


    
    #score = classifier.
    importance = classifier.feature_importances_
    print "\nFeature Importances: \n" + str(importance)
    #prob = classifier.predict_proba(train_inputs)
    #print "Probabilities: "+ str(prob)

















    #1. training size and testing accurcy
    #2. select which feature is more important?
    #3. and which subset is more important to determine the phishing site
    #4. try different classifier for the dataset
    #5. statical test and T-test for selecting useful features
