from sklearn import tree
from sklearn.metrics import accuracy_score

import numpy as np

def load_data():
    """
    This helper function loads the dataset saved in the CSV file
    and returns 4 numpy arrays containing the training set inputs
    and labels, and the testing set inputs and labels.
    """

    # Load the training data from the CSV file
    training_data = np.genfromtxt('dataset.csv', delimiter=',', dtype=np.int32)

    # Extract the inputs from the training data array (all columns but the last one)
    #inputs = training_data[:, 0]

    # Extract the outputs from the training data array (last column)
    outputs = training_data[:, -1]

    # Separate the training and testing data
    #training_inputs = inputs[:10000]
    training_outputs = outputs[:]
    #testing_inputs = inputs[10000:]
    #testing_outputs = outputs[:]

    # Return the four arrays
    #return training_inputs, training_outputs, testing_inputs, testing_outputs
    return training_outputs

if __name__ == '__main__':

    # Load the training data
    test_outputs = load_data()

    counter = 0
    counter2 = 0
    for i in range(len(test_outputs)):
        if test_outputs[i] == 1:
            counter+=1
        else:
            counter2+=1

    print "Phishing #: " + str(counter)
    print "Non-Phishing #: " + str(counter2)

    total = counter + counter2

    print "total: " + str(total)
    """
    print "Find out corrlation between IP length and phishing site"


    print "Training data loaded."

    # Create a decision tree classifier model using scikit-learn
    classifier = tree.DecisionTreeClassifier()
    print "Decision tree classifier created."

    print "Beginning model training."
    # Train the decision tree classifier
    classifier.fit(train_inputs.reshape(-1,1), train_outputs)
    print "Model training completed."
    # Use the trained classifier to make predictions on the test data
    predictions = classifier.predict(test_inputs.reshape(-1,1))
    print "Predictions on testing data computed."

    # Print the accuracy (percentage of phishing websites correctly predicted)
    accuracy = 100.0 * accuracy_score(test_outputs, predictions)
    print "The accuracy of your decision tree on testing data is: " + str(accuracy)

    
    
    """
