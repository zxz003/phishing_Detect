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
    inputs = training_data[:, 0]

    # Extract the outputs from the training data array (last column)
    outputs = training_data[:, -1]

    # Separate the training and testing data
    training_inputs = inputs[:10000]
    training_outputs = outputs[:10000]
    testing_inputs = inputs[10000:]
    testing_outputs = outputs[10000:]

    # Return the four arrays
    return training_inputs, training_outputs, testing_inputs, testing_outputs


if __name__ == '__main__':
    print "Find out corrlation between IP length and phishing site"

    # Load the training data
    train_inputs, train_outputs, test_inputs, test_outputs = load_data()

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
