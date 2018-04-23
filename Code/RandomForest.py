from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score

import numpy as np

from decision_tree import load_data


if __name__ == '__main__':
    print "Using Random Forest technique to detect phishing websites"

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
    print "The accuracy of Random Forest on testing data is: " + str(accuracy)


    #1. training size and testing accurcy
    #2. select which feature is more important?
    #3. and which subset is more important to determine the phishing site
    #4. try different classifier for the dataset
    #5. statical test and T-test for selecting useful features
