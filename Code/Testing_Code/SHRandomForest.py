from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support


import numpy as np



def load_data():

    # Load the training data from the CSV file
    training_data = np.genfromtxt('dataset.csv', delimiter=',', dtype=np.int32)

    # Extract the inputs from the training data array (all columns but the last one)
    #inputs = training_data[:, [0,1,3,8,14,16,18,22,23]]
    inputs = training_data[:, [6,7,8,9,13,14,15,16,24,26,29]]
    #inputs = training_data[:,:-1]    
    # Extract the outputs from the training data array (last column)
    outputs = training_data[:, -1]

    # Separate the training and testing data
    training_inputs = inputs[:10000]
    training_outputs = outputs[:10000]
    testing_inputs = inputs[10001:]
    testing_outputs = outputs[10001:]
    #true_input = training_data[10001:,30]

    # Return the four arrays
    return training_inputs, training_outputs, testing_inputs, testing_outputs # true_input



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
    
    #Rank each feature
    #importance = classifier.feature_importances_
    #print "\nFeature Importances: " + str(importance)
    #test_inputs[:,-1]
    #score = precision_score(true_input, predictions, average= 'binary')
    print str(test_outputs[:19])
    print str(predictions[:19])
    #print str(score)

    # Print the accuracy (percentage of phishing websites correctly predicted)
    accuracy = 100.0 *accuracy_score(test_outputs, predictions) # 
    #print "\nThe Accuracy of Random Forest on testing data is: " + str(accuracy)
    print "\nAccuracy: %.2f" %accuracy

    f1 = f1_score(test_outputs, predictions, average='macro')
    #print "The F1 score of Random Forest is: " + str(f1)
    print "F1 Score: %.4f" %f1

    pre = precision_score(test_outputs, predictions, average='macro')
    #print "The precision for Random Forest on testing data is: " + str(pre)
    #print "Precison: " +str(pre)
    print ("Precison: %.4f" %pre)

    recall = recall_score(test_outputs, predictions, average='macro')
    #print "The recall value for Random Forest on testing data is: " + str(recall)
    print "  Recall: %.4f" %recall
    
    prf= precision_recall_fscore_support(test_outputs, predictions, average='binary')
    print "\nPRF: " +str(prf)
    
    #{0:0.2f}'.format()
    #score = classifier.
    
    #prob = classifier.predict_proba(train_inputs)
    #print "Probabilities: "+ str(prob)


    #1. training size and testing accurcy
    #2. select which feature is more important?
    #3. and which subset is more important to determine the phishing site
    #4. try different classifier for the dataset
    #5. statical test and T-test for selecting useful features
