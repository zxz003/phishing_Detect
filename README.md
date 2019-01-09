# Detect Phishing Website Using Machine Learning Techniques
This repo is for Spring 2018 OSS project. Detecting phishing website via different machine learning techniques. Some of our code was from or inspired by npapernot's "Detecting phishing websites using a decision tree" you can find more [here](https://github.com/npapernot/phishing-detection).

## Publication
You can see the published results [here](https://link.springer.com/chapter/10.1007%2F978-3-030-00557-3_46).

## Installation and Setup
Just clone the repo and goto code section run the RandomForest.py and you can see the results.

## The Dataset
The dataset was collected by analyzing a collection of `11055` websites among which some were used for phishing and others not. For each website included in the dataset, `30` attributes are given.

The list includes for instance the URL length, whether the website uses pop-up windows or Iframes, or how old the domain registration is.

Each website in the dataset is labeled by `-1` if it is not a phishing
website and by `1` if it is a website used for phishing.

## What Did We Do?
1. Choose a specific method and improve its performance on predit the phishing site.
A: We are using Random Forest classifier since the dataset have many features; then we display the confiusion matrix for current feature set and calculated the accuracy, percision, recall and f-score for to evaluate the trained classifier; we also display the RoC matrix for each trained classifier.
(Done on 23/04/2018)

2. What approach to take to improve the performance? 
A: Importance selection based on features importance.
(Done on 27/04/2018)
