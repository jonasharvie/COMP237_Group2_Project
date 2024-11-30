# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 21:51:50 2024

@author: Group 2
    Jonas Harvie
    Daniel Samarin
    Elijah Robinson
    Absar Siddiqui-Atta
    Eric Lau
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# 1. Load the data into a pandas data frame.

data = pd.read_csv('Youtube05-Shakira.csv')
print("Youtube05-Shakira.csv head of data:")
print(data.head())


# 2. Carry out some basic data exploration and present your results.
# (Note: You only need two columns for this project, 
# make sure you identify them correctly, if any doubts ask your professor)
print("All Column names:")
print(data.columns)
data = data[['CONTENT', 'CLASS']]
X = data['CONTENT'] # features
y = data['CLASS'] # target
  

print(X.head())
print(y.head())

print("Shape of the dataset:", data.shape)
print("Missing values in each column:")
print(data.isnull().sum())

print("Summary statistics:")
print(data.describe())

print("X and Y Column names:")
print(data.columns)

# 3. Using nltk toolkit classes and methods prepare the data for model building, 
# refer to the third lab tutorial in module 11 (Building a Category text predictor ). 
# Use count_vectorizer.fit_transform().

count_vectorizer = CountVectorizer()
train_tc = count_vectorizer.fit_transform(X)

# 4. Present highlights of the output (initial features) 
# such as the new shape of the data and any other useful information before proceeding. 
print("\n4. Present highlights of the output (initial features) ")
print("\nDimensions of training data:", train_tc.shape) # (370, 1357) = 370 comments, 1357 =  number of unique words in the dataset
print("\ndata type:",type(train_tc))
print("\nMax value:", train_tc.max()) # max word count = 16 means at least one word appears 16 times in at least one commnet
print("Min value:", train_tc.min()) # min word count = 0 means at least one word does not appear in at least one comment
print("Mean value:", train_tc.mean()) # mean word count = .013 means most words do not appear often across all comments


# 5. Downscale the transformed data using tf-idf (Term Frequency-Inverse Document Frequency)
# and again present highlights of the output (final features) such as the new shape of the data 
# and any other useful information before proceeding.

tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)
print("\n5. Downscale the transformed data using tf-idf (Term Frequency-Inverse Document Frequency and again present highlights of the output (final features))")
print("\ndata type:",type(train_tfidf))# sparse row matrix because there are many unique words that dont appear in most 
print("\nDimensions of training data:", train_tfidf.shape)
print("\nMax value:", train_tfidf.max()) # max word count = 1 because the previous max and min have been normalized so the new range is between 1-0
print("Min value:", train_tfidf.min()) # min word count = 0 even after normalization 0 is still 0
print("Mean value:", train_tfidf.mean()) # mean word count = .0021 new mean after data is normalized

# 6. Use pandas.sample to shuffle the dataset, set frac =1
shuffled_index = pd.DataFrame(index=train_tfidf.index).sample(frac=1, random_state=2).index

# shuffle the index
X_shuffled = train_tfidf[shuffled_index]
y_shuffled = y.iloc[shuffled_index].reset_index(drop=True)

# 7. Using pandas split your dataset into 75% for training and 25% for testing
train_size = int(0.75 * X_shuffled.shape[0])

X_train = X_shuffled[:train_size]
y_train = y_shuffled[:train_size]

X_test = X_shuffled[train_size:]
y_test = y_shuffled[train_size:]

print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# 8. Fit the training data into a Naive Bayes classifier.

# 9. Cross validate the model on the training data using 5-fold 
# and print the mean results of model accuracy.

# 10. Test the model on the test data, 
# print the confusion matrix and the accuracy of the model.

# 11. As a group come up with 6 new comments 
# (4 comments should be non spam and 2 comment spam) 
# and pass them to the classifier and check the results.

# 12. Present all the results and conclusions.

# 13. Drop code, report and power point presentation into the project assessment folder for grading.
