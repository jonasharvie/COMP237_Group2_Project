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
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


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


# 6. Use pandas.sample to shuffle the dataset, set frac =1
shuffled_data = data.sample(frac=1, random_state=42)

# 7. Using pandas split your dataset into 75% for training and 25% for testing
# make sure to separate the class from the feature(s). (Do not use test_train_ split)
train_size = int(0.75 * len(shuffled_data))

X_train = shuffled_data['CONTENT'][:train_size]
y_train = shuffled_data['CLASS'][:train_size]

X_test = shuffled_data['CONTENT'][train_size:]
y_test = shuffled_data['CLASS'][train_size:]


# 3. Using nltk toolkit classes and methods prepare the data for model building, 
# refer to the third lab tutorial in module 11 (Building a Category text predictor ). 
# Use count_vectorizer.fit_transform().

count_vectorizer = CountVectorizer()
X_train_tc = count_vectorizer.fit_transform(X_train)

X_test_tc = count_vectorizer.transform(X_test)

# 4. Present highlights of the output (initial features) 
# such as the new shape of the data and any other useful information before proceeding. 
print("\n4. Present highlights of the output (initial features) ")
print("\nDimensions of training data:", X_train_tc.shape) # (370, 1357) = 370 comments, 1357 =  number of unique words in the dataset
print("\ndata type:",type(X_train_tc))
print("\nMax value:", X_train_tc.max()) # max word count = 16 means at least one word appears 16 times in at least one commnet
print("Min value:", X_train_tc.min()) # min word count = 0 means at least one word does not appear in at least one comment
print("Mean value:", X_train_tc.mean()) # mean word count = .013 means most words do not appear often across all comments


# 5. Downscale the transformed data using tf-idf (Term Frequency-Inverse Document Frequency)
# and again present highlights of the output (final features) such as the new shape of the data 
# and any other useful information before proceeding.

tfidf = TfidfTransformer()
X_train_tfidf = tfidf.fit_transform(X_train_tc)

X_test_tfidf = tfidf.transform(X_test_tc)

print("\n5. Downscale the transformed data using tf-idf (Term Frequency-Inverse Document Frequency and again present highlights of the output (final features))")
print("\ndata type:",type(X_train_tfidf))# sparse row matrix because there are many unique words that dont appear in most 
print("\nDimensions of training data:", X_train_tfidf.shape)
print("\nMax value:", X_train_tfidf.max()) # max word count = 1 because the previous max and min have been normalized so the new range is between 1-0
print("Min value:", X_train_tfidf.min()) # min word count = 0 even after normalization 0 is still 0
print("Mean value:", X_train_tfidf.mean()) # mean word count = .0021 new mean after data is normalized


print(f"Training set size: {X_train_tfidf}")
print(f"Testing set size: {X_test_tfidf}")


# 8. Fit the training data into a Naive Bayes classifier.

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')

# 9. Cross validate the model on the training data using 5-fold 
# and print the mean results of model accuracy.

cross_val_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)
mean_accuracy = np.mean(cross_val_scores)

print(f'Cross-Validation Scores: {cross_val_scores}')
print(f'Mean Cross-Validation Accuracy: {mean_accuracy}')

# 10. Test the model on the test data, 
# print the confusion matrix and the accuracy of the model.

conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Task #10 Results:")
print("Confusion Matrix:")
print(conf_matrix)
print(f"Accuracy: {accuracy * 100:.2f}%")

# 11. As a group come up with 6 new comments 
# (4 comments should be non spam and 2 comment spam) 
# and pass them to the classifier and check the results.

new_comments = [
    "I absolutely love this song!",
    "Shakira's performance is incredible.",
    "Great video, well done!",
    "This is my favorite song of all time!",
    "Click here to win a free iPhone!",
    "Subscribe to my channel for more amazing content!"
]

# Transform new comments into the same vectorized format
new_comments_tc = count_vectorizer.transform(new_comments)
new_comments_tfidf = tfidf.transform(new_comments_tc)

# Predict using the trained model
new_predictions = model.predict(new_comments_tfidf)

print("\nTask #11 Results:")
for comment, prediction in zip(new_comments, new_predictions):
    label = "Spam" if prediction == 1 else "Non-spam"
    print(f"Comment: {comment}\nPrediction: {label}\n")

# 12. Present all the results and conclusions.

# 13. Drop code, report and power point presentation into the project assessment folder for grading.
