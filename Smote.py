import pickle

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from docutils.nodes import inline
import seaborn as sns
from sklearn.multiclass import OneVsRestClassifier
# Library to suppress warnings or deprecation notes
import warnings
warnings.filterwarnings('ignore')
# Library to split data
from sklearn.model_selection import train_test_split, cross_validate
from sklearn import tree
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier


# Libtune to tune model, get different metric scores
from sklearn import datasets, linear_model, metrics
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, classification_report, roc_curve, plot_roc_curve, auc, precision_recall_curve, plot_precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_score
from collections import Counter


df=pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")
df.head()
df.drop(['CholCheck','AnyHealthcare','MentHlth','NoDocbcCost','Sex','Education','Income'], axis=1,inplace=True)
df.head()
X = df.drop(['Diabetes_012'],axis=1)
y = df['Diabetes_012']

# Splitting data into training and test set:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# summarize the class distribution of the training dataset
counter = Counter(y_train)
print(counter)

# transform the training dataset
oversample = SMOTE(random_state=33)
X_train, y_train = oversample.fit_resample(X_train, y_train)


# Creating StandardScaler instance
sc = StandardScaler()

# Fitting Standard Scaller
X_train = sc.fit_transform(X_train)

# Scaling data
X_test = sc.transform(X_test)


# #Fitting RandomForestClassifier Model
# classifier = RandomForestClassifier(criterion= 'gini', n_estimators= 200, random_state= 51)
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# y_prob = classifier.predict_proba(X_test)
# cm = confusion_matrix(y_test, y_pred)
# #print
# print("Random Forest Classifier")
# print('Accuracy Score: ',accuracy_score(y_test, y_pred))
# print("\n"+"*"*50)
# print('\nClassification_report : ')
# print(classification_report(y_test, y_pred))
# print('ROC AUC score: ', roc_auc_score(y_test, y_prob,multi_class='ovo', average='weighted'))
# print('Accuracy Score: ',accuracy_score(y_test, y_pred))
#
# # Visualizing Confusion Matrix
# plt.figure(figsize = (6, 6))
# sns.heatmap(cm, cmap = 'Blues', annot = True, fmt = 'd', linewidths = 5, cbar = False, annot_kws = {'fontsize': 15},
#             yticklabels = ['Healthy', 'Diabetic'], xticklabels = ['Predicted Healthy', 'Predicted Diabetic'])
# plt.yticks(rotation = 0)
# plt.show()


#Fitting Logistic Regression Model
accuracies = {}
classifier = LogisticRegression(C= 0.25, random_state= 1000)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)
cm = confusion_matrix(y_test, y_pred)

#print
print ("Logistic Regression accuracy:")
print('Accuracy Score: ',accuracy_score(y_test, y_pred))
print("\n"+"*"*50)
print(classification_report(y_test, y_pred))
print('ROC AUC score: ', roc_auc_score(y_test, y_prob,multi_class='ovo', average='weighted'))
print('Accuracy Score: ',accuracy_score(y_test, y_pred))

# Visualizing Confusion Matrix
print ("Confusion Marix:")
plt.figure(figsize = (6, 6))
sns.heatmap(cm, cmap = 'Blues', annot = True, fmt = 'd', linewidths = 5, cbar = False, annot_kws = {'fontsize': 15},
            yticklabels = ['Healthy', 'Prediabetic','Diabetic'], xticklabels = ['Predicted Healthy','Predicted Prediabetic', 'Predicted Diabetic'])
plt.yticks(rotation = 0)
plt.show()


pickle.dump(classifier,open("smote.pkl","wb"))
print("done")

