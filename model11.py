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
#model
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

df.info()
df.shape
df.columns
df.describe()
df.isnull().values.any()

df['Diabetes_012'].value_counts()
df['Diabetes_012'].value_counts().plot(kind = 'bar', title = 'Label Distribution')
df['Diabetes_012'].value_counts()
df['Diabetes'] = df['Diabetes_012'].replace({0.0:'Healthy', 1.0:'Pre-diabetic', 2.0:'Diabetic'})
# rename value of the column

Healthy = len(df[df.Diabetes_012 == 0])
PreDiabetic = len(df[df.Diabetes_012 == 1])
Diabteic = len(df[df.Diabetes_012 == 2])
print("Percentage of Patients Are Healthy: {:.2f}%".format((Healthy / (len(df.Diabetes_012))*100)))
print("Percentage of Patients Have Pre-Diabetic: {:.2f}%".format((PreDiabetic / (len(df.Diabetes_012))*100)))
print("Percentage of Patients Have Diabetic: {:.2f}%".format((Diabteic / (len(df.Diabetes_012))*100)))

diabetes_type = ['Healthy','Diabetic', 'Pre-Diabetic']
df.Diabetes_012.value_counts().plot.pie(labels=diabetes_type, autopct='%1.1f%%',shadow=True, startangle=90)

from sklearn.preprocessing import OneHotEncoder
df1=df.copy()
df1.info()
df1_Sex = df1[['Sex']].values.reshape(-1,1)
df1_HighBP = df1[["HighBP"]].values.reshape(-1,1)
df1_HighChol = df1[["HighChol"]].values.reshape(-1,1)

onehot_encoder=OneHotEncoder(sparse=False)
df1_OneEncoded=onehot_encoder.fit_transform(df1_Sex)
df1_OneEncoded1=onehot_encoder.fit_transform(df1_HighBP)
df1_OneEncoded2=onehot_encoder.fit_transform(df1_HighChol)

#Sex
df1["Male"] = df1_OneEncoded[:,1]
df1["Female"] = df1_OneEncoded[:,0]

# HighBP
df1["No_HighBP"] = df1_OneEncoded1[:,0]
df1["Yes_HighBP"] = df1_OneEncoded1[:,1]

#HighChol
df1["No_HighChol"] = df1_OneEncoded2[:,0]
df1["Yes_HighChol"] = df1_OneEncoded2[:,1]



df1.head()

df1.drop(['HighBP','HighChol','Sex','Education','Income','HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
         'HvyAlcoholConsump', 'GenHlth', 'PhysHlth', 'DiffWalk', 'Age'
         ,'No_HighBP', 'Yes_HighBP', 'No_HighChol', 'Yes_HighChol'],axis=1,inplace=True)
df1.head()

df1.info()

# Removing duplicate rows from the dataset
df1.drop_duplicates(inplace = True)
duplicate = df1[df1.duplicated()]
print("Duplicate Rows : ", len(duplicate))
df1['Diabetes_012'].value_counts()
# over sampling of the dataset to get a balanced dataset
class_0 = df1[df1['Diabetes_012'] == 0]
class_1 = df1[df1['Diabetes_012'] == 1]
class_2 = df1[df1['Diabetes_012'] == 2]

# over sampling of the minority class 1
class_1_over = class_1.sample(len(class_0), replace=True)
class_2_over = class_2.sample(len(class_0), replace=True)

# Creating a new dataframe with over sampled class 1 df and class 0 df
df1 = pd.concat([class_2_over,class_1_over, class_0], axis=0)

# plotting the new label distribution
df1['Diabetes_012'].value_counts().plot(kind='bar', title='Label Distribution after Oversampling',
                                           color=['#F4D03F','#D35400','#DE3163'])

df1['Diabetes_012'].value_counts()

corr_matrix= df1.corr()
fig, ax= plt.subplots(figsize=(22,10))
ax= sns.heatmap(corr_matrix,annot=True,linewidths=0.5, fmt=".2f", cmap="YlGn");
bottom, top=ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5);
df1.info()

df1.drop(['Male','Female'], axis=1, inplace=True)
df1.head()
df1.drop(['CholCheck','AnyHealthcare','MentHlth','NoDocbcCost'], axis=1,inplace=True)
df1.head()
df1.info()

df1.drop(['Diabetes'],axis=1,inplace=True)

# X = df1.drop("Diabetes_012", axis = 1)
# y = df1["Diabetes_012"]
X = df1.drop("BMI", axis = 1)
y = df1["BMI"]
print('Shape of X = ', X.shape)
print('Shape of y = ', y.shape)

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=42)
print('Shape of X_train = ', X_train.shape)
print('Shape of y_train = ', y_train.shape)
print('Shape of X_test = ', X_test.shape)
print('Shape of y_test = ', y_test.shape)

sc = StandardScaler()

# Fitting Standard Scaller
X_train = sc.fit_transform(X_train)

# Scaling data
X_test = sc.transform(X_test)

#Fitting DecisionTreeClassifier Model
accuracies = {}
#Fitting RandomForestClassifier Model
classifier = RandomForestClassifier(criterion= 'gini', n_estimators= 200, random_state= 51)
classifier.fit(X_train, y_train)



pickle.dump(classifier,open("model.pkl","wb"))
print("done")