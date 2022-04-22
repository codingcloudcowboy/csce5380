#this is for importing the data from each CSV to then help find the variance of the day
#I am determining the movement of the stock of the day
#Take the opening price, to the high and to the close. This will tell you how much it moved throughout the day
#Rough formula abs(high-open)+ abs(close-height)

#steps
#import pandas to then create dataframe with CSV of the stock
#calculate the daily change
#add change to the dataframe for apriori calculation


from tkinter.filedialog import Open
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm



#create calculation for finding daily variance of stock
#this could also be used to find volatility 
def abs_daily_variance(open, low, height, close):
    return (abs(open-low)+abs(height-low)+abs(close-height))


#import CSV with pandas
df1 = pd.read_csv('forbes2000\csv\ADSK.csv')

'''print(df1.head)

print(df1.columns)

print(df1['Date'])'''
#array for daily change to then add to the dataframe

df1['Buy'] = True 
print(df1)
#df1 = df1.drop(columns=['Date', 'Adjusted Close'])
#add category information on end
#group stock by it's moving amount 1=1-5$, 2=6-10$, 3=10-15$,4=16-20$,5=21-25$,6=26-35$,7=36-50$,8=>50$   
#use pandas cut to categorize into 'bins'
#df1['Group'] = pd.cut(df1['daily_change'],bins=[1,6,11,16,21,26,36,50], labels=False, retbins=True,include_lowest=True)

'''for row in df1:
    df1['Spread'] = df1['Open'] - df1['Close']
    if (df1['Spread'])>0:
        #df1['Buy'] = True
        print('high')
    else:
        #df1['Buy'] = False
        print('no')'''

df1['Spread'] = df1['Open'] - df1['Close']
df1.loc[df1['Spread'] > 0, 'Buy'] = 1 
df1.loc[df1['Spread'] <= 0, 'Buy'] = 0 

df1 = df1.drop(columns=['Date','Volume']) #'Adjusted Close','Spread' Dropped volume and accuracy skyrocketed to 88%
df1 = df1.round(1)
df1 = df1.dropna()
print(df1)


cols = df1.columns.tolist()
cols = cols[-1:] + cols[:-1]
print(cols)
df1 = df1[cols]
print(df1)
df1 = df1.dropna()
#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#break up the data to training set
# Import train_test_split function

from sklearn.model_selection import train_test_split

df1 = df1.sample(n=9000)
#X_cols = (df1.loc[:, df1.columns != 'Buy'], axis = 1)
X = df1.loc[:, df1.columns != 'Buy' ]
Y = df1['Buy']
Y= Y.astype('int')

# Split dataset into training set and test set
#X_train, X_test, y_train, y_test = train_test_split(df1, test_size=0.3) # 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2, random_state=10)# 70% training and 30% test

import time
t0 = time.time()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
#Import svm model
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

#Create a svm Classifier
KNN_model = KNeighborsClassifier(n_neighbors=2)

#Train the model using the training sets
KNN_model.fit(X_train, y_train)
#Predict the response for test dataset
KNN_prediction = KNN_model.predict(X_test)



#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.metrics import accuracy_score

# Model Accuracy: how often is the classifier correct?
print(accuracy_score(KNN_prediction, y_test))










t1 = time.time()

total = t1-t0
print(total)