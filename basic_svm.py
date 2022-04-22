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

    
print(df1)
#df1 = df1.drop(columns=['Date', 'Adjusted Close'])
#add category information on end
#group stock by it's moving amount 1=1-5$, 2=6-10$, 3=10-15$,4=16-20$,5=21-25$,6=26-35$,7=36-50$,8=>50$   
#use pandas cut to categorize into 'bins'
#df1['Group'] = pd.cut(df1['daily_change'],bins=[1,6,11,16,21,26,36,50], labels=False, retbins=True,include_lowest=True)


print(df1.describe())
#add some features to the data to reference for classification
lags = 2
for i in range(0,lags):
    df1['Lag%s' % str(i+1)] = df1['Close'].shift(i+1).pct_change()
for row in df1:
    df1['Open-Close'] = (df1['Open'] - df1['Close']).pct_change()
    df1['High-Low'] = (df1['High'] - df1['Low']).pct_change()
    df1['volume_gap'] = df1['Volume'].pct_change()


df1 = df1.dropna()

print(df1)

#shift -1 for next day's return
'''df1['y_clas'] = df1['Close'].shift(-1)/df1['Open'].shift(-1)-1'''
#if tomorrow's return > 0 then return 1; If tomorrow's return <= 0 then return -1
'''df1['y_clas'] = -1
df1.at[df1['forward_ret']>0.0, 'y_clas'] = 1
#remove it to make ensure no look ahead bias


df1 = df1.drop(columns=['forward_ret'])'''



'''
plt.figure(figsize=(8,4))
sns.countplot('y_clas',data=df1)
plt.title('Target Variable Counts')
plt.show()'''
df1 = df1.drop(columns=['Date','Volume'])

print(df1.columns)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#break up the data to training set

data = df1
data = data.dropna(inplace=True)

X = data
y_clas = data['Close']

SP = 0.8
split = int(SP*len(data))

#Train the data
xTrain = X[:split]; yTrain = y_clas[:split]
#Test the data
xTest = X[split:]; yTest = y_clas[split:]

print('Observations: %d' % (len(xTrain) + len(xTest)))
print(' Training Observations: %d' % (len(xTrain)))
print('Testing Observations: %d' % (len(xTest)))


regr = svm.SVR()
regr.fit(X,y_clas)

