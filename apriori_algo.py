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
change_array = []
#loop through dataframe to find the daily change
for row in df1:
    change = abs_daily_variance(open = df1['Open'], low=df1['Low'],height=df1['High'], close=df1['Close'] )
    #add calculation to the end cell in the row
    df1['daily_change'] = change
    
print(df1)
df1 = df1.drop(columns=['Date', 'Adjusted Close'])
#add category information on end
#group stock by it's moving amount 1=1-5$, 2=6-10$, 3=10-15$,4=16-20$,5=21-25$,6=26-35$,7=36-50$,8=>50$   
#use pandas cut to categorize into 'bins'
#df1['Group'] = pd.cut(df1['daily_change'],bins=[1,6,11,16,21,26,36,50], labels=False, retbins=True,include_lowest=True)
df1['Group'] = pd.cut(df1['daily_change'],500)
df1 = df1.round(1)
df1['Volume'] = pd.cut(df1['Volume'],500)

print(df1)
#print(df_updated)



#create new dataframe without date and adjusted close


df1 = df1.drop(columns=['Low', 'Open','High','Close'])
print(df1)

items = set()
for col in df1:
    items.update(df1[col].unique())
#print(items)


#items = df1['Volume']

#hot  encode the dataset
itemset = set(items)
encoded_vals = []
for index, row in df1.iterrows():
    rowset = set(row) 
    labels = {}
    uncommons = list(itemset - rowset)
    commons = list(itemset.intersection(rowset))
    for uc in uncommons:
        labels[uc] = 0
    for com in commons:
        labels[com] = 1
    encoded_vals.append(labels)
encoded_vals[0]
ohe_df = pd.DataFrame(encoded_vals)
print(ohe_df)

#import mlxtend for apriori
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
freq_items = apriori(ohe_df, min_support=0.05, use_colnames=True, verbose=1)

freq_items.head(7)
print(freq_items)


rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)
print(rules.head())