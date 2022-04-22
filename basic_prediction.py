import numpy as np 
import pandas as pd 
import datetime as dt

#################

import time
from datetime import datetime
import plotly_express as px
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

######################

"""
helper function to better understand every operation we are apply
"""
def apply_func(function,stocks):
    for s in stocks:
        function(s.tail(10))
'''
function to load stock data from the path
'''
def load_stock(name,path):
    df = pd.read_csv(r'forbes2000\csv\AAL.csv')
    return df

 
'''
we chosed to examine 5 popular stocks from usa main stock market(NYSE)
1.American Airlines
2.Ebay
3.General Electric
4.Amazon
5.Bank of America
'''
names=[("AAL","nasdaq"),("EBAY","sp500"),("GE","nyse"),("AMZN","sp500"),("BA","sp500")]
stocks=[load_stock(n[0],n[1])for n in names]


################################


apply_func(print,stocks)
df=stocks[0]

for i,stock in enumerate(stocks):
    print(names[i][0])
    print(stock.info())
    print(stock.isnull().sum())
    print(stock.describe())

"""
before feature creation there is no null values and the data is pretty clean.
"""



##################

ma_days = [7,10,14,21,50,100]
maxi_days=[7,30,365,730]
# create features that calulates moving average, maximum values for n days, ,minimum values for n days, std of last 7 days

"""
average for x days
""" 
def calculate_average(df,ma_days):
    for ma in ma_days:
        column_name = "MA for %s days" %(str(ma))
        df.loc[:,column_name]=pd.DataFrame.rolling(df['Close'],ma).mean()
        
"""
maximum for x days
""" 
def calculate_maximum(df,ma_days):
    for ma in maxi_days:
        column_name = "Maximum for %s days" %(str(ma))
        df.loc[:,column_name]=pd.DataFrame.rolling(df['Close'],ma).max()
"""
minimum for x days
""" 
def calculate_minimum(df,ma_days):
    for ma in maxi_days:
        column_name = "Minimum for %s days" %(str(ma))
        df.loc[:,column_name]=pd.DataFrame.rolling(df['Close'],ma).min()

calculate_average(df,ma_days)
calculate_maximum(df,maxi_days)
calculate_minimum(df,maxi_days)
df.loc[:,"std for 7 days"]=pd.DataFrame.rolling(df['Close'],7).std()
#add difference high-low daily feature
df.loc[:,'Diff High Low']=df['High']-df['Low']
#add diff open-close feature
df.loc[:,'Diff Open Close']=df['Open']-df['Close']
df.loc[:,'Daily Return'] = df['Close'].pct_change()*100

print("number of features {}".format(len(df.columns)))
df.columns


##################

from sklearn.preprocessing import MinMaxScaler
def normalize(df):
    scaler = MinMaxScaler(feature_range=(0,1))
    df = scaler.fit_transform(df)
    df=pd.DataFrame(df)
    return df

#################


"""
we decided to chose the last 6 years for the data set. 
That decision was made Based on articles that we read and empirical  tests.
"""

for index,stock in enumerate(stocks):
    stock['Date'] = pd.to_datetime(stock['Date'],dayfirst=True)
    start_date=pd.to_datetime("01/01/2016")
    end_date=pd.to_datetime("01/01/2022")
    stock = stock[(stock["Date"] >= start_date) & (stock["Date"] <= end_date)]
    stock.set_index('Date', inplace=True)
    stocks[index]=stock
df=stocks[0]


#################

y = df['Close'][1:] #target column -close value
df.loc[:,'Close previous']=df['Close']
df=df.drop(['Close'],axis=1)
df=df.drop(['Adjusted Close'],axis=1)
df=df.shift(periods=1)[1:]

# we decided not to normalize because the result's performance decreased 
# df=normalize(df)


# print(x_train.head(30))
'''print(df.isnull().sum())
print(y.isnull().sum)
'''
###################

stocks_close=[stock["Close"] for stock in stocks]
close_show = pd.concat(stocks_close, axis=1, keys=names)
axes =close_show.plot(marker='.', alpha=0.7, linestyle='None', figsize=(15, 9), subplots=True)

"""
plotting the close value of each stock that we analyzed
"""

from sklearn.model_selection import train_test_split

'''
because our model is time series, shuffle must be false because every value depened on the previous one
'''

x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.25, shuffle=False)


from sklearn.feature_selection import SelectKBest,f_classif,f_regression,mutual_info_regression

#feature selection for from train data!
import matplotlib.pyplot as plt
import seaborn as sns

##todo how much features to select for the model##
def select_k_features(func,k=8,x_train=x_train,y_train=y_train):
    #apply SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(score_func=func, k=5)
    fit = bestfeatures.fit(x_train,y_train)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(x_train.columns)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    featureScores.nlargest(10,'Score').plot.bar(x="Specs",y="Score",figsize=(6,5),color="purple")
    plt.title("f regression")
    plt.show()
    best_10=list(featureScores.nlargest(5,'Score')['Specs'])
    for col in x_train:
        if col not in best_10:
            x_train.drop([col], axis=1, inplace=True)
    return x_train

x_train=select_k_features(func=f_regression)

print(x_train)
x_test=x_test[x_train.columns]
x_train.columns

'''plt.figure(figsize=(20,8))
sns.heatmap(x_train.corr(),cmap=plt.cm.Blues,annot=True)
plt.title('Heatmap displaying the relationship between the features of the data',
         fontsize=13)
plt.show()'''


import sklearn.metrics
import math
from sklearn.model_selection import TimeSeriesSplit

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



def train_test_and_measure(model_name,model,x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test):
        
    model=model.fit(x_train,y_train)
    prediction = model.predict(x_test)
    mse = sklearn.metrics.mean_squared_error(y_test,prediction)
    rmse = math.sqrt(mse)
    MBE = np.mean(prediction - y_test)
    print("Test set MAPE: {} ".format(model_name), mean_absolute_percentage_error(y_test,prediction))
    print("Test set RMSE: {} ".format(model_name), rmse)
    print("Test set MBE: {} ".format(model_name), MBE)
    print("r^2: {} ".format(model.score(x_test,y_test)))
    plot_df_train=pd.DataFrame(y_train)
    plot_df=pd.DataFrame(y_test)
    plot_df["predictions"]=prediction
    print(plot_df[-5:])
    plt.figure(figsize=(18,5))
    plt.plot(plot_df_train[-360:])
    plt.plot(plot_df)
    plt.show()    
    plot_df=pd.DataFrame(y_test)
    plot_df["predictions"]=prediction
    plot_df[-360:].plot(legend=True, figsize = (15, 8))
    plt.show()
    
def mlp_model(X, Y):
    estimator=MLPRegressor(learning_rate_init=0.002)
    param_grid = {'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,1)],
              'activation': ['relu','tanh','logistic'],
              'alpha': [0.0001, 0.05],
              'learning_rate': ['constant','adaptive'],
              'solver': ['adam']}
    gsc = GridSearchCV(
        estimator,
        param_grid,
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    grid_result = gsc.fit(X, Y)
    best_params = grid_result.best_params_
    print(best_params)
    return best_params

from sklearn.neural_network import MLPRegressor

from sklearn.neural_network import MLPRegressor

"""
create models:
1. Linear Regression
2.Random Forest Regresson
3.Support Vector Regression 
4.Gradient boosting regressor
5.Mlp Regressor
"""
regression_model = LinearRegression()
randomForestRegressor_model = RandomForestRegressor(max_depth=6, random_state=1)
svr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
gradientReg = GradientBoostingRegressor(random_state=0)
MLPRegressor=MLPRegressor(hidden_layer_sizes=(50, 50, 50), activation='relu',  solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=300, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)



#notes
# randomForestRegressor_model = RandomForestRegressor(n_estimators=800,min_samples_split=10, min_samples_leaf=2, max_depth=None,bootstrap=False)
# mlp_model(x_train,y_train)
# {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50, 50), 'learning_rate': 'constant', 'solver': 'adam'}
# MLPRegressor=MLPRegressor(hidden_layer_sizes=(200,), activation='tanh',  solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=300, shuffle=False, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
# MLPRegressor=MLPRegressor(hidden_layer_sizes=(100,), activation='logistic',  solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)


tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(df):
    df=df[['Close previous']]
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = df.loc[df.index[train_index]], df.loc[df.index[test_index]]
    y_train, y_test = y.loc[y.index[train_index]], y.loc[y.index[test_index]]
    print(y_train)
    train_test_and_measure("MLPRegressor_model",MLPRegressor,x_train,x_test,y_train,y_test)


