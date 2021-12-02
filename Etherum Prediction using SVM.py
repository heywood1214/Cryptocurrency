#Description this program attempts to predict the future price of ETH
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from pandas_datareader import data as web
from datetime import datetime

plt.style.use('five thirtyeight')
today = datetime.today().strftime('%Y-%m-%d')
CryptoStartDate ='2018-01-01'

#load ETH data
df_Eth=web.DataReader('ETH-CAD','yahoo',start=CryptoStartDate, end=today)

print(df_Eth)

future_days = 5

#Create a new column that contains future price
df_Eth[str(future_days)+'_Day_Price_Forecast']=df_Eth[['Close']].shift(-future_days)

#Show the data
df_Eth[['Close',str(future_days)+'_Day_Price_Forecast']]

#independent data set
X=np.array(df_Eth[['Close']])

#last n rows of data
X=X[:df_Eth.shape[0]-future_days]
print(X.shape)

y = np.array(df_Eth[str(future_days)+'_Day_Price_Forecast'])
y = y[:-future_days]


#Split the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size = 0.2 ) 

from sklearn.svm import SVR
svr_rbf = SVR(kernel='rbf',C=1e3, gamma = 0.00001)
svr_rbf.fit(x_train,y_train)

svr_rbf_confidence = svr_rbf.score(x_test,y_test)
print('svr_rbf accuracy: ',svr_rbf_confidence)

#print predicted values and compare with actual values

svm_prediction = svr_rbf.predict(x_test)
print(svm_prediction)

print(y_test)

plt.figure(figsize=(12,4))
plt.plot(svm_prediction,label='Prediction',lw=2, alpha = 0.7)
plt.plot(y_test,label='Actual',lw=2, alpha = 0.7)
plt.title('Prediction vs Actual')
plt.ylabel('Price in CAD')
plt.xlabel('Time')
plt.legend()
plt.xticks(rotation = 45)
plt.show()