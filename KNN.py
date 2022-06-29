#import all relevant libraries 
import pandas as pd
df=pd.read_csv('heart-1.csv')

df1=df.reshape(1,-1)

import seaborn as sns
sns.pairplot(df)

#using min max scaler

#separate into x and y 
x=df['target']
y=df.drop('target',axis=1)

#step 1: split into test and train 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)

#step 2: standardize the x
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train_scaled=scaler.transform(x_test)


#do knn modeling 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

knn=KNeighborsClassifier
knn.fit(x_train_scaled,y_train)

#predictions 
y_pred=knn.predict(x_test_scaled)

#f1
f1_score(y_test,y_pred)
