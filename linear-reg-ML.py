import pandas as pd 
df=pd.read_csv("train_test.csv")

#first thing we can do is separate data into x and y variables 
x=df[['Bedrooms','Sq.Feet','Neighborhood']]
y=df[['Price']]

#next divide x and y into test and train data 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test,=train_test_split(x,y,test_size=0.3,random_state=42)
#what we just did: 
    #we had data that we split into x and y data
    #then we used sklearn model to split data into train and test. 
        #test size here is 30% of the data
        #the random state as 42 is so that we all have the same random number. 42 is like setting a seed
            #it can be any number really 
            

import pandas as pd #importing all functionalities
df=pd.read_csv('Advertising-1.csv')
#sales(units) is the y variable. that's what we're supposed to predict 

import seaborn as sns
sns.pairplot(df)

#simple linear regression
    #x is TV, y is sales 
#separating into x and y variables 
x=df[['TV','radio','newspaper']]#ALWAYS use double brackets for x
y=df['sales']


#lets do splitting into training and testing 
    #test size=0.3
    #random state 101

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)

#now on to the regession 
from sklearn.linear_model import LinearRegression
lm=LinearRegression() #initialize the model, this is not trained yet 

lm.fit(x_train,y_train)#training the model

##what is the model (coefficients and intercept)
lm.coef_
lm.intercept_

##first make predictions
predictions=lm.predict(x_test)
#for any number would be:


##evaluate the model by comparing y_test with predictions
##what exactly is y_test? it is the actual values 
##preditions are y hat
#to evaluate a metric, we need to compare the actual values with yhat values
from sklearn.metrics import r2_score,mean_squared_error
print('R2 is:',r2_score(y_test,predictions))

mse=mean_squared_error(y_test,predictions)
print('rmse is:',mse**0.5)
 
'''Simple linear regressiong results
R2 is: 0.6345141851817353
rmse is: 3212.2352689386476'''

#rmse is our preferred metric. we want it to be as low as possible 

'''for multiple regression and using additional variables
R2 is: 0.9185780903322446
rmse is: 1516.1519375993878'''
#going by this result, we can see that the model improved 



#get auto.csv data and fit a regression such that
#x is horsepower
#y is mpg
#use test size of 0.3, random state of 1
#find rmse and R2

import pandas as pd 
df=pd.read_csv('Auto (1).csv')

import seaborn as sns
sns.pairplot(df)

x=df[['horsepower']]
x["horse_sq"]=df[['horsepower']]**2
y=df['mpg']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)


from sklearn.linear_model import LinearRegression
lm=LinearRegression()

lm.fit(x_train,y_train)

lm.coef_
lm.intercept_

predictions=lm.predict(x_test)

from sklearn.metrics import r2_score,mean_squared_error
print('R2 is:',r2_score(y_test,predictions))

mse=mean_squared_error(y_test,predictions)
print('rmse is:',mse**0.5)

'''Results
R2 is: 0.598509498272342
rmse is: 5.234568588823988'''

'''using horse_sq
R2 is: 0.683726726758932
rmse is: 4.6459497017563836'''

#class excercise 
#model 1
import pandas as pd
df1=pd.read_csv('shark_attacks-1.csv')

import seaborn as sns
sns.pairplot(df1)

x=df1[['IceCreamSales']]
y=df1['SharkAttacks']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

from sklearn.linear_model import LinearRegression
lm=LinearRegression()

lm.fit(x_train,y_train)

lm.coef_
lm.intercept_

predictions=lm.predict(x_test)

from sklearn.metrics import r2_score,mean_squared_error
print('R2 is:',r2_score(y_test,predictions))

mse=mean_squared_error(y_test,predictions)
print('rmse is:',mse**0.5)

'''R2 is: -0.007154544483181757
rmse is: 7.693580942486933'''



#model 2
x=df1[['IceCreamSales','Temperature']]
y=df1['SharkAttacks']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

from sklearn.linear_model import LinearRegression
lm=LinearRegression()

lm.fit(x_train,y_train)

lm.coef_
lm.intercept_

predictions=lm.predict(x_test)

from sklearn.metrics import r2_score,mean_squared_error
print('R2 is:',r2_score(y_test,predictions))

mse=mean_squared_error(y_test,predictions)
print('rmse is:',mse**0.5)

'''R2 is: 0.3091833833733332
rmse is: 6.371795906949918'''


#model 3
x=df1[['Temperature']]
y=df1['SharkAttacks']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

from sklearn.linear_model import LinearRegression
lm=LinearRegression()

lm.fit(x_train,y_train)

lm.coef_
lm.intercept_

predictions=lm.predict(x_test)

from sklearn.metrics import r2_score,mean_squared_error
print('R2 is:',r2_score(y_test,predictions))

mse=mean_squared_error(y_test,predictions)
print('rmse is:',mse**0.5)

'''R2 is: 0.4045252557746677
rmse is: 5.915781735360626'''