import pandas as pd 
df=pd.read_csv('AmesHousing.csv')
df.info()

##dropping all id columns 
df=df.drop(['Order','PID'],axis=1)

##dropping all rows with missing values 
df=df.dropna()

##regression model to predict sales price of house 
#separate x and y
x=df.drop('SalePrice',axis=1)
y=df['SalePrice']

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

from sklearn.linear_model import LinearRegression
lm=LinearRegression()

lm.fit(x_train,y_train)


prediction=lm.predict(x_test)

from sklearn.metrics import r2_score
print('r square is',r2_score(y_test,prediction))

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,prediction)
rmse=mse**0.5
print('rmse is',rmse)

#result without feature selection
'''r square is 0.8451202107848694
rmse is 31796.36504263581'''

#can the model be improved with feature selection 
##feature selection using sklearn

from sklearn.feature_selection import SelectKBest,f_regression
bestfeatures=SelectKBest(score_func=f_regression,k=5)
new_x=bestfeatures.fit_transform(x,y)


##run regesstion on selected five features and see if the model improves

x_train,x_test,y_train,y_test=train_test_split(new_x,y,test_size=0.3,random_state=1)


lm=LinearRegression()

lm.fit(x_train,y_train)


prediction=lm.predict(x_test)

print('r square is',r2_score(y_test,prediction))


mse=mean_squared_error(y_test,prediction)
rmse=mse**0.5
print('rmse is',rmse)

'''with selected 5 features 
r square is 0.8147972106285185
rmse is 34769.93361932471'''



#exercise
##run loop to find best values of K based on rmse
 
for i in (1,28):
    bestfeatures=SelectKBest(score_func=f_regression,k=i)
   
    x_train_new=bestfeatures.fit_transform(x,y)
    x_train,x_test,y_train,y_test=train_test_split(new_x,y,test_size=0.3,random_state=1)
    lm=LinearRegression()
    lm.fit(x_train,y_train)
    print('The intercepts is',lm.intercept_)
    print('The coefficients are',lm.coef_)
    print('The coefficients are',lm.coef_.tolist())
    predictions=lm.predict(x_test)
    print('r square is',r2_score(y_test,predictions))
    print("r square is",r2_score(y_test, predictions)) 
    mse=mean_squared_error(y_test,predictions)
    rmse=mse**0.5
    print ("rmse is",rmse)
    print(i)
    



















