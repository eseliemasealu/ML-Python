import pandas as pd 

y_test=[0,1,0,0,1,0]
y_pred=[0,1,1,1,0,0]

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score,precision_score

C_mat=pd.DataFrame(confusion_matrix(y_test,y_pred,labels=[0,1]),index=["Actual:0","Actual:1"],
                   columns=["Pred:0","Pred:1"])

print (C_mat)

print ("Accccuray is",accuracy_score(y_test,y_pred))
print ("Recall is",recall_score(y_test,y_pred))
print ("Precision is",precision_score(y_test,y_pred))



######################################################


import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('default-2.csv')
sns.boxplot(x='default',y='balance',data=df)

#dummify the data 
df=pd.get_dummies(df,drop_first=True)

#separate the x and y
#y=default, x=balance 
y=df["default_Yes"]
x=df[["balance"]]

##train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)

###linear regression
'''Using regression for classification may not be a good idea'''
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x_train,y_train)

y_pred_lm=lm.predict(x_test)

'''Using logistic regression model'''
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression(solver="liblinear")
logmodel.fit(x_train,y_train,y_train)

##make predictions
y_pred_log=logmodel.predict(x_test)

##to get probability
y_probab=logmodel.predict_proba(x_test)

##coef and intercept: b0 and b1
logmodel.coef_ 
logmodel.intercept_

bing=pd.DataFrame(y_probab)
plt.scatter(x_test,bing[1])

##f1 score of the model 
from sklearn.metrics import confusion_matrix,f1_score

C_mat=pd.DataFrame(confusion_matrix(y_test,y_pred_log,labels=[0,1]),index=["Actual:0","Actual:1"],
                   columns=["Pred:0","Pred:1"])

print(C_mat)

f1_score(y_test,y_pred_log)

'''Out[3]: 0.6711864406779661'''

###############can you run model for all the x variables? 

###students

#separate the x and y
#y=default, x=balance 
y=df["default_Yes"]
x=df[["student_Yes"]]

##train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)

###linear regression
'''Using regression for classification may not be a good idea'''
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x_train,y_train)

y_pred_lm=lm.predict(x_test)

'''Using logistic regression model'''
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression(solver="liblinear")
logmodel.fit(x_train,y_train,y_train)

##make predictions
y_pred_log=logmodel.predict(x_test)

##to get probability
y_probab=logmodel.predict_proba(x_test)

##coef and intercept: b0 and b1
logmodel.coef_ 
logmodel.intercept_

bing=pd.DataFrame(y_probab)
plt.scatter(x_test,bing[1])

##f1 score of the model 
from sklearn.metrics import confusion_matrix,f1_score

C_mat=pd.DataFrame(confusion_matrix(y_test,y_pred_log,labels=[0,1]),index=["Actual:0","Actual:1"],
                   columns=["Pred:0","Pred:1"])

print(C_mat)

f1_score(y_test,y_pred_log)

###income

y=df["default_Yes"]
x=df[["income"]]

##train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)

###linear regression
'''Using regression for classification may not be a good idea'''
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x_train,y_train)

y_pred_lm=lm.predict(x_test)

'''Using logistic regression model'''
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression(solver="liblinear")
logmodel.fit(x_train,y_train,y_train)

##make predictions
y_pred_log=logmodel.predict(x_test)

##to get probability
y_probab=logmodel.predict_proba(x_test)

##coef and intercept: b0 and b1
logmodel.coef_ 
logmodel.intercept_

bing=pd.DataFrame(y_probab)
plt.scatter(x_test,bing[1])

##f1 score of the model 
from sklearn.metrics import confusion_matrix,f1_score

C_mat=pd.DataFrame(confusion_matrix(y_test,y_pred_log,labels=[0,1]),index=["Actual:0","Actual:1"],
                   columns=["Pred:0","Pred:1"])

print(C_mat)

f1_score(y_test,y_pred_log)











