import pandas as pd
df=pd.read_csv("cancer-2.csv")

##separate into x and y
x=df.drop(["target"],axis=1)
y=df["target"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

##get the model
from sklearn.svm import SVC
model=SVC()
model.fit(x_train,y_train)

##make predictions
y_pred=model.predict(x_test)

##evaluate 
from sklearn.metrics import f1_score,recall_score
f1_score(y_test,y_pred)
recall_score(y_test,y_pred)

##grid search to find best values of C, gamma, kernel

param_grid={'C':[1,10,100],"gamma":[1,0.1,0.01],"kernel":["rbf","linear"]}

from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(model,param_grid,verbose=3,scoring="f1") #this is initializing the grid 

grid.fit(x_train,y_train)

grid.best_params_

#with tuned parameters 
model=SVC(C=1,gamma=1,kernel="linear")
model.fit(x_train,y_train)

##make predictions
y_pred=model.predict(x_test)

##evaluate 
from sklearn.metrics import f1_score,recall_score
f1_score(y_test,y_pred)
recall_score(y_test,y_pred)
