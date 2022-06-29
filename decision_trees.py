import pandas as pd
df=pd.read_csv('loan.csv')

#drop the id column bc it's not important here
df=df.drop('Loan_ID',axis=1)

import seaborn as sns

#to keep simple, drop missing values the rerun heatmap
df=df.dropna()
sns.heatmap(df.isnull())

#get dummies
df=pd.get_dummies(df,drop_first=True)

##split into x and y 
x=df.drop("Loan_Status_Y",axis=1)
y=df["Loan_Status_Y"]

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split=train_test_split(x,y,test_size=0.3,random_state=41)

#build decision tree
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)

#evaluate them
##training set 

from sklearn.metrics import f1_score
y_pred_train=dt.predict(x_train)
f1_score(y_train,y_pred_train)

#testing set
y_pred_test=dt.predict(x_test)
f1_score(y_test,y_pred_test)

#plot the tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plot_tree(dt)
plt.show()

###max depth of the tree 
dt.tree_.max_depth
##this tree is large with a high max depth so it is overfitting the data. we will need to cut down depth

#we can use grid search help improve the tree
parameter_grid={"max_depth":range(2,16),"min_samples_split":range(2,6)}

from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(dt,parameter_grid,verbose=3,scoring="f1")

#fit grid
grid.fit(x_train,y_train)

#best parameters are
grid.best_params_

#lets use optimized parameters to see if model performance improves 
dt=DecisionTreeClassifier(max_depth=2,min_samples_split=2)
dt.fit(x_train,y_train)

from sklearn.metrics import f1_score
y_pred_train=dt.predict(x_train)
print("training score is",f1_score(y_train,y_pred_train))

y_pred_test=dt.predict(x_test)
print("testing score is",f1_score(y_test,y_pred_test))


plt.figure(figsize=(10,10))
plot_tree(dt)
plt.show()


#lets try to see if random forests help

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=500)

rfc.fit(x_train,y_train)


#how the model perfoms in training and testing
#training
y_pred_train=rfc.predict(x_train)
print("training score is",f1_score(y_train,y_pred_train))
#testing 
y_pred_test=rfc.predict(x_test)
print("testing score is",f1_score(y_test,y_pred_test))


#excercise: do grid search on rfc to improve the model 
parameter_grid={"max_depth":range(2,16),"min_samples_split":range(2,6)}

from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(rfc,parameter_grid,verbose=3,scoring="f1")

#fit grid
grid.fit(x_train,y_train)

#best parameters are
grid.best_params_

#lets use optimized parameters to see if model performance improves 
dt=DecisionTreeClassifier(max_depth=2,min_samples_split=2)
dt.fit(x_train,y_train)

from sklearn.metrics import f1_score
y_pred_train=rfc.predict(x_train)
print("training score is",f1_score(y_train,y_pred_train))

y_pred_test=rfc.predict(x_test)
print("testing score is",f1_score(y_test,y_pred_test))


plt.figure(figsize=(10,10))
plot_tree(dt)
plt.show()











