import pandas as pd
df=pd.read_csv('baseball.csv')

#separate x and y
x=df[['Hits','Years']]
y=df["Salary"]

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.3)

#get the model
from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
dtr.fit(x_train,y_train)

#make predictions
y_pred_dtr=dtr.predict(x_test)

#evaluate the model
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test, y_pred_dtr)
rmse=mse**0.5

#visualize the tree
from sklearn.tree import plot_tree
plot_tree(dtr)

#exercise:tune the tree using grid search,improve the tree,find hyper parameters 

parameter_grid={"max_depth":range(2,16),"min_samples_split":range(2,6)}

from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(dtr,parameter_grid,verbose=3,scoring="f1")

grid.fit(x_train,y_train)

grid.best_params_

#optimized parameters to see if model performance improves 
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(max_depth=2,min_samples_split=2)
dt.fit(x_train,y_train)

from sklearn.metrics import f1_score
y_pred_train=dt.predict(x_train)
print("training score is",f1_score(y_train,y_pred_train))

y_pred_test=dt.predict(x_test)
print("testing score is",f1_score(y_test,y_pred_test))

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plot_tree(dt)
plt.show()

#doing the random forests
from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(n_estimators=500)
rfr.fit(x_train,y_train)

y_pred_rfr=rfr.predict(x_test)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test, y_pred_rfr)
rmse=mse**0.5
print(rmse)

from sklearn.ensemble import RandomForestRegressor
