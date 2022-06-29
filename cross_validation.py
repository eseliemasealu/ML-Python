import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier

df=pd.read_csv('cancer.csv')
x=df.drop('target',axis=1)
y=df['target']

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

#get the adaboost
adb=AdaBoostClassifier(n_estimators=100)
adb.fit(x_train,y_train)

#make predictions
y_pred=adb.predict(x_test)

#get the f1
from sklearn.metrics import f1_score
f1_score(y_test,y_pred)

#can you get f1 score using cross validation?
#use cv=10

from sklearn.model_selection import cross_val_score
adb_score=cross_val_score(adb,x,y,scoring='f1',cv=10,verbose=3)
adb_score.mean()

df=pd.read_csv('cancer.csv')
x=df.drop('target',axis=1)
y=df['target']

#initialize the 3 models and their combos
model1=LogisticRegression(solver='liblinear')
model2=SVC()
model3=DecisionTreeClassifier()
model_combo=VotingClassifier(estimators=[( 'lrp',model1 ),('sv',model2),('dt',model3)])

#using cross val to evaluate the models 
model1_score=cross_val_score(model1,x,y,scoring='f1',cv=10,verbose=3)#cv 10 is for running the model 10 times
print('the score is',model1_score.mean())


#for model2
model2_score=cross_val_score(model2,x,y,scoring='f1',cv=10,verbose=3)
print('the score is',model2_score.mean())

#for model3
model3_score=cross_val_score(model3,x,y,scoring='f1',cv=10,verbose=3)
print('the score is',model3_score.mean())

##for combo model
model_combo_score=cross_val_score(model_combo,x,y,scoring='f1',cv=10,verbose=3)
print('the score is',model_combo_score.mean())

#so this is one way to do an ensemble of voting classifier. The improvement is not guaranteed but it usually works