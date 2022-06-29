"""
Created on Wed Apr  7 15:10:08 2021

@author: srtherri
"""

##Cleaning##



import pandas as pd
df_raw=pd.read_csv("Mental Health Data updated label2.csv")

#summary of raw data
df_raw.info()

#company size cleaning#
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

df_raw.number_of_employees = df_raw.number_of_employees.fillna('self')

df=df_raw.loc[df_raw["self_employed"]==0]

#tech role nan conversion to 0 because if the answer was left blank
#we can assume that the people is not a tech role, same for tech company#

df.tech_company = df.tech_company.fillna(0)

df.tech_role = df.tech_role.fillna(0)

#fill in "I am not sure" for people who did not answer the know mhb options question#

df.know_opt_mhb = df.know_opt_mhb.fillna("I am not sure")

#delete un-used columns that were only for self employed#

df=df.drop("know_local_online_rss_mh",axis=1)
df=df.drop("percent_effect_mh",axis=1)
df=df.drop("mh_effect_productivity",axis=1)

#mental health condition nan to none#

df.mh_condition = df.mh_condition.fillna("None")

#replace nan from out of US state with Non-US

df.state = df.state.fillna("Non-US")

df.employment_state  = df.employment_state .fillna("Non-US")

#gender labels and normalizations

df.gender = df.gender.fillna("Other")


#Check that there are no nan#
df.isnull().values.any()

#save data to csv

#df.to_csv('/Users/srtherri/Documents/MSBA/Data Mining/Mental Health Data Cleaned updated.csv')

###############################################################################

#gender normalization

df['gender'].replace(to_replace = ['Male', 'male', 'Male ', 'M', 'm',
       'man', 'Cis male', 'Male.', 'male 9:1 female, roughly', 'Male (cis)', 'Man', 'Sex is male',
       'cis male', 'Malr', 'Dude', "I'm a man why didn't you make this a drop down question. You should of asked sex? And I would of answered yes please. Seriously how much text can this take? ",
       'mail', 'M|', 'Male/genderqueer', 'male ',
       'Cis Male', 'Male (trans, FtM)',
       'cisdude', 'cis man', 'MALE'], value = 'male', inplace = True)
df['gender'].replace(to_replace = ['Female', 'female', 'I identify as female.', 'female ',
       'Female assigned at birth ', 'F', 'Woman', 'fm', 'f', 'Cis female ', 'Transitioned, M2F',
       'Genderfluid (born female)', 'Female or Multi-Gender Femme', 'Female ', 'woman', 'female/woman',
       'Cisgender Female', 'fem', 'Female (props for making this a freeform field, though)',
       ' Female', 'Cis-woman', 'female-bodied; no feelings about gender',
       'AFAB'], value = 'female', inplace = True)
df['gender'].replace(to_replace = ['Bigender', 'non-binary', 'Other/Transfeminine',
       'Androgynous', 'Other', 'nb masculine',
       'none of your business', 'genderqueer', 'Human', 'Genderfluid',
       'Enby', 'genderqueer woman', 'mtf', 'Queer', 'Agender', 'Fluid',
       'Nonbinary', 'human', 'Unicorn', 'Genderqueer',
       'Genderflux demi-girl', 'Transgender woman'], value = 'other', inplace = True)


###############################################################################

#number of employee dummy variables
#1: 1 to 5
#2: 6 to 25
#3: 26 to 99, 26 to 100
#4: 100 to 500
#5: 500 to 1000
#6: 1000+

#1
df['number_of_employees'].replace(to_replace = ['1 to 5'], value = 1, inplace = True)
#2
df['number_of_employees'].replace(to_replace = ['6 to 25'], value = 2, inplace = True)
#3
df['number_of_employees'].replace(to_replace = ['26-99'], value = 3, inplace = True)
df['number_of_employees'].replace(to_replace = ['26-100'], value = 3, inplace = True)
#4
df['number_of_employees'].replace(to_replace = ['100-500'], value = 4, inplace = True)
#5
df['number_of_employees'].replace(to_replace = ['500-1000'], value = 5, inplace = True)
#6
df['number_of_employees'].replace(to_replace = ['More than 1000'], value = 6, inplace = True)


# emp_provide_mhb binary
#1: yes
#2: No, i dont know

#1
df['emp_provide_mhb'].replace(to_replace = ['Yes'], value = 1, inplace = True)
#2
df['emp_provide_mhb'].replace(to_replace = ['No'], value = 2, inplace = True)
df['emp_provide_mhb'].replace(to_replace = ["I don't know"], value = 2, inplace = True)
df['emp_provide_mhb'].replace(to_replace = ['Not eligible for coverage / N/A'], value = 2, inplace = True)


#know_opt_mhb
#1: Yes
#2: No, i am not sure

#1
df['know_opt_mhb'].replace(to_replace = ['Yes'], value = 1, inplace = True)
#2
df['know_opt_mhb'].replace(to_replace = ['No'], value = 2, inplace = True)
df['know_opt_mhb'].replace(to_replace = ["I am not sure"], value = 2, inplace = True)

#difficulty of leave
#1: Very easy
#2: Somewhat easy
#3: Neither easy nor difficult
#4: Somewhat difficult
#5: Very difficult
#6: I don't know

#1
df['leave_difficulty_mhb'].replace(to_replace = ['Very easy'], value = 1, inplace = True)
#2
df['leave_difficulty_mhb'].replace(to_replace = ['Somewhat easy'], value = 2, inplace = True)
#3
df['leave_difficulty_mhb'].replace(to_replace = ['Neither easy nor difficult'], value = 3, inplace = True)
#4
df['leave_difficulty_mhb'].replace(to_replace = ['Somewhat difficult'], value = 4, inplace = True)
#5
df['leave_difficulty_mhb'].replace(to_replace = ['Very difficult'], value = 5, inplace = True)
#6
df['leave_difficulty_mhb'].replace(to_replace = ["I don't know"], value = 1, inplace = True)


#diagnosed_mh to binary
#1: Yes
#2: No

#1
df['diagnosed_mh'].replace(to_replace = ['Yes'], value = 1, inplace = True)
#2
df['diagnosed_mh'].replace(to_replace = ['No'], value = 2, inplace = True)


#Does mh effect work when treated
#1: Often
#2: Sometimes
#3: No mental illness
#4: Rarely
#5: Never

#1
df['mh_effects_work_when_treated'].replace(to_replace = ['Often'], value = 1, inplace = True)
#2
df['mh_effects_work_when_treated'].replace(to_replace = ['Sometimes'], value = 2, inplace = True)
#3
df['mh_effects_work_when_treated'].replace(to_replace = ['Not applicable to me'], value = 3, inplace = True)
#4
df['mh_effects_work_when_treated'].replace(to_replace = ['Rarely'], value = 4, inplace = True)
#5
df['mh_effects_work_when_treated'].replace(to_replace = ['Never'], value = 5, inplace = True)


#Does mh effect work when untreated
#1: Often
#2: Sometimes
#3: No mental illness
#4: Rarely
#5: Never

#1
df['mh_effects_work_when_untreated'].replace(to_replace = ['Often'], value = 1, inplace = True)
#2
df['mh_effects_work_when_untreated'].replace(to_replace = ['Sometimes'], value = 2, inplace = True)
#3
df['mh_effects_work_when_untreated'].replace(to_replace = ['Not applicable to me'], value = 3, inplace = True)
#4
df['mh_effects_work_when_untreated'].replace(to_replace = ['Rarely'], value = 4, inplace = True)
#5
df['mh_effects_work_when_untreated'].replace(to_replace = ['Never'], value = 5, inplace = True)


#Gender dummies
#1: male
#2: female
#3: other

#1
df['gender'].replace(to_replace = ['male'], value = 1, inplace = True)
#2
df['gender'].replace(to_replace = ['female'], value = 2, inplace = True)
#3
df['gender'].replace(to_replace = ['other'], value = 3, inplace = True)


#discuss with employer
#1: yes
#2: No

#1
df['discuss_mhb_emp'].replace(to_replace = ['Yes'], value = 1, inplace = True)
#2
df['discuss_mhb_emp'].replace(to_replace = ['No'], value = 2, inplace = True)
df['discuss_mhb_emp'].replace(to_replace = ["I don't know"], value = 2, inplace = True)


#mhb_rss
#1: yes
#2: No

#1
df['mhb_rss'].replace(to_replace = ['Yes'], value = 1, inplace = True)
#2
df['mhb_rss'].replace(to_replace = ['No'], value = 2, inplace = True)
df['mhb_rss'].replace(to_replace = ["I don't know"], value = 2, inplace = True)


#does the employer take mh seriously
#1: yes
#2: No
#3: I don't know

#1
df['emp_takes_mh_seriously'].replace(to_replace = ['Yes'], value = 1, inplace = True)
#2
df['emp_takes_mh_seriously'].replace(to_replace = ['No'], value = 2, inplace = True)
#3
df['emp_takes_mh_seriously'].replace(to_replace = ["I don't know"], value = 3, inplace = True)


#family history
#1: yes
#2: No
#3: I don't know

#1
df['family_history_mh'].replace(to_replace = ['Yes'], value = 1, inplace = True)
#2
df['family_history_mh'].replace(to_replace = ['No'], value = 2, inplace = True)
#3
df['family_history_mh'].replace(to_replace = ["I don't know"], value = 3, inplace = True)


#past_mh
#1: yes
#2: No

#1
df['past_mh'].replace(to_replace = ['Yes'], value = 1, inplace = True)
#2
df['past_mh'].replace(to_replace = ['No'], value = 2, inplace = True)
df['past_mh'].replace(to_replace = ["Maybe"], value = 2, inplace = True)


#current_mh
#1: yes
#2: No

#1
df['current_mh'].replace(to_replace = ['Yes'], value = 1, inplace = True)
#2
df['current_mh'].replace(to_replace = ['No'], value = 2, inplace = True)
df['current_mh'].replace(to_replace = ["Maybe"], value = 2, inplace = True)


#discuss_mhb_coworks
#1: Yes, Maybe
#2: No

#1
df['discuss_mhb_coworker'].replace(to_replace = ['Yes'], value = 1, inplace = True)
df['discuss_mhb_coworker'].replace(to_replace = ["Maybe"], value = 1, inplace = True)
#2
df['discuss_mhb_coworker'].replace(to_replace = ['No'], value = 2, inplace = True)


#discuss_mh_family
#1: Very open, somewhat Open, Neutral
#2: Somewhat not open, Not open at all
#3: Not applicable to me (I do not have a mental illness)


#1
df['discuss_mh_family'].replace(to_replace = ['Very open'], value = 1, inplace = True)
df['discuss_mh_family'].replace(to_replace = ["Somewhat open"], value = 1, inplace = True)
df['discuss_mh_family'].replace(to_replace = ["Neutral"], value = 1, inplace = True)
#2
df['discuss_mh_family'].replace(to_replace = ['Somewhat not open'], value = 2, inplace = True)
df['discuss_mh_family'].replace(to_replace = ['Not open at all'], value = 2, inplace = True)
#3
df['discuss_mh_family'].replace(to_replace = ['Not applicable to me (I do not have a mental illness)'], value = 3, inplace = True)


#can work from remote
#1: Always, Sometimes
#2: Never

#1
df['remote'].replace(to_replace = ['Always'], value = 1, inplace = True)
df['remote'].replace(to_replace = ["Sometimes"], value = 1, inplace = True)
#2
df['remote'].replace(to_replace = ['Never'], value = 2, inplace = True)

df.to_csv('/Users/srtherri/Documents/MSBA/Data Mining/Mental Health Data Final.csv')

###############################################################################

df.info()

df2 = df[['number_of_employees','tech_company','tech_role','emp_provide_mhb','know_opt_mhb','leave_difficulty_mhb','diagnosed_mh',
         'mh_professional_treatment','mh_effects_work_when_treated','mh_effects_work_when_untreated','age','gender']]


df3 = df[['number_of_employees','tech_company','tech_role','emp_provide_mhb','know_opt_mhb','leave_difficulty_mhb','diagnosed_mh',
         'mh_professional_treatment','mh_effects_work_when_treated','mh_effects_work_when_untreated','age','gender','mhb_rss',
         'emp_takes_mh_seriously','family_history_mh','past_mh','current_mh','discuss_mhb_coworker','remote',
         'discuss_mh_family']]


df3.to_csv('/Users/srtherri/Documents/MSBA/Data Mining/Mental Health Data Diagnosis.csv')


###############################################################################


##run model on y=effects work when treated to predict if an employees work would still be
#effected with treated. for simplicity we will convert the effects_work_when_treated variable
#to a yes and no based on analysis of the answers given. ex: sometimes and often = Yes 

df4 = df3


#Does mh effect work when treated
#1: Often = Yes: 1
#2: Sometimes = Yes: 1
#3: No mental illness = No: 0
#4: Rarely = Yes: 1
#5: Never = No: 0

#1
#already coded
#2
df['mh_effects_work_when_treated'].replace(to_replace = [2], value = 1, inplace = True)
#3
df['mh_effects_work_when_treated'].replace(to_replace = [3], value = 0, inplace = True)
#4
df['mh_effects_work_when_treated'].replace(to_replace = [4], value = 1, inplace = True)
#5
df['mh_effects_work_when_treated'].replace(to_replace = [5], value = 0, inplace = True)


df.to_csv('/Users/srtherri/Documents/MSBA/Data Mining/Mental Health Data Final.csv')
###############################################################################

##Visuals##


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 18:32:31 2021

@author: srtherri
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("Mental Health Data Final.csv")

 

##Pie charts: Are those who do not identify as Male or Female diagnosed more than the other gender categories?

plt.figure(figsize=(10,10))

plt.subplot(1,3,1)
plt.title("Male")
plt.pie(df[df.gender==1]['diagnosed_mh'].value_counts(),
       autopct='%.1f%%',radius=1)
plt.legend(title="Diagnosed",labels=("Yes","No"), loc="best")  
 
plt.subplot(1,3,2)
plt.title("Female")
plt.pie(df[df.gender==2]['diagnosed_mh'].value_counts(),
       autopct='%.1f%%',radius=1)
plt.legend(title="Diagnosed",labels=("Yes","No"), loc="best")
    
plt.subplot(1,3,3)
plt.title("Unassigned")
plt.pie(df[df.gender==3]['diagnosed_mh'].value_counts(),
       autopct='%.1f%%',radius=1)
plt.legend(title="Diagnosed",labels=("Yes","No"), loc="best")
 

##Do large company sizes affect mental health conditions?

##Significanly different? Possibly for 'More than 1000'

##MH diagnosis in different size companies##

bar = sns.countplot(data=df,x='number_of_employees',hue='diagnosed_mh')
plt.title('Mental health in different size companies')
plt.legend(title="Diagnosis", labels=("Yes", "No"))
plt.xlabel("Company Size Category")
plt.ylabel("Frequency")
plt.show()



##Are those who do not identify as Male or Female older or younger? -- significant age difference between those who are diagnosed

#sns.catplot(x='gender', y='age', hue='diagnosed_mh', kind='bar', data=df)
#sns.catplot(x='gender', y='leave_difficulty_mhb', kind='bar', hue='gender', data=df)

 
##MHB offered by different company size

sns.countplot(data=df,x='number_of_employees',hue='emp_provide_mhb')
plt.title('Mental Health Benefits Offered by Company Size')
plt.legend(title="MHB Offered", labels=("Yes", "No"))
plt.xlabel("Company Size Category")
plt.ylabel("Frequency of Benefits Offered")
plt.show()

 

##number of employees vs how many know about benefits
df_mhb_yes = df.loc[df['emp_provide_mhb']==1]

sns.countplot(data=df_mhb_yes,x='number_of_employees',hue='know_opt_mhb')
plt.title('Awareness of Employees of MH Benefits by Company Size')
plt.legend(title="Knows MHB Options", labels=("Yes", "No"))
plt.xlabel("Company Size Category")
plt.ylabel("Frequency of Knowledge of Benefits")
plt.show()

 

##Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?

sns.countplot(data=df,x='number_of_employees',hue='discuss_mhb_emp')
plt.title('Has Your Employer Discussed MHB by Company Size')
plt.legend(title="MHB Discussed", labels=("Yes", "No"))
plt.xlabel("Company Size Category")
plt.ylabel("Frequency of Benefits Discussed")
plt.show()

 

##do employers take MHB seriously?

sns.countplot(data=df,x='number_of_employees',hue='emp_takes_mh_seriously')
plt.title('Company takes MHB Seriously by Company Size')
plt.legend(title="MHB Discussed", labels=("Yes", "No","I don't know"))
plt.xlabel("Company Size Category")
plt.ylabel("Frequency")
plt.show()


df_mhb_diagnosed=df.loc[df['diagnosed_mh']==1]

sns.countplot(data=df_mhb_diagnosed,x='Domestic_Foreign',hue='emp_provide_mhb')
plt.title('Domestic vs Foreign Employment Mental Health Benefits')
plt.legend(title="MHB Offered", labels=("Yes", "No"))
plt.xlabel("Domestic of Foreign")
plt.ylabel("Frequency")
plt.show()

###############################################################################

from scipy.stats import chi2_contingency

contingency = pd.crosstab(df['emp_provide_mhb'], df['mh_effects_work_when_treated'])
contingency

#heatmap
plt.figure(figsize=(12,8))
sns.heatmap(contingency, annot=True, cmap="YlGnBu")
plt.xticks([0.5,1.5], ['No','Yes'])
plt.yticks([0.5,1.5], ['No','Yes'])
plt.xlabel("Affects work when treated")
plt.ylabel("Employer provides MHB")
plt.title("MH affects work when Employer Provides MHB")

#Chi-square test of significance
c, p, dof, expected = chi2_contingency(contingency)
print(p)

###############################################################################


##Analysis##


"""
Created on Thu Apr 15 12:30:01 2021

@author: srtherri
"""

import pandas as pd
df=pd.read_csv('Mental Health Data Diagnosis.csv')

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

###############################################################################

#Failed clustering attempt##

#Kmeans
#from sklearn.cluster import KMeans
#from sklearn.preprocessing import MinMaxScaler
#scaler=MinMaxScaler()
#scaled_df=scaler.fit_transform(df2)

#km=KMeans(n_clusters=4)
#km.fit(scaled_df)

#df2["labels"]=km.labels_

#sns.scatterplot(x="leave_difficulty_mhb",y="mh_effects_work_when_untreated",data=df2,hue="labels")

#this did not work as i wanted it to. It is hard to cluster on categorical data
#even if it has dummy variables coded into it using kmeans.

###############################################################################

##MODEL JUDGED ON ACCURACY##

#Decision Tree

y=df.diagnosed_mh
x=df.drop("diagnosed_mh", axis=1)

### train test split

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)


#testing set

y_pred_test=dt.predict(x_test)
accuracy_score(y_test,y_pred_test)
f1_score(y_test,y_pred_test)


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plot_tree(dt)
plt.show()


###############################################################################

####imporving the DT by grid search####

dt.tree_.max_depth

parameter_grid={"max_depth":range(2,13),"min_samples_split":range(2,6)}

from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(dt,parameter_grid,verbose=3,scoring="accuracy")


#fit the grid
grid.fit(x_train,y_train)

grid.best_params_

#use optimized parameters to see if performance improves

dt=DecisionTreeClassifier(max_depth=3,min_samples_split=2)
dt.fit(x_train,y_train)

#predict on accuary

y_pred_train=dt.predict(x_train)
print("training score is", accuracy_score(y_train,y_pred_train))
print("training f1 score is", f1_score(y_train,y_pred_train))
print("training precision score is", precision_score(y_train,y_pred_train))
print("training recall score is", recall_score(y_train,y_pred_train))

#testing set

y_pred_test=dt.predict(x_test)
print("testing accuracy score is",accuracy_score(y_test,y_pred_test))
print("testing f1 score is",f1_score(y_test,y_pred_test))
print("testing precision score is",precision_score(y_test,y_pred_test))
print("testing recall score is",recall_score(y_test,y_pred_test))

#plot the optimzed tree

plt.figure(figsize=(10,10))
plot_tree(dt)
plt.show()


#RESULTS: Pruned Decision Tree#
#testing accuracy score is 0.8953488372093024  *
#testing f1 score is 0.8941176470588236
#testing precision score is 0.8837209302325582
#testing recall score is 0.9047619047619048

###############################################################################

import pandas as pd
df=pd.read_csv('Mental Health Data Diagnosis.csv')

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

###############################################################################

#random forest

y2=df.diagnosed_mh
x2=df.drop("diagnosed_mh", axis=1)

### train test split

from sklearn.model_selection import train_test_split
x2_train,x2_test,y2_train,y2_test=train_test_split(x2,y2,test_size=0.3,random_state=101)

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=500)

rfc.fit(x2_train,y2_train)

#model performance in training and testing

y2_pred_train=rfc.predict(x2_train)
print("training accuracy score is", accuracy_score(y2_train,y2_pred_train))
print("training f1 score is", f1_score(y2_train,y2_pred_train))
print("training precision score is", precision_score(y2_train,y2_pred_train))
print("training recall score is", recall_score(y2_train,y2_pred_train))

#testing set

y2_pred_test=rfc.predict(x2_test)
print("testing accuracy score is",accuracy_score(y2_test,y2_pred_test))
print("testing f1 score is",f1_score(y2_test,y2_pred_test))
print("testing precision score is",precision_score(y2_test,y2_pred_test))
print("testing recall score is",recall_score(y2_test,y2_pred_test))

#RESULTS: Random Forest Model#
#testing accuracy score is 0.8837209302325582
#testing f1 score is 0.8857142857142857
#testing precision score is 0.8516483516483516
#testing recall score is 0.9226190476190477

###############################################################################


#### Improving the model by tuning the random forest####

parameter_grid={"max_depth":range(2,11),"min_samples_split":range(2,6)}

from sklearn.model_selection import GridSearchCV

grid=GridSearchCV(rfc,parameter_grid,verbose=3,scoring="f1")
grid.fit(x2_train,y2_train)

grid.best_params_

#run with optimized parameters

rfc=RandomForestClassifier(n_estimators=500, max_depth=6,min_samples_split=5)

rfc.fit(x2_train,y2_train)

#training results

y_pred_train_gs_rfc=rfc.predict(x2_train)

print("training accuarcy score is", accuracy_score(y2_train,y_pred_train_gs_rfc))
print("training f1 score is", f1_score(y2_train,y_pred_train_gs_rfc))
print("training recall score is", recall_score(y2_train,y_pred_train_gs_rfc))
print("training precision score is", precision_score(y2_train,y_pred_train_gs_rfc))

#test results
y_pred_test_gs_rfc=rfc.predict(x2_test)

print("Accuary score is",accuracy_score(y2_test,y_pred_test_gs_rfc))
print("F1 score is", f1_score(y2_test,y_pred_test_gs_rfc))
print("Recall Score is",recall_score(y2_test,y_pred_test_gs_rfc))
print("Precision score is",precision_score(y2_test,y_pred_test_gs_rfc))

#RESULTS: Tuned Random Forest Model#

#Accuary score is 0.8953488372093024
#F1 score is 0.8959537572254336
#Recall Score is 0.9226190476190477
#Precision score is 0.8707865168539326

#Accuracy of the Tuned RFC gave the best reults at just over 89%

##############################################################################
##############################################################################
##############################################################################

##MODEL JUDGED ON F1##

#Decision Tree

y=df.diagnosed_mh
x=df.drop("diagnosed_mh", axis=1)

### train test split

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)


#testing set

y_pred_test=dt.predict(x_test)
accuracy_score(y_test,y_pred_test)
f1_score(y_test,y_pred_test)


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plot_tree(dt)
plt.show()


###############################################################################

####imporving the DT by grid search####

dt.tree_.max_depth

parameter_grid={"max_depth":range(2,13),"min_samples_split":range(2,6)}

from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(dt,parameter_grid,verbose=3,scoring="f1")


#fit the grid
grid.fit(x_train,y_train)

grid.best_params_

#use optimized parameters to see if performance improves

dt=DecisionTreeClassifier(max_depth=3,min_samples_split=2)
dt.fit(x_train,y_train)

#predict on accuary

y_pred_train=dt.predict(x_train)
print("training score is", accuracy_score(y_train,y_pred_train))
print("training f1 score is", f1_score(y_train,y_pred_train))
print("training precision score is", precision_score(y_train,y_pred_train))
print("training recall score is", recall_score(y_train,y_pred_train))

#testing set

y_pred_test=dt.predict(x_test)
print("testing accuracy score is",accuracy_score(y_test,y_pred_test))
print("testing f1 score is",f1_score(y_test,y_pred_test))
print("testing precision score is",precision_score(y_test,y_pred_test))
print("testing recall score is",recall_score(y_test,y_pred_test))

#plot the optimzed tree

plt.figure(figsize=(10,10))
plot_tree(dt)
plt.show()


#RESULTS: Pruned Decision Tree#
#testing accuracy score is 0.8953488372093024  *
#testing f1 score is 0.8941176470588236
#testing precision score is 0.8837209302325582
#testing recall score is 0.9047619047619048

###############################################################################

import pandas as pd
df=pd.read_csv('Mental Health Data Diagnosis.csv')

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

###############################################################################

#random forest

y2=df.diagnosed_mh
x2=df.drop("diagnosed_mh", axis=1)

### train test split

from sklearn.model_selection import train_test_split
x2_train,x2_test,y2_train,y2_test=train_test_split(x2,y2,test_size=0.3,random_state=101)

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=500)

rfc.fit(x2_train,y2_train)

#model performance in training and testing

y2_pred_train=rfc.predict(x2_train)
print("training accuracy score is", accuracy_score(y2_train,y2_pred_train))
print("training f1 score is", f1_score(y2_train,y2_pred_train))
print("training precision score is", precision_score(y2_train,y2_pred_train))
print("training recall score is", recall_score(y2_train,y2_pred_train))

#testing set

y2_pred_test=rfc.predict(x2_test)
print("testing accuracy score is",accuracy_score(y2_test,y2_pred_test))
print("testing f1 score is",f1_score(y2_test,y2_pred_test))
print("testing precision score is",precision_score(y2_test,y2_pred_test))
print("testing recall score is",recall_score(y2_test,y2_pred_test))

#RESULTS: Random Forest Model#
#testing accuracy score is 0.8895348837209303  *
#testing f1 score is 0.8901734104046243
#testing precision score is 0.8651685393258427
#testing recall score is 0.9166666666666666

###############################################################################


#### Improving the model by tuning the random forest####

parameter_grid={"max_depth":range(2,11),"min_samples_split":range(2,6)}

from sklearn.model_selection import GridSearchCV

grid=GridSearchCV(rfc,parameter_grid,verbose=3,scoring="f1")
grid.fit(x2_train,y2_train)

grid.best_params_

#run with optimized parameters

rfc=RandomForestClassifier(n_estimators=500, max_depth=6,min_samples_split=4)

rfc.fit(x2_train,y2_train)

#training results

y_pred_train_gs_rfc=rfc.predict(x2_train)

print("training accuarcy score is", accuracy_score(y2_train,y_pred_train_gs_rfc))
print("training f1 score is", f1_score(y2_train,y_pred_train_gs_rfc))
print("training recall score is", recall_score(y2_train,y_pred_train_gs_rfc))
print("training precision score is", precision_score(y2_train,y_pred_train_gs_rfc))

#test results
y_pred_test_gs_rfc=rfc.predict(x2_test)

print("Accuary score is",accuracy_score(y2_test,y_pred_test_gs_rfc))
print("F1 score is", f1_score(y2_test,y_pred_test_gs_rfc))
print("Recall Score is",recall_score(y2_test,y_pred_test_gs_rfc))
print("Precision score is",precision_score(y2_test,y_pred_test_gs_rfc))

#RESULTS: Tuned Random Forest Model#

#Accuary score is 0.9011627906976745  *
#F1 score is 0.9017341040462428
#Recall Score is 0.9285714285714286
#Precision score is 0.8764044943820225

##############################################################################
##############################################################################
##############################################################################

#Does the employer offer MHB

y=df3.emp_provide_mhb
x=df3.drop("emp_provide_mhb", axis=1)

### train test split

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)

#evaluate the training set

from sklearn.metrics import accuracy_score

y_pred_train=dt.predict(x_train)
accuracy_score(y_train,y_pred_train)

#testing set

y_pred_test=dt.predict(x_test)
accuracy_score(y_test,y_pred_test)
#0.7063953488372093

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plot_tree(dt)
plt.show()


dt.tree_.max_depth

parameter_grid={"max_depth":range(2,20),"min_samples_split":range(2,6)}

from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(dt,parameter_grid,verbose=3,scoring="accuracy")


#fit the grid
grid.fit(x_train,y_train)

grid.best_params_

#use optimized parameters to see if performance improves

dt=DecisionTreeClassifier(max_depth=2,min_samples_split=2)
dt.fit(x_train,y_train)

from sklearn.metrics import accuracy_score

#testing set

y_pred_test=dt.predict(x_test)
print("testing score is",accuracy_score(y_test,y_pred_test))
#testing score is 0.7558139534883721

#plot the optimzed tree

plt.figure(figsize=(10,10))
plot_tree(dt)
plt.show()

###############################################################################

#random forest

y=df3.emp_provide_mhb
x=df3.drop("emp_provide_mhb", axis=1)

### train test split

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=500)

rfc.fit(x_train,y_train)

#model performance in training and testing

y_pred_train=rfc.predict(x_train)
print("training accuracy score is", accuracy_score(y_train,y_pred_train))
print("training f1 score is", f1_score(y_train,y_pred_train))
print("training precision score is", precision_score(y_train,y_pred_train))
print("training recall score is", recall_score(y_train,y_pred_train))

#testing set

y_pred_test=rfc.predict(x_test)
print("testing accuracy score is",accuracy_score(y_test,y_pred_test))
print("testing f1 score is",f1_score(y_test,y_pred_test))
print("testing precision score is",precision_score(y_test,y_pred_test))
print("testing recall score is",recall_score(y_test,y_pred_test))

#Results#
#testing accuracy score is 0.7616279069767442
#testing f1 score is 0.7172413793103448
#testing precision score is 0.8
#testing recall score is 0.65

#tuning the random forest model

parameter_grid={"max_depth":range(2,15),"min_samples_split":range(2,6)}

from sklearn.model_selection import GridSearchCV

grid=GridSearchCV(rfc,parameter_grid,verbose=3,scoring="accuracy")
grid.fit(x_train,y_train)

grid.best_params_

#run with optimized parameters

rfc=RandomForestClassifier(n_estimators=500, max_depth=5,min_samples_split=3)

rfc.fit(x_train,y_train)

#training results

y_pred_train_gs_rfc=rfc.predict(x_train)

print("training f1 score is", f1_score(y_train,y_pred_train_gs_rfc))
print("training recall score is", recall_score(y_train,y_pred_train_gs_rfc))
print("training precision score is", precision_score(y_train,y_pred_train_gs_rfc))
print("training accuarcy score is", accuracy_score(y_train,y_pred_train_gs_rfc))

#test results
y_pred_test_gs_rfc=rfc.predict(x_test)

print("F1 score is", f1_score(y_test,y_pred_test_gs_rfc))
print("Recall Score is",recall_score(y_test,y_pred_test_gs_rfc))
print("Precision score is",precision_score(y_test,y_pred_test_gs_rfc))
print("Accuary score is",accuracy_score(y_test,y_pred_test_gs_rfc))

#Results#

#Accuary score is 0.7732558139534884
#F1 score is 0.7214285714285714
#Recall Score is 0.63125
#Precision score is 0.8416666666666667

#accuracy score for the tuned random forest classifier worked best, however is still
#only at 77% accuracy


# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 16:55:01 2021

@author: TRAM LE
"""

import pandas as pd
df=pd.read_csv("C:/NEU/MASTERRRRR/2nd semester/MISM 6212 Data Mining Mon 750am/Project/Mental Health Data.csv")

#rename all column
a=list(df)
b=['self_employed','number_of_employees','tech_company','tech_role','emp_provide_mhb','know_opt_mhb',
   'discuss_mhb_emp','mhb_rss','leave_difficulty_mhb',
   'discuss_mhb_coworker','emp_takes_mh_seriously','knowledge_of_local_online_resources',
   'productivity_affected_by_mental_health','percentage_work_time_affected_mental_health','discuss_mh_family',
   'family_history_mh','past_mh','current_mh','diagnosed_mh','mh_condition','mh_professional_treatment',
   'mh_effects_work_when_treated','mh_effects_work_when_untreated','age','gender','country','state',
   'employment_country','employment_state','role','remote','']
for i,j in zip(a,b):
    df.rename(columns={i:j},inplace=True)
    
#gender normalization

df['gender'].replace(to_replace = ['Male', 'male', 'Male ', 'M', 'm',
       'man', 'Cis male', 'Male.', 'male 9:1 female, roughly', 'Male (cis)', 'Man', 'Sex is male',
       'cis male', 'Malr', 'Dude', "I'm a man why didn't you make this a drop down question. You should of asked sex? And I would of answered yes please. Seriously how much text can this take? ",
       'mail', 'M|', 'Male/genderqueer', 'male ',
       'Cis Male', 'Male (trans, FtM)',
       'cisdude', 'cis man', 'MALE'], value = 'male', inplace = True)
df['gender'].replace(to_replace = ['Female', 'female', 'I identify as female.', 'female ',
       'Female assigned at birth ', 'F', 'Woman', 'fm', 'f', 'Cis female ', 'Transitioned, M2F',
       'Genderfluid (born female)', 'Female or Multi-Gender Femme', 'Female ', 'woman', 'female/woman',
       'Cisgender Female', 'fem', 'Female (props for making this a freeform field, though)',
       ' Female', 'Cis-woman', 'female-bodied; no feelings about gender',
       'AFAB'], value = 'female', inplace = True)
df['gender'].replace(to_replace = ['Bigender', 'non-binary', 'Other/Transfeminine',
       'Androgynous', 'Other', 'nb masculine',
       'none of your business', 'genderqueer', 'Human', 'Genderfluid',
       'Enby', 'genderqueer woman', 'mtf', 'Queer', 'Agender', 'Fluid',
       'Nonbinary', 'human', 'Unicorn', 'Genderqueer',
       'Genderflux demi-girl', 'Transgender woman'], value = 'other', inplace = True)

# emp_provide_mhb binary
#1: yes
#2: No, i dont know

#1
df['emp_provide_mhb'].replace(to_replace = ['Yes'], value = 1, inplace = True)
#2
df['emp_provide_mhb'].replace(to_replace = ['No'], value = 2, inplace = True)
df['emp_provide_mhb'].replace(to_replace = ["I don't know"], value = 2, inplace = True)
df['emp_provide_mhb'].replace(to_replace = ['Not eligible for coverage / N/A'], value = 2, inplace = True)

#difficulty of leave
#1: Very easy
#2: Somewhat easy
#3: Neither easy nor difficult
#4: Somewhat difficult
#5: Very difficult
#6: I don't know

#1
df['leave_difficulty_mhb'].replace(to_replace = ['Very easy'], value = 1, inplace = True)
#2
df['leave_difficulty_mhb'].replace(to_replace = ['Somewhat easy'], value = 2, inplace = True)
#3
df['leave_difficulty_mhb'].replace(to_replace = ['Neither easy nor difficult'], value = 3, inplace = True)
#4
df['leave_difficulty_mhb'].replace(to_replace = ['Somewhat difficult'], value = 4, inplace = True)
#5
df['leave_difficulty_mhb'].replace(to_replace = ['Very difficult'], value = 5, inplace = True)
#6
df['leave_difficulty_mhb'].replace(to_replace = ["I don't know"], value = 1, inplace = True)


#diagnosed_mh to binary
#1: Yes
#2: No

#1
df['diagnosed_mh'].replace(to_replace = ['Yes'], value = 1, inplace = True)
#2
df['diagnosed_mh'].replace(to_replace = ['No'], value = 2, inplace = True)


#Does mh effect work when treated
#1: Often
#2: Sometimes
#3: No mental illness
#4: Rarely
#5: Never

#1
df['mh_effects_work_when_treated'].replace(to_replace = ['Often'], value = 1, inplace = True)
#2
df['mh_effects_work_when_treated'].replace(to_replace = ['Sometimes'], value = 2, inplace = True)
#3
df['mh_effects_work_when_treated'].replace(to_replace = ['Not applicable to me'], value = 3, inplace = True)
#4
df['mh_effects_work_when_treated'].replace(to_replace = ['Rarely'], value = 4, inplace = True)
#5
df['mh_effects_work_when_treated'].replace(to_replace = ['Never'], value = 5, inplace = True)


#Does mh effect work when untreated
#1: Often
#2: Sometimes
#3: No mental illness
#4: Rarely
#5: Never

#1
df['mh_effects_work_when_untreated'].replace(to_replace = ['Often'], value = 1, inplace = True)
#2
df['mh_effects_work_when_untreated'].replace(to_replace = ['Sometimes'], value = 2, inplace = True)
#3
df['mh_effects_work_when_untreated'].replace(to_replace = ['Not applicable to me'], value = 3, inplace = True)
#4
df['mh_effects_work_when_untreated'].replace(to_replace = ['Rarely'], value = 4, inplace = True)
#5
df['mh_effects_work_when_untreated'].replace(to_replace = ['Never'], value = 5, inplace = True)


#Gender dummies
#1: male
#2: female
#3: other

#1
df['gender'].replace(to_replace = ['male'], value = 1, inplace = True)
#2
df['gender'].replace(to_replace = ['female'], value = 2, inplace = True)
#3
df['gender'].replace(to_replace = ['other'], value = 3, inplace = True)


#discuss with employer
#1: yes
#2: No

#1
df['discuss_mhb_emp'].replace(to_replace = ['Yes'], value = 1, inplace = True)
#2
df['discuss_mhb_emp'].replace(to_replace = ['No'], value = 2, inplace = True)
df['discuss_mhb_emp'].replace(to_replace = ["I don't know"], value = 2, inplace = True)


#mhb_rss
#1: yes
#2: No

#1
df['mhb_rss'].replace(to_replace = ['Yes'], value = 1, inplace = True)
#2
df['mhb_rss'].replace(to_replace = ['No'], value = 2, inplace = True)
df['mhb_rss'].replace(to_replace = ["I don't know"], value = 2, inplace = True)


#does the employer take mh seriously
#1: yes
#2: No
#3: I don't know

#1
df['emp_takes_mh_seriously'].replace(to_replace = ['Yes'], value = 1, inplace = True)
#2
df['emp_takes_mh_seriously'].replace(to_replace = ['No'], value = 2, inplace = True)
#3
df['emp_takes_mh_seriously'].replace(to_replace = ["I don't know"], value = 3, inplace = True)


#family history
#1: yes
#2: No
#3: I don't know

#1
df['family_history_mh'].replace(to_replace = ['Yes'], value = 1, inplace = True)
#2
df['family_history_mh'].replace(to_replace = ['No'], value = 2, inplace = True)
#3
df['family_history_mh'].replace(to_replace = ["I don't know"], value = 3, inplace = True)


#past_mh
#1: yes
#2: No

#1
df['past_mh'].replace(to_replace = ['Yes'], value = 1, inplace = True)
#2
df['past_mh'].replace(to_replace = ['No'], value = 2, inplace = True)
df['past_mh'].replace(to_replace = ["Maybe"], value = 2, inplace = True)


#current_mh
#1: yes
#2: No

#1
df['current_mh'].replace(to_replace = ['Yes'], value = 1, inplace = True)
#2
df['current_mh'].replace(to_replace = ['No'], value = 2, inplace = True)
df['current_mh'].replace(to_replace = ["Maybe"], value = 2, inplace = True)
    
df.info()

############################################################################
#Dataset preparation 
df_1 = df[df['self_employed']==1]
cols = ['self_employed','number_of_employees','tech_company','tech_role','emp_provide_mhb','know_opt_mhb',
   'discuss_mhb_emp','mhb_rss','leave_difficulty_mhb',
   'discuss_mhb_coworker','emp_takes_mh_seriously','knowledge_of_local_online_resources','mh_condition',
   'country','state','employment_country','employment_state','role']
df_red=df_1.drop(cols,axis=1)

#How willing would you be to share with friends and family that you have a mental illness?
#discuss_mh_family

#1: Not applicable to me (I do not have a mental illness)
#2: Not open at all
#3: Somewhat not open
#4: Neutral
#5: Somewhat open
#6: Very open

#1
df_red['discuss_mh_family'].replace(to_replace = ['Not applicable to me (I do not have a mental illness)'], value = 1, inplace = True)
#2
df_red['discuss_mh_family'].replace(to_replace = ['Not open at all'], value = 2, inplace = True)
#3
df_red['discuss_mh_family'].replace(to_replace = ['Somewhat not open'], value = 3, inplace = True)
#4
df_red['discuss_mh_family'].replace(to_replace = ['Neutral'], value = 4, inplace = True)
#5
df_red['discuss_mh_family'].replace(to_replace = ['Somewhat open'], value = 5, inplace = True)
#6
df_red['discuss_mh_family'].replace(to_replace = ["Very open"], value = 6, inplace = True)

#If yes, what percentage of your work time (time performing primary or secondary job functions) is affected by a mental health issue?
#percentage_work_time_affected_mental_health
#1: 1-25%
#2: 26-50%
#3: 51-75%
#4: 76-100%
#5: na

df_red.percentage_work_time_affected_mental_health = df_red.percentage_work_time_affected_mental_health.fillna("dont know")
#1
df_red['percentage_work_time_affected_mental_health'].replace(to_replace = ['1-25%'], value = 1, inplace = True)
#2
df_red['percentage_work_time_affected_mental_health'].replace(to_replace = ['26-50%'], value = 2, inplace = True)
#3
df_red['percentage_work_time_affected_mental_health'].replace(to_replace = ['51-75%'], value = 3, inplace = True)
#4
df_red['percentage_work_time_affected_mental_health'].replace(to_replace = ['76-100%'], value = 4 , inplace = True)
#5
df_red['percentage_work_time_affected_mental_health'].replace(to_replace = ['dont know'], value = 5, inplace = True)

#Remote
#1: Never
#2: Sometimes
#3: Always

#1
df_red['remote'].replace(to_replace = ['Never'], value = 1, inplace = True)
#2
df_red['remote'].replace(to_replace = ['Sometimes'], value = 2, inplace = True)
#3
df_red['remote'].replace(to_replace = ['Always'], value = 3, inplace = True)
##############################################################################

##K-means cluster
##seperate the x and y
#y=productivity=Yes
#x=the rest
 
df_red1=pd.get_dummies(df_red,drop_first=True)
y=df_red1["productivity_affected_by_mental_health_Yes"]
x=df_red1.drop(["productivity_affected_by_mental_health_Yes"],axis=1)

import seaborn as sns
sns.pairplot(df_red1)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaled_df=scaler.fit_transform(df_red1)

#identify the number of cluster
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
wcv=[]#within cluster variation
silk_score=[]

for i in range(2,11):
    km=KMeans(n_clusters=i,random_state=0)
    km.fit(scaled_df)
    wcv.append(km.inertia_)
    silk_score.append(silhouette_score(scaled_df,km.labels_))
    
import matplotlib.pyplot as plt
plt.plot(range(2,11),wcv   )
plt.xlabel("no of clusters")
plt.ylabel("with in cluster variation")

plt.plot(range(2,11),silk_score   )
plt.xlabel("no of clusters")
plt.ylabel("silk score")
plt.grid()

## k=2
from sklearn.cluster import KMeans
km2=KMeans(n_clusters=2,random_state=0)
km2.fit(scaled_df)

#labels
km2.labels_

#visualize the clusters

df_red1["label"]=km2.labels_


#interpret it

#cluster 0
cluster0=df_red1.loc[df_red1["label"]==0].describe()


#cluster=1
cluster1=df_red1.loc[df_red1["label"]==1].describe()
















