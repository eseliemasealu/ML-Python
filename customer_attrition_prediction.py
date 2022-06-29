#!/usr/bin/env python
# coding: utf-8

# In[307]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# # Exploratory Data Analysis (EDA)

# In[308]:


# Load dataset and print first 5 rows
df= pd.read_csv("BankChurners.csv")
df.head()


# In[309]:


# Drop the columns that will have no relevance to our analysis
df = df.drop(['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'CLIENTNUM'],axis = 1)        
        


# In[310]:


# Check for Null Values
df.isnull().sum()


# In[311]:


#Fill "Unknown" with NaN
df = df.replace('Unknown', np.NaN)


# In[312]:


#Check for missing values
# Marital Status is  one hot column (not ordinal)
df.isna().sum()


# In[313]:


# Fill ordinal missing values with modes
df['Education_Level'] = df['Education_Level'].fillna('Graduate')
df['Income_Category'] = df['Income_Category'].fillna('Less than 40k')
df.isnull().sum()


# In[314]:


#Plot a distribution of Customer Age
sns.histplot(df['Customer_Age'], bins = 10, kde = True)
plt.title("Distribution of Customer Age" )
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.xlim(0, 100)
plt.show()


# In[315]:


#Plot a distribution of Credit_Limit
sns.histplot(df['Credit_Limit'], bins = 20, kde = True)
plt.title("Distribution of Credit_Limit" )
plt.xlabel('Credit_Limit')
plt.ylabel('Frequency')
plt.xlim(100, 40000 )
plt.show()


# In[316]:


# Check out the various types of attributes
df.dtypes


# In[317]:


#Count unique values per column
print(df.Gender.value_counts())
print(df.Dependent_count.value_counts())
print(df.Education_Level.value_counts())
print(df.Marital_Status.value_counts())
print(df.Income_Category.value_counts())
print(df.Card_Category.value_counts())


# In[318]:


#Encoding to create dummy variables
def binary_encode(df, column, positive_value):
    df = df.copy()    
    df[column] =df[column].apply(lambda x : 1 if x == positive_value else 0)
    return df

def ordinal_encode(df, column, ordering):
    df = df.copy()
    df[column] = df[column].apply(lambda x: ordering.index(x))
    return df


def onehot_encode(df, column, prefix):
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=prefix)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(column, axis=1)
    return df


# In[319]:


# Encode Binary Columns
df = binary_encode(df, 'Attrition_Flag', positive_value = 'Attrited Customer')
df = binary_encode(df, 'Gender', positive_value = 'M')

# Encode Ordinal Columns
education_ordering =  [
    'Uneducated',
    'High School',                      
    'College' , 
    'Graduate',
    'Post-Graduate',  
    'Doctorate' 
]
income_ordering = [
        'Less than $40K',
        'Less than 40k',
        '$40K - $60K',
        '$60K - $80K',
        '$80K - $120K',
        '$120K +'
]

df = ordinal_encode(df, 'Education_Level', ordering = education_ordering )

df = ordinal_encode(df, 'Income_Category', ordering=income_ordering)

#Endcode Nominal Functions
df = onehot_encode(df, 'Marital_Status', prefix = 'MS')
df = onehot_encode(df, 'Card_Category', prefix = 'CC')


# In[390]:


# Use Standard Scaler to Scale the data 
from sklearn.preprocessing import StandardScaler

#Define X & Y 
x = df.drop(['Attrition_Flag'], axis = 1)
y = df[['Attrition_Flag']]


#Scale x-variables
scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x), columns = x.columns)


# In[391]:


x.head()


# In[402]:


df1 = pd.concat([x.loc[:, ['Customer_Age', 'Months_on_book']], x.loc[:,'Credit_Limit':'Avg_Utilization_Ratio']], axis=1).copy()


# In[405]:


corr = pd.concat([df1, y], axis=1).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, vmin=-1.0, cmap='mako')
plt.show()


# # Build Prediction Models

# In[398]:


# Load packages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=123)


# In[399]:


# Fit the Logistic regression

logmodel=LogisticRegression(solver='liblinear')
logmodel.fit(x_train,y_train.values.ravel())


# In[400]:


# Make Predictions
y_pred=logmodel.predict(x_test)

#To get probability 
y_probab=logmodel.predict_proba(x_test)

#coef and intercept : b0 an b1
logmodel.coef_


# ## Evaluate Model Performance

# In[416]:


# Load Packagees
from sklearn.metrics import confusion_matrix,f1_score, recall_score, precision_score
# Print Scores
print('model score is', logmodel.score(x_test, y_test) * 100,'%')


# In[414]:


C_mat = pd.DataFrame(confusion_matrix(y_test,y_pred,labels = [0,1]),index=["Actual:0","Actual:1"],
                     columns = ["Pred:0","Pred:1"])

print( C_mat)


# # Seeing if we can improve precision, recall & f1_score

# In[385]:


data_corr = df.select_dtypes(include=[np.number])
data_corr.head()
corr = data_corr.corr()
corr.head(20)
corr.sort_values(['Attrition_Flag'], ascending=False, inplace=True)
corr['Attrition_Flag']


# In[370]:


x = df[['Contacts_Count_12_mon','Months_Inactive_12_mon', 'Education_Level', 'Gender',
        'Education_Level', 'MS_Single', 'Dependent_count', 'Customer_Age', 'Months_on_book','Total_Trans_Ct','Total_Trans_Amt',
        'Total_Relationship_Count','Total_Ct_Chng_Q4_Q1','Total_Revolving_Bal',
       'Avg_Utilization_Ratio']]
y = df[['Attrition_Flag']]


# In[371]:


#train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)
logmodel=LogisticRegression(solver='liblinear')
logmodel.fit(x_train,y_train.values.ravel())
# Make Predictions
y_pred=logmodel.predict(x_test)

#To get probability 
y_probab=logmodel.predict_proba(x_test)

#coef and intercept : b0 an b1
logmodel.coef_


# In[372]:


# Print Scores
print('recall score is',recall_score(y_test, y_pred))
print('f1 score is',f1_score(y_test,y_pred))
print('precision score is' ,precision_score(y_test, y_pred))


# In[346]:


C_mat = pd.DataFrame(confusion_matrix(y_test,y_pred,labels = [0,1]),index=["Actual:0","Actual:1"],
                     columns = ["Pred:0","Pred:1"])

print( C_mat)


# In[347]:


logmodel.intercept_


# In[348]:


logmodel.predict_proba(x_test)


# In[417]:


print( '291 out of 3,039 predictions were misclassified') 


# In[ ]:




