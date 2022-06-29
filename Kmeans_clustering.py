#kmeans clustering 

import pandas as pd 
df=pd.read_csv('future.csv')

import seaborn as sns 
sns.pairplot(df)

##we will use the minmax scaler
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaled_df=scaler.fit_transform(df)

#random guess k=4
from sklearn.cluster import KMeans
km4=KMeans(n_clusters=4,random_state=0)
km4.fit(scaled_df)#clusters are dound in this step

##labels
km4.labels_
 
##visualize the clusters 
df['label']=km4.labels_
 
sns.scatterplot(x='INCOME',y='SPEND',data=df,hue='label',palette='Set1') 

from sklearn.metrics import silhouette_score
wcv=[]
silk_score=[]

for i in range(2,11):
    km=KMeans(n_clusters=i,random_state=0)
    km.fit(scaled_df)
    wcv.append(km.inertia_)
    silk_score.append(silhouette_score(scaled_df,km.labels_))
    
    
import matplotlib.pyplot as plt
plt.plot ( range(2,11),wcv )
plt.xlabel( 'no of clusters')
plt.ylabel("silk score")
plt.grid()


##class excercise
#do the knn for k=3 and interpret the results 

from sklearn.cluster import KMeans
km3=KMeans(n_clusters=3,random_state=0)
km3.fit(scaled_df)#clusters are dound in this step

##labels
km3.labels_
 
##visualize the clusters 
df['label']=km3.labels_
 
sns.scatterplot(x='INCOME',y='SPEND',data=df,hue='label',palette='Set1') 


###interpret the clusters
#0: high spending and low income
#1: high income and medium spend
#2: low income and low spend 



from sklearn.metrics import silhouette_score
wcv=[]
silk_score=[]

for i in range(2,11):
    km=KMeans(n_clusters=i,random_state=0)
    km.fit(scaled_df)
    wcv.append(km.inertia_)
    silk_score.append(silhouette_score(scaled_df,km.labels_))

    
import matplotlib.pyplot as plt
plt.plot ( range(2,11),wcv )
plt.xlabel( 'no of clusters')
plt.ylabel("silk score")
plt.grid()



##########################################



import pandas as pd 
df=pd.read_csv("food.csv")

df=df.drop('Item',axis=1)

import seaborn as sns
sns.pairplot(df)

#we will use the minmax scaler
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaled_df=scaler.fit_transform(df)

#figuring out the #of cluster
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
wcv=[] ##with cluster variation
silk_score=[]

for i in range(2,11):
    km=KMeans(n_clusters=i,random_state=0)
    km.fit(scaled_df)
    wcv.append(km.inertia_)
    silk_score.append(silhouette_score(scaled_df,km.labels_))


import matplotlib.pyplot as plt
plt.plot ( range(2,11),wcv )
plt.xlabel( 'no of clusters')
plt.ylabel("with in cluster variation")


plt.plot( range(2,11),silk_score)
plt.xlabel('no of clusters')
plt.ylabel('silk score')
plt.grid()

km3=KMeans(n_clusters=3,random_state=0)
km3.fit(scaled_df)


#visualize the clusters 
df['label']=km3.labels_


##interpret it
#cluster 0
df.loc[df['label']==0].describe()
#high calories, fat foods

#cluster 1
df.loc[df['label']==1].describe()
#medium calories, high calcium foods

#cluster 2
df.loc[df['label']==2].describe()







































