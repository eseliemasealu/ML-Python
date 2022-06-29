#hierarchical clustering 
import pandas as pd
df=pd.read_csv('circles.csv')

import seaborn as sns
sns.scatterplot(x='x1',y='x2',data=df)

##run kmeans with k=2 and see if kmeans can separate these two clusters 
###always good idea to scale the data 
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaled_df=scaler.fit_transform(df)

#lets do Kmeans with k=2
from sklearn.cluster import KMeans
km=KMeans(n_clusters=2)
km.fit(scaled_df)

df['labels']=km.labels_

sns.scatterplot(x='x1',y='x2',data=df,hue='labels')


#####let us do the ward method
#first we will start with dendogram
from scipy.cluster.hierarchy import dendrogram,linkage
import matplotlib.pyplot as plt

linked=linkage(scaled_df,method='ward')#this is basically creating a huge matrix
dendrogram(linked)
plt.show()
#hierarchical clustering can take some time


###single method (to see the dendrogram)
linked=linkage(scaled_df,method='single')#this is basically creating a huge matrix
dendrogram(linked)
plt.show()
#majority of the time the single method does not work but when it does, it works really well

##single hierarchical method with 2 clusters
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=2,linkage='single')
hc.fit(scaled_df)

df['labels']=hc.labels_

sns.scatterplot(x='x1',y='x2',data=df,hue='labels')
#here we see that the single method has identified the two clusters so it works in this case

##excercise: can you try the different approaches 
###ward, single, complete, average on the future dataset from previous week and see which gives good results

df1=pd.read_csv('future.csv')
import seaborn as sns
sns.scatterplot(x='INCOME',y='SPEND',data=df1)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaled_df1=scaler.fit_transform(df1)

#lets do Kmeans with k=2
from sklearn.cluster import KMeans
km=KMeans(n_clusters=2)
km.fit(scaled_df1)

df1['labels']=km.labels_

sns.scatterplot(x='INCOME',y='SPEND',data=df1)

from scipy.cluster.hierarchy import dendrogram,linkage
import matplotlib.pyplot as plt

linked=linkage(scaled_df1,method='ward')
dendrogram(linked)
plt.show()

linked=linkage(scaled_df1,method='single')
dendrogram(linked)
plt.show()

linked=linkage(scaled_df1,method='complete')
dendrogram(linked)
plt.show()

linked=linkage(scaled_df1,method='average')
dendrogram(linked)
plt.show()

##### in this case, the ward method works best
