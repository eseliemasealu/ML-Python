#matplotlib: old but relevant library
#seaborn: new library that has good visuals

import matplotlib.pyplot as plt

pip install yfinance 
import yfinance as yf #stock prices

TSLA_df=yf.download("TSLA")

import pandas as pd
TSLA_df=pd.read_csv("Tesla-12.csv")

plt.figure(figsize=(10,6)) #this always needs to be at the top
plt.plot(TSLA_df["Open"],label="Tesla stock",color="red")
plt.legend(fontsize=16)
plt.xlabel("Date",fontsize=14)
plt.ylabel("Stock price",fontsize=14)
#we can also change the x and y ticks (the numbers on the axes)
plt.xticks(fontsize=14,rotation=90) #rotation rotates the number on the x axis by 90 degrees
plt.yticks(fontsize=14)
plt.show()#this is like a print statement for visualization 

##do the same thing for google
GOOGL_df=yf.download("GOOGL", "2018-01-01","2021-02-08")
plt.plot(GOOGL_df["Open"], label="Google stock",color="green")
plt.legend(fontsize=18)
plt.xlabel("Date",fontsize=15)
plt.ylabel("Stock price",fontsize=15)
plt.xticks(fontsize=15,rotation=90) 
plt.yticks(fontsize=15)
plt.show()

###comparing google and tesla stock
plt.plot(TSLA_df["Open"],label="Tesla stock",color="red")
plt.plot(GOOGL_df["Open"], label="Google stock",color="green")
plt.legend(fontsize=16)
plt.xlabel("Date",fontsize=14)
plt.ylabel("Stock price",fontsize=14)
plt.xticks(fontsize=14,rotation=90) #rotation rotates the number on the x axis by 90 degrees
plt.yticks(fontsize=14)
plt.show()


###plotting coordinates 
x_coord=[10,20,30]
y_coord=[14,30,10]
plt.plot(x_coord,y_coord)
plt.text(15,5,"This is my first text")

###scatterplots
plt.scatter(x_coord,y_coord,color="red")

#excercise do scatterplot between HP and Defense from pokemon data
import pandas as pd
df=pd.read_csv("pokemon_data.csv")
plt.scatter(df['HP'],df['Defense'],color="blue")

#histograms 
import pandas as pd
df=pd.read_csv("height_gender-1.csv")

#plot histogram of male heights
df_male=df.loc[df["Gender"]=="Male"]

#get all female
df_female=df.loc[df["Gender"]=="Female"]

#histogram for males
plt.hist(df_male["Height"],bins=20,edgecolor="black",label="male height",color="red")
plt.legend()
plt.xlabel("Height")
plt.ylabel("Frequency")

#do same thing for female height
plt.hist(df_female["Height"],bins=20,edgecolor="black",label="female height",color="green")
plt.legend()
plt.xlabel("Height")
plt.ylabel("Frequency")

#plot female and male together
plt.hist(df_male["Height"],bins=20,edgecolor="black",label="male height",color="red",alpha=0.5)
plt.hist(df_female["Height"],bins=20,edgecolor="black",label="female height",color="green",alpha=0.5)
#alpha controls the degree of transparency
plt.legend()
plt.xlabel("Height")
plt.ylabel("Frequency")

#use pkmn data to plot HP, Defense and attack on the same plot. 
import pandas as pd
df=pd.read_csv("pokemon_data.csv")
plt.hist(df['HP'],bins=20,edgecolor ='Black', label ='HP',  color= 'green',alpha=0.5)
plt.hist(df['Defense'],bins=20,edgecolor ='Black', label ='Defense',  color= 'red',alpha=0.5)
plt.hist(df['Attack'],bins=20,edgecolor ='Black', label ='Attack',  color= 'blue',alpha=0.5)
plt.show()





#import libraries 

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#read tips
tips=pd.read_csv("tips.csv")

#histogram 
sns.histplot(tips["total_bill"],bins=20,color="orange",label="Total bill",kde=True)
plt.legend()

#do same for tips. use green and appropriate labels.
sns.histplot(tips["tip"],bins=20,color="green",label="Tips given",kde=True)
plt.legend()

#scatterplots using sns
sns.relplot(x="total_bill",y="tip",data=tips)
#three variables
sns.relplot(x="total_bill",y="tip",data=tips,hue="sex")
#four variables 
sns.relplot(x="total_bill",y="tip",data=tips,hue="sex",size='smoker')


#class excercise: do scatter plot between HP and defense, use hue as legendary 
pkmn=pd.read_csv("pokemon_data.csv")
sns.relplot(x="HP",y="Defense",data=pkmn,hue="Legendary")


#some fancy plots
sns.jointplot(x="total_bill",y="tip",data=tips)
##fancy plot 2
sns.jointplot(x="total_bill",y="tip",data=tips,hue="sex")
#fancy plot 3
sns.jointplot(x="total_bill",y="tip",data=tips,kind="reg") #this shows line of best fit
#fancy plot 4
sns.jointplot(x="total_bill",y="tip",data=tips,kind="hex")
#fancy plot 5
sns.jointplot(x="total_bill",y="tip",data=tips,kind="kde")


#lazy plot/most popular plot
sns.pairplot(tips)
sns.pairplot(tips,hue="sex")

#boxplots
##usually we are dealing with 2types of data; wide format and long format 
format_wide=pd.read_excel("format.xlsx",sheet_name="wide")
format_long=pd.read_excel("format.xlsx",sheet_name="long")

#long data
sns.boxplot(x="Attribute",y="Value",data=format_long)

#wide format
sns.boxplot(data=format_wide)
plt.ylabel("Value")

#use height gender data and do a boxplot
df=pd.read_csv('height_gender-1.csv')
sns.boxplot(x="Height",y="Gender",data=df)

#do boxplot for pokemon
df=pd.read_csv("pokemon_data.csv")
pd.melt(df)
dflong=pd.melt(df, id_vars=['Generation'], 
               value_vars=[ 'HP', 'Attack', 'Defense','Sp. Atk','Sp. Def', 'Speed'])
sns.boxplot(data=dflong)


























