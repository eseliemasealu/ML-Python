import pandas as pd 
pip install mysql-connector-python-rf

import mysql.connector
from mysql.connector import Error
conn = mysql.connector.connect(host='localinstance', database='pydb',
                               user='pydbuser', password='pydbpwd123')
if conn.is_connected():
    print('Connected to MySQL database')
# Prepare a cursor object using cursor() method
cursor = conn.cursor()
# Setting up the variable to fetch data from the specified table
#sql_select_Query = "select * from %s" % tablename
cursor.execute('SELECT * FROM pydb.baseball')
sql_data = pd.DataFrame(cursor.fetchall())
sql_data.columns = cursor.column_names

# Close the session
conn.close()

import seaborn as sns
sns.distplot(sql_data['Salary']) 
