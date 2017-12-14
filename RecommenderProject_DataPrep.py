
# coding: utf-8

# # Project Recommending Music with Audioscrobbler Data
# 
# 
# ## Exploring Audioscrobbler (lastfm) data

# In[1]:

#get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql.types import StructType, StructField, DoubleType, LongType, StringType, Row
import pyspark.sql.functions as func
from datetime import datetime


# In[2]:

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
spark


# Read data

# In[3]:

# path to files
artistdata_path = './data/artist_data.csv'
#userartist_path = './data/user_artist_data_train_small.csv'
userartist_path = './data/user_artist_data_train.csv'


# In[4]:

# Schemas for both files
artistdata_struct = StructType([StructField('artistId', LongType()),                                 StructField('name', StringType())])
userartist_struct = StructType([StructField('userId', LongType()),                                 StructField('artistId', LongType()),                                 StructField('count', LongType())])


# In[5]:

# read artist_data file
artistdata_df = spark.read.csv(artistdata_path, sep = '\t', schema = artistdata_struct)
artistdata_df.cache()
artistdata_df.take(10)


# In[21]:

# read user_artist_data file
userartist_df = spark.read.csv(userartist_path, sep = '\t', schema = userartist_struct)
userartist_df.cache()
userartist_df.take(10)


# Summary statistics

# In[22]:

# some statistics on user-artist dataframe
userartist_df.describe().show()


# <font color=blue>
# 
# Find 20 most popular artists.
# 
# **Hint**: Use the following methods.
# 
# `DataFrame.groupBy('cols')` - groups DataFrame using the specified columns `cols` to prepare for aggregation on them
# 
# `GroupedDataFrame.agg(exprs)` - computes aggregates and returns result as a DataFrame. Available aggregate functions are `avg`, `max`, `min`, `sum`, `count`.
# 
# Use `.join()` to add column of real names of the artists

# In[23]:

# Create dataframe
#salespeople=sc.parallelize(['1\tHenry\t100',
#                           '2\tKaren\t100',
#                           '3\tPaul\t101',
#                           '4\tJimmy\t102',
#                           '5\tJanice\t103']) \
#.map(lambda x: x.split('\t')) \
#.map(lambda x: (int(x[0]),x[1],int(x[2])))
#salespeople_df=sqlContext.createDataFrame(salespeople,['Num','Name','Store'])
#print('Original DataFrame')
#salespeople_df.show()

# Group the dataframe by store
#salespeople_df_grouped=salespeople_df.groupBy('Store')

# Aggregate using `count`
#print('Grouped and Aggregated by "count"')
#salespeople_df_grouped.agg({'Store': "count"}).show()

# Alternatively, use pyspark.sql.functions as func and count group members after assigning 
# each of them literal(constant) value 1 by func.lit(1)
#print('Grouped and aggregated by count after lit(1)')
#salespeople_df_grouped.agg(func.count(func.lit(1)).alias('New column')).show()

# Aggregate by sum of column 'Num'
#print('Grouped and aggregated by "sum" of column "Num"')
#salespeople_df_grouped.agg(func.sum('Num')).show()


# <font color=blue>
# 
# In the following cell enter code selecting 20 most popular artists by creating DataFrame `artists` of format below. <br>
# Cache object `artists`.
# 
# Order the table by number of users who listened each artist.
# 
# `artists.cache()` <br>
# 
# `artists.orderBy("num_of_users", ascending=False).show(20)`
#  
# +--------+-----------+------------+--------------------+ <br>
# |artistId|total_count|num_of_users|                name| <br>
# +--------+-----------+------------+--------------------+ <br>
# | 2000710|     272386|         641|           Radiohead| <br>
# | 2003097|     215464|         584|           Green Day| <br>
# | 2004180|     226094|         578|         The Beatles| <br>
# | 2000708|     131860|         573|             Nirvana| <br>
# | 2000842|      80399|         541|            Coldplay| <br>
# | 2002433|      98676|         541|              Weezer| <br>
# | 2000914|      99656|         540|Red Hot Chili Pep...| <br>
# | 2000137|      99322|         505|            The Cure| <br>
# | 2028381|      22950|         499|           [unknown]| <br>
# | 2000868|      99696|         498|                  U2| <br>
# | 2155446|     110588|         497|              R.E.M.| <br>
# | 2000088|      70315|         496|        Beastie Boys| <br>
# | 2005113|      99871|         489|The Smashing Pump...| <br>
# | 2004129|     182005|         488|           Metallica| <br>
# | 2000061|     178506|         488|          Pink Floyd| <br>
# | 2004152|      67627|         486|        Foo Fighters| <br>
# | 2005175|     159844|         480|        Modest Mouse| <br>
# | 2000940|      58010|         478|   The White Stripes| <br>
# | 2001006|      95648|         477|        Led Zeppelin| <br>
# | 2013825|      73260|         476|Rage Against the ...| <br>
# +--------+-----------+------------+--------------------+ <br>
# only showing top 20 rows <br>

# In[24]:

# Skipped code
# top 20 most popular artists

# Renaming artistId before joining
userartist_df = userartist_df.selectExpr("artistId as artistId2","userId as userId","count as count")
userartist_df.show(5)

# Joining userartist with artistdata
c = userartist_df.join(artistdata_df,userartist_df.artistId2 == artistdata_df.artistId)
c= c.drop('artistId2')
c.show(5)

# Group the dataframe by artistid/name
c_grouped=c.groupBy('artistId','name')


# Aggregate by Total count
print('Grouped and Aggregated by sum of "count"')
d = c_grouped.agg(func.sum('count').alias('total_count'))
d.show(5)

# Aggregate using No of users

print('Grouped and Aggregated by no. of users for each artist')
e= c_grouped.agg({'userId': "count"}).withColumnRenamed("count(userId)", "num_of_users")
e.show(5)

# Join the two dataframes d and e created above to form artists
d = d.selectExpr("artistId as artistId2","name as name2","total_count as total_count")
artists = d.join(e,d.artistId2 == e.artistId)
artists= artists.drop('artistId2','name2')
artists = artists.select("artistId","total_count","num_of_users","name")

artists.cache() 
print('Top 20 artists')
artists.orderBy("num_of_users", ascending=False).show(20)


# Note that the 10-th most popular artist is *[unknown]*. It is an artifact of our dataset.
# 
# Maybe there are some other artifacts (such as websites instead of artist names)?
# 
# It might be a good idea to clean the dataset before doing deep analysis. <br>
# 
# Data in both files have been cleaned for you relative to the version available publicly.
# 
# **But some obvious problems are still there. <br>
# Explore the dataset further and try to clean up the data as much as you can. <br>
# It may help improving the score**.

# <font color=blue>
# 
# Find top most active users. <br>
# Create object `users` of the of the format below. <br>
# Again, cache the object and show it sorted by number of artists played by each user . <br>
# 
# `users.cache()`
# 
# `users.orderBy("num_of_artists", ascending=False).show(10)`
# 
# +-------+-----------+--------------+ <br>
# | userId|total_count|num_of_artists| <br>
# +-------+-----------+--------------+ <br>
# |1030183|      37674|           705| <br>
# |1010710|      14498|           688| <br>
# |1006800|      39260|           661| <br>
# |1027581|      22037|           621| <br>
# |1006344|      49817|           581| <br>
# |1007183|      13811|           561| <br>
# |1007977|      16108|           557| <br>
# |1006104|      15784|           548| <br>
# |1057406|      25337|           539| <br>
# |1007882|      15970|           524| <br>
# +-------+-----------+--------------+ <br>
# only showing top 10 rows <br>

# In[25]:

# Skipped code
# top 10 most active users
# Group the dataframe by userid
c_grouped=c.groupBy('userId')


# Aggregate by Total count
print('Grouped and Aggregated by sum of "count"')
d = c_grouped.agg(func.sum('count').alias('total_count'))
d.show(5)

# Aggregate using No of users

print('Grouped and Aggregated by no. of artists for each user')
e= c_grouped.agg({'artistId': "count"}).withColumnRenamed("count(artistId)", "num_of_artists")
e.show(5)

# Joining d and e to get users
d = d.selectExpr("userId as userId2","total_count as total_count")
users = d.join(e,d.userId2 == e.userId)
users= users.drop('userId2')
users = users.select("userId","total_count","num_of_artists")

users.cache() 
print('Top 10 users')
users.orderBy("num_of_artists", ascending=False).show(10)


# Create table of user-artist pairs with number of times played.

# In[37]:

userartist_df.orderBy("count", ascending=False).show(10)


# **Note that some users have played some artists songs enormous number of times! <br>
# This is also an artifact (someone's mistake, spam-attack, etc). <br>
# It seems reasonable to remove all records with large *count* value**.

# <font color=blue>
# 
# Calculate sparsity coefficient of user-artist matrix defined as
# $$\frac{Number~of~rows~in~data~set}{Number~of~users~\times~Number~of~artists}$$
# 
# Enter code in the following cell
# 
# `Sparsity Сoeff = 0.1819149149149149`

# In[27]:

# Skipped code
# sparsity coefficient of user-artist matrix

# Number of rows in dataset
print("No. of rows in dataset = ")
print(userartist_df.count())

# Number of artists
print("No. of artists = ")
print(userartist_df.select('artistId2').distinct().count())

# Number of users
print("No. of users = ")
print(userartist_df.select('userId').distinct().count())

# Sparsity Coefficient

print("Sparsity Coefficient = ")
print((userartist_df.count())/((userartist_df.select('userId').distinct().count())*(userartist_df.select('artistId2').distinct().count())))


# <font color=blue>
# 
# Note that user-artist matrix is *very* sparse.
#/plt
 
# Plot histogram of user counts per artist. <br>
# 
# **Hint**. Create a list of user counts per artist, save the data to file, then read the file and create histogram similar to the example in notebook "Linear Regression in Spark MLLib" 
# 
# Enter code in the following 2 cells.

# In[31]:

# Skipped code
# save data for histogram of users per artist
artists.show(5)
print("Saving this dataframe")
artists.coalesce(1).write.option("header", "true").csv('./data/artists.csv')


# In[32]:

#Skipped code
#plot histogram of artists' popularity
artists2 = spark.read.csv('./data/artists.csv', sep = ',',header=True,inferSchema=True)
#plt.hist(artists2.select('num_of_users').rdd.flatMap(lambda x:x).collect())
#plt.title('Histogram of artists popularity- no. of users listened by')


# <font color=blue>
# 
# Plot histogram of artists per user.
# 
# Enter code in the following 2 cells.

# In[33]:

# Skipped code
# save data for histogram of artists per user
# We save the users dataframe created earlier
users.show(5)
print("Saving this dataframe")
users.coalesce(1).write.option("header", "true").csv('./data/users.csv')


# In[34]:

#Skipped code
#plot histogram of users' activity
users2 = spark.read.csv('./data/users.csv', sep = ',',header=True,inferSchema=True)
#plt.hist(users2.select('num_of_artists').rdd.flatMap(lambda x:x).collect())
#plt.title('Histogram of users activity- no. of artists listened to')


# Check large data set for number of users who played only very few artists and number of artists almost never played by any user. <br>
# If a lot of artists have only few listeners that may be not good. <br>
# Also if many users have only few played artists it may cause a problem. <br>
# Should we also remove such users and artists before fitting the model?

# In[51]:

# As seen from above histogram, majority of artists had between 0-5000 users. Also majortiy of users listened to 0-700 artists 
# with some going upto 2000 artists. 

# Checking summary statistics of the count column again-
userartist_df.describe('count').show()
# Mean is only 15 but max is 439771. Checking the top 100 records-
userartist_df.orderBy("count", ascending=False).show(100)


# In[60]:

# It seems reasonable to remove records with more than 10000 counts. The high values are skewing the data
userartist_df2 = userartist_df.filter(userartist_df['count'] < 10000)
userartist_df2.orderBy("count", ascending=False).show(10)
userartist_df2.describe().show()


# In[67]:

# Saving the new train file
userartist_df2.coalesce(1).write.option("header", "true").csv('./data/userartist_new.csv')

