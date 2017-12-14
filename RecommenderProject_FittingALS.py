
# coding: utf-8

 
# # Project Recommending Music with Audioscrobbler Data
# 
# 
# ## Fitting ALS model to Audioscrobbler (LastFM) data

# In[1]:

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType, Row
from pyspark.ml.recommendation import ALS
import pyspark.sql.functions as func
import random
import time
from datetime import datetime


# In[2]:

spark = SparkSession.builder.getOrCreate()
spark


# ## Data
# 
# Create paths to the data files. Add path to file with predictions for the test that will be calculated at the end of this notebook.  

# In[3]:

# paths to files
artistdata_path = './data/artist_data.csv'
userartist_path = './data/userartist_new.csv'
test_path = './data/LastFM_Test_Sample.csv'


# In[4]:

# defining schemas
artistdata_struct = StructType([StructField('artistId', IntegerType()),                                 StructField('name', StringType())])


# In[5]:

# read artist names data
artistdata_df = spark.read.csv(artistdata_path, sep = '\t', schema = artistdata_struct)
artistdata_df.cache()
artistdata_df.show(10)


# In[6]:

# read user-artist data (the final trained file in DataPrep notebook)
userartist_df = spark.read.csv('./data/userartist_new.csv', sep = ',',header=True,inferSchema=True)
userartist_df = userartist_df.selectExpr("artistId2 as artistId","userId as userId","count as count")
userartist_df.cache()
userartist_df.show(10)


# In[7]:

# split data:
(training, test) = userartist_df.randomSplit([0.9, 0.1], seed=0)
training.cache()
# remove 'count' column from test:
test = test.drop('count')
test.cache()
test.show(10)


# ## Fitting model
# 
# Fit the ALS model. <br>
# Hyperparameters to specify: <br>
# 
# -  `rank` between 5 and 40; default 10; the number of latent factors in the model
# -  `regParam` between 0.01 and 8; default 0.1; regularization parameter $\lambda$
# -  `alpha` between 1 and 40; default 1; parameter $\alpha$ appears in the expression for confidence $$c_{u,i}=1+\alpha r_{u,i}$$ or $$c_{u,i}=1+\alpha \ln(1+\frac{r_{u,i}}{\epsilon}).$$ If $\alpha=0$  confidence is always 1 regardless of rating$r_{u,i}$. As $\alpha=0$ grows we pay more and more attention to how many times user $u$ consumed item $i$. Thus $\alpha$ controls the relative weight of observed versus unobserved ratings. 
# 
# Search for hyperparameters on the grid of 4-5 values in each range.

# In[44]:

# building a model
# Note that there are some hyperparameters, that should be fitted during cross-validation 
# (here we use default values for all hyperparameters but rank) 
#t1 = time.perf_counter()
#model = ALS(implicitPrefs=True, userCol="userId", itemCol="artistId", ratingCol="count", 
#            rank=3).fit(training)
#t2 = time.perf_counter()
#print('Fitting time:', t2-t1)


# In[46]:

# save model
#timestamp = datetime.now().isoformat(sep='T')
#model.save('./als_model_1')


# In[47]:

# Artist factors:
#model.itemFactors.orderBy('id').take(3)


# In[48]:

# User factors:
#model.userFactors.orderBy('id').take(3)


# ## Evaluating the model using meanAUC-metric

# In[49]:

# predict test data
#predictions = model.transform(test)
#predictions.cache()
#predictions.take(3)


# In[8]:

# broadcast all artist ids
allItemIDs = userartist_df.select('artistId').distinct().rdd.map(lambda x: x[0]).collect()
bAllItemIDs = spark.sparkContext.broadcast(allItemIDs)


# In[9]:

 
# broadcast 10000 most popular artist ids
artists = userartist_df.groupBy('artistId') .agg(func.count(func.lit(1)).alias('num_of_users'))

artists.cache()
top_artists = artists. orderBy('num_of_users', ascending=False).limit(10000). rdd.map(lambda x: x['artistId']).collect()

bTopItemIDs = spark.sparkContext.broadcast(top_artists)


# Calculation of AUC is described in the book Advanced Analytics with Spark.
# 
# In the calculation below parameter `positiveData` has the meaning of "positive" or "good" artist for the user. Parameter `predictFunction` is a function that takes user-item pairs and predicts estimated strength of interactions between them.

# In[10]:

# define meanAUC logic according to 'Advanced Analytics with Spark'

def areaUnderCurve(positiveData, bAllItemIDs, predictFunction):
    positivePredictions = predictFunction(positiveData.select("userId", "artistId"))        .withColumnRenamed("prediction", "positivePrediction")
        
    negativeData = positiveData.select("userId", "artistId").rdd                    .groupByKey()                    .mapPartitions(lambda userIDAndPosItemIDs: 
                                   createNegativeItemSet(userIDAndPosItemIDs, 
                                                         bAllItemIDs))\
                    .flatMap(lambda x: x).map(lambda x: Row(userId=x[0], artistId=x[1])) \
                .toDF()
    
    negativePredictions = predictFunction(negativeData)        .withColumnRenamed("prediction", "negativePrediction")

    joinedPredictions = positivePredictions.join(negativePredictions, "userId")        .select("userId", "positivePrediction", "negativePrediction").cache()
        
    allCounts = joinedPredictions        .groupBy("userId").agg(func.count(func.lit("1")).alias("total"))        .select("userId", "total")
    correctCounts = joinedPredictions        .where(joinedPredictions.positivePrediction > joinedPredictions.negativePrediction)        .groupBy("userId").agg(func.count("userId").alias("correct"))        .select("userId", "correct")

    joinedCounts = allCounts.join(correctCounts, "userId")
    meanAUC = joinedCounts.select("userId", (joinedCounts.correct / joinedCounts.total).                                   alias("auc"))        .agg(func.mean("auc")).first()

    joinedPredictions.unpersist()

    return meanAUC[0]


def createNegativeItemSet(userIDAndPosItemIDs, bAllItemIDs):
    allItemIDs = bAllItemIDs.value
    return map(lambda x: getNegativeItemsForSingleUser(x[0], x[1], allItemIDs), 
               userIDAndPosItemIDs)


def getNegativeItemsForSingleUser(userID, posItemIDs, allItemIDs):
    posItemIDSet = set(posItemIDs)
    negative = []
    i = 0
    # Keep about as many negative examples per user as positive.
    # Duplicates are OK
    while i < len(allItemIDs) and len(negative) < len(posItemIDSet):
        itemID = random.choice(allItemIDs) 
        if itemID not in posItemIDSet:
            negative.append(itemID)
        i += 1
    # Result is a collection of (user,negative-item) tuples
    return map(lambda itemID: (userID, itemID), negative)


# ## Comparing ALS with simple proposals

# In[53]:

# calc Mean AUC using top artists for negative proposals
#t1 = time.perf_counter()
#print('meanAUC =', areaUnderCurve(test, bTopItemIDs, model.transform), 'for ALS-PREDICTION')
#t2 = time.perf_counter()
#print('t2-t1 =', t2-t1)


# In[54]:

# random-prediction logic
#def random_predict(ua_df):
#    return ua_df.withColumn('prediction', func.rand())


# In[55]:

#t1 = time.perf_counter()
#print('meanAUC =', areaUnderCurve(test, bTopItemIDs, random_predict), 'for RANDOM-PREDICTION')
#t2 = time.perf_counter()
#print('t2-t1 =', t2-t1)


# In[56]:

# prediction by global popularity
#def predict_by_global_popularity(ua_df):
#    return ua_df.join(artists, 'artistId').withColumnRenamed("num_of_users", "prediction")                .select('userId', 'artistID', 'prediction')


# In[57]:

#t1 = time.perf_counter()
#print('meanAUC =', areaUnderCurve(test, bTopItemIDs, predict_by_global_popularity), 
#      'for GLOBAL-POPULARITY-PREDICTION')
#t2 = time.perf_counter()
#print('t2-t1 =', t2-t1)


# ### Grid Search on Hyperparameters using the top 10000 artists

# In[11]:

seed = 49247
iterations = 10
#lambdas = [0.01, 0.1, 3.0, 6.0]
#ranks = [5,15,25,40]
#alphas = [0.1, 0.5, 1.0, 5.0, 20.0]

#for lambda_ in lambdas:
#    for rank in ranks:
#        for alpha in alphas:
#            model = ALS(implicitPrefs=True, userCol="userId", itemCol="artistId", ratingCol="count", rank=rank, seed=seed, maxIter=iterations,regParam=lambda_,alpha=alpha).fit(training)
#            print('for rank {0} at alpha: {1} and lambda: {2},'.format(rank, alpha, lambda_))
#            print('meanAUC =', areaUnderCurve(test, bTopItemIDs, model.transform), 'for ALS-PREDICTION')


# ### Now using the best parameters obtained by gridsearch to calculate meanAUC for all artists

# In[12]:

# Model 1- rank= 40 ; alpha= 20; lambda= 0.01 
model1 = ALS(implicitPrefs=True, userCol="userId", itemCol="artistId", ratingCol="count", rank=40, seed=seed, maxIter=iterations,regParam=0.01,alpha=20).fit(training)
print('meanAUC =', areaUnderCurve(test, bAllItemIDs, model1.transform), 'for ALS-PREDICTION for Model 1')

# Model 2- rank= 40; alpha= 20; lambda= 0.1 
model2 = ALS(implicitPrefs=True, userCol="userId", itemCol="artistId", ratingCol="count", rank=40, seed=seed, maxIter=iterations,regParam=0.1,alpha=20).fit(training)
print('meanAUC =', areaUnderCurve(test, bAllItemIDs, model2.transform), 'for ALS-PREDICTION for Model 2')


# ### Based on the results above, we have achieved project goal of getting AUC above 85% (The two models above give 91.93% and 91.96% respectively)
# 
# ### Since Model2's AUC is slightly higher, we will use this model for our Predictions on the Test sample

# ## Predict test data
# 
# From the test shiny download your test sample.
# 
# Use it in the following cell to predict ratings, save the results as csv file and upload back to the test shiny for scoring.
# 
# Of course, predictions obtained without tuning hyperparameters and using small sample are not expected to be good.

# In[10]:

# reading test file
test_struct = StructType([StructField('userId', IntegerType()),                           StructField('artistId', IntegerType())])
test_df = spark.read.csv(test_path, sep = '\t', schema = test_struct)
test_df.show(10)


# In[11]:

# Note that many predictions are NaN since some users and artists might be out of 
# small train-data
# Full train file has to be used to avoid this.
# However, even using full train file, some users might be new. 
# What artists should we propose to them?
predictions = model2.transform(test_df)
predictions.show(10)
assert predictions.count() == test_df.count()


# In[12]:

# Save test predictions to CSV-file
predictions.coalesce(1).write.csv('./data/test_predictions.csv',sep = '\t')


# Check saved results in `./data` directory. <br>
# Solution is saved as a folder with multiple files. <br>
# There should be only one file .csv. Upload it in the test shiny.
