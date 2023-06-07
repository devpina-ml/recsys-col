# -*- coding: utf-8 -*-
"""collaborative_filtering_pina.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tgbU85kxhU_n2pDKLTMCEQg_ZSnWFSFL

# DF EVENT
"""

import pandas as pd

df = pd.read_json('/content/df_all_events.json')

df

df.event.value_counts()

# Define custom function to calculate rating
def calculate_rating(group):
 
    relevant_maxn = {
        'product_clicked':5,
        'buy_stock':3,
        'sell_stock':2,
        'product_detail_viewed':2,
        'stock_watchlist_initiated':1
        }

    # Mapping of values to ratings
    weight_dict = {
        'product_clicked': 1, 
        'stock_watchlist_initiated': 8, 
        'buy_stock': 10, 
        'sell_stock': 3, 
        'product_detail_viewed':5, 
        #'product_added':6
        }  

    value_counts = group['event'].value_counts()
    #rating = min(value_counts.get('product_clicked',0), relevant_maxn['product_clicked']) * weight_dict['product_clicked'] + min(value_counts.get('stock_watchlist_initiated',0), relevant_maxn['stock_watchlist_initiated']) * weight_dict['stock_watchlist_initiated'] + min(value_counts.get('buy_stock',0), relevant_maxn['buy_stock']) * weight_dict['buy_stock'] + min(value_counts.get('sell_stock',0), relevant_maxn['sell_stock']) * weight_dict['sell_stock'] + min(value_counts.get('product_detail_viewed',0), relevant_maxn['product_detail_viewed']) * weight_dict['product_detail_viewed'] #+ value_counts.get('product_added',0) * weight_dict['product_added']
    rating = min(value_counts.get('product_clicked',0), relevant_maxn['product_clicked']) * weight_dict['product_clicked'] + min(value_counts.get('stock_watchlist_initiated',0), relevant_maxn['stock_watchlist_initiated']) * weight_dict['stock_watchlist_initiated'] + value_counts.get('buy_stock',0) * weight_dict['buy_stock'] + min(value_counts.get('sell_stock',0), relevant_maxn['sell_stock']) * weight_dict['sell_stock'] + min(value_counts.get('product_detail_viewed',0), relevant_maxn['product_detail_viewed']) * weight_dict['product_detail_viewed']
    return pd.Series({'rating': rating})

# Apply the custom function to each group
df = df.groupby(['username', 'stock_code']).apply(calculate_rating).reset_index()

import numpy as np

def scal_rating(df):

  ratingScal = np.zeros(len(df))

  for name in df.username.unique():

    idx = df[df['username'] == name].index.tolist()
    max_val = df['rating'][df['username'] == name].max()
    value = df['rating'][df['username'] == name].apply(lambda x: 5*(x/max_val)).values

    for k,v in enumerate(idx):
      ratingScal[v] = value[k]

  return ratingScal

#df['ratingScal'] = df['rating'].apply(lambda x: 5*(x/df['rating'].max()))

df['ratingScal'] = scal_rating(df)

df

"""# DF BALANCE"""

import pandas as pd
import numpy as np

df_user = pd.read_csv("/content/user_portfolio.csv", header=0)

df_user

df_user = df_user.drop_duplicates().reset_index(drop=True)

df_user = df_user.copy().iloc[:,:3]

df_user

# Make explicit rating for each user-item
def rating(df, column):
  rat = np.zeros(len(df))

  for i in df['clientportfolioclientid'].unique():

    idx = df[df['clientportfolioclientid']==i].index
    total_quantity = sum(df[column][df['clientportfolioclientid']==i])

    quantity = [1 + 4*(j/total_quantity) for j in df[column][df['clientportfolioclientid']==i]]

    for k,v in enumerate(idx):
      rat[v] = quantity[k]

  return rat

rating_balance = rating(df_user, 'clientportfoliobalance')
#rating_std = rating(df_user, 'scaler_std')
#rating_minmax = rating(df_user, 'scaler_minmax')

df_user['ratingBalance'] = rating_balance
#df_user['rating_std'] = rating_std
#df_user['rating_minmax'] = rating_minmax

df_user

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
sns.histplot(data=df_user, x='ratingBalance', kde=True)
plt.show()

a = df_user.groupby('clientportfolioclientid')['clientportfolioclientid'].count()

a.loc[a>5]

import scipy.sparse as sp

sparse_matrix = df_user.pivot(index='clientportfolioclientid', columns='clientportfoliostockid', values='clientportfoliobalance').fillna(0)

sparse_matrix

"""# Implicit"""

!pip install implicit

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.evaluation import precision_at_k, mean_average_precision_at_k
from sklearn.preprocessing import LabelEncoder

df_user = df.copy()

df_user

# Step 2: Encode Categorical Variables
label_encoder_stock = LabelEncoder()
label_encoder_client= LabelEncoder()
df_user['stock_id'] = label_encoder_stock.fit_transform(df_user['stock_code'])
df_user['username_id'] = label_encoder_client.fit_transform(df_user['username'])

df_user

sparse_item_user = csr_matrix((df_user['ratingScal'].astype(float), (df_user['stock_id'], df_user['username_id'])))
sparse_user_item = csr_matrix((df_user['ratingScal'].astype(float), (df_user['username_id'], df_user['stock_id'])))

# Step 4: Collaborative Filtering Algorithm Selection
model = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=20, alpha=15, random_state=42)

#Fit the model
model.fit(sparse_user_item)
# Step 5: Model Training
#model.fit(user_item_matrix)

le_name_mapping = dict(zip(label_encoder_client.classes_, label_encoder_client.transform(label_encoder_client.classes_)))
print(le_name_mapping)

user_id = 0

# Use the implicit recommender.
recommended = model.recommend(user_id, sparse_user_item[user_id])

print(recommended)

stocks = []
scores = []

# Get artist names from ids

for i in recommended[0]:
  stocks.append(df_user.stock_code.loc[df_user.stock_id == i].values[0])

for j in recommended[1]:
  scores.append(j)

# Create a dataframe of artist names and scores
recommendations = pd.DataFrame({'stoks': stocks, 'score': scores})

recommendations

df_user.loc[df_user['username']=='AGILCAHY48']

def evaluate_model(model, sparse_user_item, k=10):
    # Generate recommendations for all users
    all_recommendations = []
    for user_id in range(sparse_user_item.shape[0]):
        recommendations = model.recommend(user_id, sparse_user_item[user_id], N=k)
        all_recommendations.append([rec[0] for rec in recommendations])

    # Prepare ground truth data
    ground_truth = sparse_user_item.tocsr()

    # Calculate precision at K
    precision = precision_at_k(model, sparse_user_item, k=k)

    # Calculate mean average precision (MAP) at K
    map_score = mean_average_precision_at_k(model, sparse_user_item, k=k)

    return precision, map_score

# Evaluate the model
precision_at_10, map_score = evaluate_model(model, sparse_user_item, k=10)
print("Precision at K:", precision_at_10)
print("Mean Average Precision (MAP):", map_score)

user_id = label_encoder.transform(['YULISPRA20'])  # Replace 'udin2019' with the desired user ID
recommendations = model.recommend(user_id[0], user_item_matrix, N=10)  # Get top 10 recommendations
decoded_recommendations = label_encoder.inverse_transform([rec[0] for rec in recommendations])
print("Recommendations for User", user_id[0], ":", decoded_recommendations)

"""# Pyspark"""

!pip install pyspark

check_df = df.copy()

df = df.rename(columns={'stock_code':'stockcode'})

df

df.username.nunique()

from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder.getOrCreate()

# Assuming your Pandas DataFrame is named "pandas_df"
df = spark.createDataFrame(df)

df.printSchema()

'''from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder.appName("CSV Read").getOrCreate()

# Read the CSV file
df = spark.read.csv("/content/user_portfolio.csv", header=True, inferSchema=True)

# Show the data
df.show()'''

columns_to_drop = ["rating"]
#columns_to_drop = ["clientportfoliodate", "date_rank_per_user"]
df = df.drop(*columns_to_drop)

"""ini dipake klo ratingBalance udah ada dari processing di pandas"""

df.show()

import pyspark.sql.functions as sql_func
from pyspark.sql.types import *
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Create a StringIndexer to convert the categorical column to numeric
indexer = StringIndexer(inputCol="username", outputCol="userIndex")

# Fit and transform the data to obtain the indexed column
indexedData = indexer.fit(df).transform(df)

indexedData.show()

# Create a StringIndexer to convert the categorical column to numeric
indexer = StringIndexer(inputCol="stockcode", outputCol="itemIndex")
# Fit and transform the data to obtain the indexed column
indexedData = indexer.fit(indexedData).transform(indexedData)

indexedData.show()

(training, test) = indexedData.randomSplit([0.8, 0.2], 0)

training.select('username').distinct().count()

test.show()

# Check value ada atau tidak di df
training.filter(training['userIndex'] == 380).count()

als = ALS(userCol="userIndex", itemCol="itemIndex", ratingCol="ratingScal",
          coldStartStrategy="drop", implicitPrefs=True)

# Create the parameter grid for tuning
param_grid = ParamGridBuilder() \
    .addGrid(als.alpha, [0.1, 1, 40]) \
    .addGrid(als.rank, [10, 20, 30, 40]) \
    .addGrid(als.maxIter, [5, 10, 15, 20]) \
    .addGrid(als.regParam, [0.01, 0.05, 0.1]) \
    .build()

# Create the evaluator for model selection
evaluator = RegressionEvaluator(metricName="rmse", labelCol="ratingScal")

# Create the cross-validator for tuning
cross_validator = CrossValidator(estimator=als,
                                estimatorParamMaps=param_grid,
                                evaluator=evaluator,
                                numFolds=5)

# Fit the cross-validator on the training data
cv_model = cross_validator.fit(indexedData)

print(cv_model.avgMetrics)
print(cv_model.avgMetrics[0])

# Make predictions on the test data
best_model = cv_model.bestModel
predictions = best_model.transform(indexedData)
#predictions = cv_model.transform(test)

# Evaluate the model performance
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE):", rmse)

# Get the best model from the cross-validator
#best_model = cv_model.bestModel
rank = best_model.rank
alpha = best_model._java_obj.parent().getAlpha()
max_iter = best_model._java_obj.parent().getMaxIter()
reg_param = best_model._java_obj.parent().getRegParam()

print("Best model parameters: Rank =", rank,", Alpha =", alpha, ", Max Iter =", max_iter, ", Reg Param =", reg_param)

"""Root Mean Squared Error (RMSE): 0.14727512634598997
Best model parameters: Rank = 40 , Alpha = 0.1 , Max Iter = 20 , Reg Param = 0.1
"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# # Generate top 10 stock recommendations for each user
# userRecs = best_model.recommendForAllUsers(10)
# userRecs.count()
# # Generate top 10 user recommendations for each stock
# stockRecs = best_model.recommendForAllItems(10)
# stockRecs.count()

userRecs_df = userRecs.toPandas()
print(userRecs_df.shape)

stockRecs_df = stockRecs.toPandas()
print(stockRecs_df.shape)

stockRecs_df.head()

userRecs_df.head()

indexStockIdPairs = indexedData.select("itemIndex", "stockcode").toPandas()

dictionary = indexStockIdPairs.groupby('itemIndex')['stockcode'].apply(lambda x: list(set(x))).to_dict()

indexUserPairs = indexedData.select("userIndex", "username").toPandas()

user_dictionary = indexUserPairs.groupby('userIndex')['username'].apply(lambda x: list(set(x))).to_dict()

user_dictionary[642]

def get_recom(df):
  stck_rec_all = []

  for i in df['recommendations']:
    stck_rec = []

    for j in i:
      stck_rec.append(dictionary[j['itemIndex']][0])

    stck_rec_all.append(stck_rec)
    
  return stck_rec_all

userRecs_df['recommendations_stock_name'] = get_recom(userRecs_df)

userRecs_df['username'] = userRecs_df['userIndex'].apply(lambda x: user_dictionary[x][0])

userRecs_df

userRecs_df['recommendations'][userRecs_df['userIndex']==1].values

"""## Recommendation"""

username = userRecs_df['username'][userRecs_df['userIndex']==1].values[0]

inv = [i for i in check_df['stock_code'][check_df['username']==username].values]

not_inv = [i for i in userRecs_df['recommendations_stock_name'][userRecs_df['username']==username].values[0]]

recommendation = [i for i in not_inv if i not in inv]

inv

recommendation

"""# Surprise"""

!pip install surprise

df

import pandas as pd
from surprise import Dataset, Reader
from surprise import KNNBasic, accuracy
from surprise.model_selection import cross_validate, train_test_split
from collections import defaultdict

# Define the rating scale
reader = Reader(rating_scale=(0, 5))

# Load the data into the Surprise Dataset format
data = Dataset.load_from_df(df[['username', 'stock_code', 'ratingScal']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2)

from surprise.model_selection import GridSearchCV
from surprise import KNNBasic
from surprise.model_selection.split import KFold

# Define the parameter grid to search over
param_grid = {'k': [10, 20, 30, 50, 80],
              'min_k': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'sim_options': {'name': ['cosine'],
                              'user_based': [True]}
               
             }

cv = KFold(n_splits=5, random_state=42, shuffle=True)

# Define the grid search object
grid_search = GridSearchCV(algo_class=KNNBasic, param_grid=param_grid, measures=['rmse'], cv=cv, n_jobs=-1)

# Fit the grid search object to the data
grid_search.fit(data)

# Print the best RMSE score and the corresponding parameters
print('Best RMSE score: {:.2f}'.format(grid_search.best_score['rmse']))
print('Best parameters: ', grid_search.best_params['rmse'])

best_algo = KNNBasic(k=grid_search.best_params['rmse']['k'],
                     min_k=grid_search.best_params['rmse']['min_k'],
                     sim_options=grid_search.best_params['rmse']['sim_options'])
best_algo.fit(data.build_full_trainset())

# Use the algorithm to make predictions on the testset
#predictions = best_algo.test(testset)
# Evaluate the performance of the algorithm
#accuracy.rmse(predictions)

best_algo.predict('OKI', 'BEKS')

all_stock = df['stock_code'].unique()

invest = df['stock_code'][df['username']=='ALDIALHA39'].values

not_invest = [i for i in all_stock if i not in invest]

score = [best_algo.predict('ALDIALHA39', stock).est for stock in not_invest]

result = pd.DataFrame({'stock':not_invest, 'pred_score':score})
result.sort_values('pred_score', ascending=False, inplace=True)
result.head(10)

df.loc[df['username']=='ALDIALHA39']