import pandas as pd
import numpy as np
from surprise import dump
import os

df = pd.read_json('./df_all_events.json')

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
        }

    value_counts = group['event'].value_counts()
    rating = min(value_counts.get('product_clicked',0), relevant_maxn['product_clicked']) * weight_dict['product_clicked'] + min(value_counts.get('stock_watchlist_initiated',0), relevant_maxn['stock_watchlist_initiated']) * weight_dict['stock_watchlist_initiated'] + value_counts.get('buy_stock',0) * weight_dict['buy_stock'] + min(value_counts.get('sell_stock',0), relevant_maxn['sell_stock']) * weight_dict['sell_stock'] + min(value_counts.get('product_detail_viewed',0), relevant_maxn['product_detail_viewed']) * weight_dict['product_detail_viewed']
    return pd.Series({'rating': rating})

# group df based on username and stock code
df = df.groupby(['username', 'stock_code']).apply(calculate_rating).reset_index()

# scalarization rating
def scal_rating(df):

  ratingScal = np.zeros(len(df))

  for name in df.username.unique():

    idx = df[df['username'] == name].index.tolist()
    max_val = df['rating'][df['username'] == name].max()
    value = df['rating'][df['username'] == name].apply(lambda x: 5*(x/max_val)).values

    for k,v in enumerate(idx):
      ratingScal[v] = value[k]

  return ratingScal

df['ratingScal'] = scal_rating(df)

# load save model
def load_model(model_filename):
    
    file_name = os.path.expanduser(model_filename)
    _, loaded_model = dump.load(file_name)
    return loaded_model

#'ALDIALHA39'
def get_collaborative(algo, username, df=df, top_n=5):

    all_stock = df['stock_code'].unique()

    invest = df['stock_code'][df['username'] == username].values

    not_invest = [i for i in all_stock if i not in invest]

    score = [algo.predict(username, stock).est for stock in not_invest]

    result = pd.DataFrame({'stock':not_invest, 'pred_score':score})

    result.sort_values('pred_score', ascending=False, inplace=True)

    return result.head(top_n).to_json(orient='records')
