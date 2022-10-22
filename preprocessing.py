import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('song_data.csv', index_col = 0)
# list of all the columns that have null values
null_col_list = []
for col in filter((lambda x : df[x].isnull().sum() > 0), df.isnull().sum().index):
    null_col_list.append(col)
# filling the null values with median value of the respective columns
for col in null_col_list:
    median = df[col].median()
    df[col].fillna(median, inplace = True)
X = df.drop('song_popularity', axis= 1)
y = df['song_popularity']
# normalizing the data
scaler = StandardScaler()
scaler.fit(X)
X_trans = scaler.transform(X)
# using Logistic regression model
model = LogisticRegression()
model.fit(X_trans,y)
# dumping the model in pickle file
with open('model.pkl', 'wb') as files:
    pickle.dump(model, files)
