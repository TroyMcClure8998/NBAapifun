
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


dt = pd.read_csv('/Users/g/Desktop/Python/nba/nba_game_2022-23.csv')

bt = pd.read_csv('/Users/g/Desktop/Python/nba/nba_best_odds.csv')


dth = dt.head()

bth = bt.head()

home_dic_ml =  dict(zip(bt.home_date_id, bt.ml_home))
away_dic_ml =  dict(zip(bt.away_date_id, bt.ml_away))

ml_dic = {**home_dic_ml, **away_dic_ml}



dt['ml_line'] = dt['bet_date_id'].map(ml_dic)


dtm = dt.select_dtypes(include=[np.number]) 
move_columns = ['TEAM_ID','GAME_ID','bookie_date', 'MIN', 'home', 'Win_Percent', 'ml_line','Wins' ]

dtm = dtm[ move_columns + [col for col in dtm.columns if col not in move_columns]]

dtm = dtm.drop(columns= ['VIDEO_AVAILABLE'])

#df = dtm.join(dtm.groupby('TEAM_ID')[list(dtm)[6:]].expanding().mean().reset_index(level=0, drop=True).groupby(dtm['TEAM_ID']).shift().add_prefix('Rolling_Avg_'))

df = dtm.join(dtm.groupby('TEAM_ID')[list(dtm)[6:]].rolling(10).mean().reset_index(level=0, drop=True).groupby(dtm['TEAM_ID']).shift().add_prefix('Rolling_Avg_'))





columns_list = list(df)[:7] + list(df)[26:]


data = df[columns_list]

data.dropna(inplace=True)





from sklearn.ensemble import AdaBoostClassifier  # Import AdaBoostClassifier






X = data.drop(columns= ['Wins', 'GAME_ID'])
y = data['Wins']



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional but can be beneficial for some algorithms)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an AdaBoostClassifier (you can adjust parameters as needed)
clf = AdaBoostClassifier(n_estimators=50, random_state=42)  # Example: 50 base estimators

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')
