#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 00:07:33 2020

@author: g
"""

import pandas as pd
import numpy as np

df = pd.read_csv('nbateamgames.csv')

from nba_api.stats.endpoints import leaguegamefinder
df2 = leaguegamefinder.LeagueGameFinder(game_id_nullable='0012000044').get_data_frames()[0]

dtg=df2.groupby("TEAM_ID")
tm_gm=[group for _, group in dtg]

tl = []

for i in tm_gm:
    if len(i) > 500:
        tl.append(i)

tm1 = tl[0]


def team_avg_logs(x):
    tm=x
    
    tm['DATE'] =pd.to_datetime(tm.GAME_DATE)
        
    tm=tm.sort_values(by='DATE') 
    
    
    tm['WIN'] = np.where(tm['WL']=="W", 1, 0)
    
    
    tm['WIN%']= (tm['WIN'].expanding().sum()/tm['WIN'].expanding().count()).shift()
    
    
    
    kfi=tm.iloc[:, 9:28].rolling(3).mean().shift()
    
    kfi2=tm.iloc[:, 9:28].expanding().mean().shift()
    
    kfi.columns = [str(col) + '_team_3_mean' for col in kfi.columns]
    
    kfi2.columns = [str(col) + '_team_expand_mean' for col in kfi2.columns]
    
    tm.columns = [str(col) + '_team' for col in tm.columns]
    r = pd.concat([tm,kfi2,kfi], axis=1)
    
    r['rests'] =  r.DATE_team.diff().dt.days.fillna(0).astype(int)


    return r.round(2)




t = []

for i in tl:
    t.append(team_avg_logs(i))
    
rt = pd.concat(t)

rth = rt.head()


tm = t[0]

dta = rt.dropna()

dth = dta.head()
list(dta)


dth = dta.head(100)


list(dta)[29:70]

test = dta.head(18100)




X = dta[list(dta)[30:70]]

list(X)
y = dta['WIN_team'] 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=True, test_size=0.25)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter=10000)
logreg.fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))



from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print('Accuracy of LDA classifier on training set: {:.2f}'
     .format(lda.score(X_train, y_train)))
print('Accuracy of LDA classifier on test set: {:.2f}'
     .format(lda.score(X_test, y_test)))


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'
     .format(gnb.score(X_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'
     .format(gnb.score(X_test, y_test)))

from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(X_test, y_test)))
