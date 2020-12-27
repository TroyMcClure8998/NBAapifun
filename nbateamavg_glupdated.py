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


    return r.round(2)




t = []

for i in tl:
    t.append(team_avg_logs(i))
    
rt = pd.concat(t)

rth = rt.head()
