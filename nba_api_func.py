# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 14:36:11 2023

@author: sal
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 13:46:17 2023

@author: sal
"""


import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


dt = pd.read_csv(r"C:\Users\sal\Desktop\python\Nba\nba_game_2022-23.csv")

bt = pd.read_csv(r"C:\Users\sal\Desktop\python\Nba\nba_best_odds.csv")


dth = dt.head()

bth = bt.head()

home_dic_ml = dict(zip(bt.home_date_id, bt.ml_home))
away_dic_ml = dict(zip(bt.away_date_id, bt.ml_away))

ml_dic = {**home_dic_ml, **away_dic_ml}



dt['ml_line'] = dt['bet_date_id'].map(ml_dic)


dtm = dt.select_dtypes(include=[np.number])

move_columns = ['TEAM_ID','GAME_ID','bookie_date', 'MIN', 'home', 'Win_Percent', 'ml_line','Wins' ]

dtm = dtm[ move_columns + [col for col in dtm.columns if col not in move_columns]]

dtm = dtm.drop(columns= ['VIDEO_AVAILABLE'])


dtm['game_id'] = '00' + dtm['GAME_ID'].astype(str)

game_ids = dtm['game_id'].tolist()
game_ids = list(set(game_ids))



def pl_tm(gameid):
    from nba_api.stats.endpoints import boxscorefourfactorsv2
    from nba_api.stats.endpoints import boxscoreadvancedv2
    from nba_api.stats.endpoints import boxscoremiscv2
    from nba_api.stats.endpoints import boxscorescoringv2
    from nba_api.stats.endpoints import boxscoresummaryv2
    from nba_api.stats.endpoints import infographicfanduelplayer
    from nba_api.stats.endpoints import winprobabilitypbp
    
    
    bs4 = boxscorefourfactorsv2.BoxScoreFourFactorsV2(game_id=gameid).get_data_frames()
    bsa = boxscoreadvancedv2.BoxScoreAdvancedV2(gameid).get_data_frames()
    bsmis = boxscoremiscv2.BoxScoreMiscV2(game_id=gameid).get_data_frames()
    bspct = boxscorescoringv2.BoxScoreScoringV2(game_id=gameid).get_data_frames()
    bs1 = boxscoresummaryv2.BoxScoreSummaryV2(game_id=gameid).get_data_frames()
    fn = infographicfanduelplayer.InfographicFanDuelPlayer(game_id=gameid).get_data_frames()[0]
    odds = winprobabilitypbp.WinProbabilityPBP(game_id=gameid).get_data_frames()[1]
    
    bs = {'tm_pt_sum':bs1[1], 'quarterly_score':bs1[5], 'referee':bs1[2]}
    
    bs4_tm = bs4[1]
    bs4_pl = bs4[0]
    
    bsa_tm = bsa[1]
    bsa_pl = bsa[0]
    
    bsmis_tm = bsmis[1] 
    bsmis_pl = bsmis[0]
    
    bspct_tm = bspct[1]
    bspct_pl = bspct[0]
    
    
    
    pl = bs4_pl.combine_first(bsa_pl)
    tm = bs4_tm.combine_first(bsa_tm)
    pl = pl.combine_first(bsmis_pl) 
    tm = tm.combine_first(bsmis_tm)
    pl = pl.combine_first(bspct_pl)
    tm = tm.combine_first(bspct_tm)
    
    ref = bs['referee'] 
    
    pl['ref1']=ref['OFFICIAL_ID'][0]
    pl['ref2']=ref['OFFICIAL_ID'][1]
    pl['ref3']=ref['OFFICIAL_ID'][2]

    bs_pt = bs['tm_pt_sum']
    bs_qs = bs['quarterly_score']
    
    tm = tm.combine_first(bs_pt)
    tm = tm.combine_first(bs_qs)
    
    tm['ref1']=ref['OFFICIAL_ID'][0]
    tm['ref2']=ref['OFFICIAL_ID'][1]
    tm['ref3']=ref['OFFICIAL_ID'][2]
     
    pl = pl.combine_first(fn)
    # tm = tm.combine_first(odds.head(1))
    fn['GAME_ID'] = gameid
    
    dt = [pl, tm, bs4_tm, bsa_tm, bsmis_tm, bspct_tm, odds, bs, bs4_pl, bsa_pl, bsmis_pl, bspct_pl, fn]
    
    team_list = [dt[2],dt[3],dt[4],dt[5],dt[7]['quarterly_score'],dt[7]['tm_pt_sum']]
    
    # merge dataframe baase
    tm = team_list[0]
    
    for i in team_list:
        # Merge DataFrames based on 'id'
        tm = pd.merge(tm, i, on='TEAM_ID', suffixes=('', '_y'))
        # Remove columns with suffix '_y'
        tm = tm.loc[:,~tm.columns.str.endswith('_y')]
    
    move_columns = ['GAME_ID','TEAM_ABBREVIATION','TEAM_CITY', 'GAME_DATE_EST','TEAM_CITY_NAME', 'TEAM_NAME','TEAM_NICKNAME', 
                     'TEAM_ID', 'GAME_SEQUENCE']
    
    tmm = tm[move_columns + [col for col in tm.columns if col not in move_columns]]
    tmm = tmm.drop(columns= ['MIN', 'LEAGUE_ID', 'TEAM_WINS_LOSSES'])
    
    
    # Create a list of DataFrames
    player_list = [dt[8],dt[9],dt[10],dt[11],dt[12]]

    # merge dataframe baase
    pt = player_list[0]

    move_columns = ['GAME_ID','TEAM_ABBREVIATION','TEAM_CITY', 'PLAYER_NAME', 'NICKNAME','START_POSITION','COMMENT', 'TEAM_NAME','JERSEY_NUM','PLAYER_POSITION', 'LOCATION', 'TEAM_ID', 'PLAYER_ID','MIN']

    for i in player_list:
        # Merge DataFrames based on 'id'
        pt = pd.merge(pt, i, on='PLAYER_ID', suffixes=('', '_y'))    
        # Remove columns with suffix '_y'
        pt = pt.loc[:,~pt.columns.str.endswith('_y')]
        
    ptm = pt[ move_columns + [col for col in pt.columns if col not in move_columns]]
    ptm['MIN'] = ptm['MIN'].str.split(':').str[0].str.split('.').str[0].astype(int)
    


    return [dt, tmm, ptm]


dk = pl_tm('0022200446')









































