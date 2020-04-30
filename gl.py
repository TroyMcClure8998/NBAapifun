
import pandas as pd 

import numpy as np
def game_logs(x):
    
    from nba_api.stats.endpoints import leaguegamelog
    
    box_usg23 = leaguegamelog.LeagueGameLog(season= x)
    
    lg = box_usg23.get_data_frames()[0]
    
    lg['tm_gm_id']  =  lg['GAME_ID'].astype(str)+"-"+lg['TEAM_ID'].astype(str)
    return lg
    
    

lg=game_logs("2018")


lg.isnull().sum()


g_id=lg["GAME_ID"].values.tolist()


def remove_duplicates(l):
    return list(set(l))

lit=remove_duplicates(g_id)



dtg=lg.groupby("TEAM_ID")
tm_gm=[group for _, group in dtg]

tm=tm_gm[0]

#x from leaguegamelog
def team_avg_logs(x):
    tm=x
    
    tm['DATE'] =pd.to_datetime(tm.GAME_DATE)
        
    tm=tm.sort_values(by='DATE') 
    
    
    tm['WIN'] = np.where(tm['WL']=="W", 1, 0)
    
    
    tm['WIN%']= (tm['WIN'].expanding().sum()/tm['WIN'].expanding().count()).shift()
    
    
    kfi=tm.iloc[:, 9:28].rolling(3).mean().shift()
    
    kfi.columns = [str(col) + '_team_3_mean' for col in kfi.columns]
    
    
    
    tl=tm.columns.tolist()
    
    
    
    kfi[tl[0:5]] =tm.iloc[:, 0:5]
    
    
    kfi[tl[28:32]] =tm.iloc[:, 28:32]
    return kfi.round(2)




tm2=team_avg_logs(tm_gm[0])


