p1 = df1[0]

weights = np.array(list(range(1,11,1))) #weighted by 

p1['dfk_weighted']=p1['dk_score'].rolling(len(weights)).apply(lambda column: np.correlate(column,weights/sum(weights))).shift()

#DFK points 
def dfk(p,t,r,a,s,b,to):
    p = p * 1
    t = t * .5
    r = r * 1.25
    a = a * 1.5
    s = s * 2
    b = b * 2
    to = to * -.5
    dd = [p,t,r,s,a,b]
    tt = [p,t,r,s,a,b]
    dd = len(list(filter(lambda x: x >= 10, dd))) 
    if dd == 2:
        dd = 1.5
    else:
        dd = 0
    tt = len(list(filter(lambda x: x >= 10, tt))) 
    if tt >= 3:
        tt = 3
    else:
        tt = 0
    
    return sum([p,t,r,s,a,b,to,dd,tt])

#examples
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('/Users/g/Desktop/School/ECON 322/nbadata.csv')

dh = df.head()

l = list(dh)
grouped = df.groupby('player_id')

dg =  [group for _, group in grouped]

g = df.groupby('game_id')

gms =  [group for _, group in g]


dt = []



for i in dg:
    
    
    dm = i.sort_values(by='game_date') 
    
    weights = np.array(list(range(1,11,1)))
    
    
    fm = dm[['dk_score','dk_salary','min','pts', 'fg3m','reb', 'ast', 'stl', 'blk', 'tov' , 'pf', 'plus_minus']]
    
    avg=fm.iloc[:, 0:28].rolling(len(weights)).apply(lambda column: np.correlate(column,weights/sum(weights))).shift()
    
    avg.columns = [str(col) + '_avg' for col in avg.columns]
    
    dm= pd.concat([dm,avg], axis = 1)
    
    dm = dm.dropna()
    
    if len(dm) >=10:
        
        dt.append(dm)
    
    
#dfs = ['min','pts', 'fg3m','reb', 'ast', 'stl', 'blk', 'tov', 'dk_score']

dfs = ['dk_score']


def dfk(p,t,r,a,s,b,to):
    p = p * 1
    t = t * .5
    r = r * 1.25
    a = a * 1.5
    s = s * 2
    b = b * 2
    to = to * -.5
    dd = [p,t,r,s,a,b]
    tt = [p,t,r,s,a,b]
    dd = len(list(filter(lambda x: x >= 10, dd))) 
    if dd == 2:
        dd = 1.5
    else:
        dd = 0
    tt = len(list(filter(lambda x: x >= 10, tt))) 
    if tt >= 3:
        tt = 3
    else:
        tt = 0
    
    return sum([p,t,r,s,a,b,to,dd,tt])


fn = pd.concat(dt)

fn = fn.reset_index().drop(columns='index')
d_k = []

for i in range(len(fn)):
    #print(i)
    v = dfk(fn['pts_avg'][i],fn['fg3m_avg'][i],fn['reb_avg'][i],fn['ast_avg'][i], fn['stl_avg'][i],fn['blk_avg'][i], fn['tov_avg'][i])
    d_k.append(v)
    
    
    
fn['fpoint'] = d_k    

fh = fn.head()
l2 = list(fn)

#fn.to_csv('nba_data_for_r.csv', index=False)

fn['home'] = np.where(fn['matchup'].str.contains('@'),0,1)
mean_squared_error(fn['fpoint'], fn['dk_score'])
mean_absolute_error(fn['fpoint'], fn['dk_score'])
