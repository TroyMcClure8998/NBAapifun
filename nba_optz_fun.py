from pulp import *


def optz(dt):
    
  
    # PROBLEM DATA:
    costs = dt['dk_salary'].tolist()
    profits = dt['dk_score_x'].tolist()
    pos = dt['dk_position'].tolist()
    team= dt['team_id'].tolist()
    id_name = dt['player_id'].tolist()
    dfk = dt['dk_score_x'].tolist()
    
    max_cost = 50000
    max_to_pick = 8
    # DECLARE PROBLEM OBJECT:
    prob = LpProblem("Mixed Problem", LpMaximize)
    # VARIABLES
    n = len(costs)
    N = range(n)
    x = LpVariable.dicts('x', N, cat="Binary")
    # OBJECTIVE
    prob += lpSum([profits[i]*x[i] for i in N])
    # CONSTRAINTS
    prob += lpSum([x[i] for i in N]) == max_to_pick        # Limit number to include
    prob += lpSum([x[i]*costs[i] for i in N]) <= max_cost  # Limit max. cost
    
    for r in set(team):
      team_list = [i for i in N if pos[i] == r] 
    
    
    # NEW CONSTRAINT
    for c in set(pos):
      index_list = [i for i in N if pos[i] == c] 
      prob += lpSum([x[i] for i in index_list and team_list]) <= 1
    
    # SOLVE & PRINT RESULTS
    prob.solve()
    #print(LpStatus[prob.status])
    ##print('Profit = ' + str(value(prob.objective)))
    #print('Cost = ' + str(sum([x[i].varValue*costs[i] for i in N])))
    
    #for v in prob.variables ():
        #print (v.name, "=", v.varValue)
        
    dfk_chosen = [dfk[i] for i in N if x[i].varValue > 0]
    id_name_chosen = [id_name[i] for i in N if x[i].varValue > 0]
    pos_chosen = [pos[i] for i in N if x[i].varValue > 0]
    costs_chosen = [costs[i] for i in N if x[i].varValue > 0]
    profits_chosen = [profits[i] for i in N if x[i].varValue > 0]
    team_chosen = [team[i] for i in N if x[i].varValue > 0]
    
    
    #print('chosen: ', id_name_chosen,",", pos_chosen,",", costs_chosen,",", profits_chosen,",", team_chosen)
    
    opt = pd.DataFrame({'player':id_name_chosen, 'team':team_chosen, 'pos':pos_chosen, 'Salary':costs_chosen, 'pred':profits_chosen, 'dfk':dfk_chosen    })
    
    opt['score'] = sum(dfk_chosen)
    
    opt['score_pred'] = sum(profits_chosen)
    
    print(sum(dfk_chosen))
    
    return opt
