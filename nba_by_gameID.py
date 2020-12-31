
n = '0012000032'

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
    
    
    bs = {'tm_pt_sum':bs1[1],  'quarterly_score':bs1[5], 'referee':bs1[2]}
    
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
    
    dt = [pl, tm, bs4_tm, bsa_tm, bsmis_tm, bspct_tm, odds, bs, bs4_pl, bsa_pl, bsmis_pl, bspct_pl, fn]
    
    return dt
    
  
dt = pl_tm(n)

dta2 = dt[1]


