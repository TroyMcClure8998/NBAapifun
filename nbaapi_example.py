

from nba_api.stats.endpoints import leaguegamefinder
df = leaguegamefinder.TeamEstimatedMetrics(game_id_nullable='0012000044').get_data_frames()[0]

from nba_api.stats.endpoints import teamestimatedmetrics
df = teamestimatedmetrics.TeamEstimatedMetrics().get_data_frames()[0]

list(df)
from nba_api.stats.endpoints import boxscorefourfactorsv2
df2p = boxscorefourfactorsv2.BoxScoreFourFactorsV2(game_id='0012000044').get_data_frames()

df2 = df2p[1]


from nba_api.stats.endpoints import boxscoreadvancedv2
df2pa = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id='0012000044').get_data_frames()

df2a = df2pa[1]


from nba_api.stats.endpoints import boxscoresummaryv2
a = boxscoresummaryv2.BoxScoreSummaryV2(game_id='0012000032').get_data_frames()

a1 = a[1]


from nba_api.stats.endpoints import boxscorescoringv2
a2 = boxscorescoringv2.BoxScoreScoringV2(game_id='0012000032').get_data_frames()

a21 = a2[1]


from nba_api.stats.endpoints import boxscoremiscv2
b = boxscoremiscv2.BoxScoreMiscV2(game_id='0012000032').get_data_frames()

b1 = b[1]


from nba_api.stats.endpoints import winprobabilitypbp
bm = winprobabilitypbp.WinProbabilityPBP(game_id='0012000032').get_data_frames()


bm1 = bm[0]



from nba_api.stats.endpoints import infographicfanduelplayer
fn = infographicfanduelplayer.InfographicFanDuelPlayer(game_id='0012000032').get_data_frames()


fn1 = fn[0]
