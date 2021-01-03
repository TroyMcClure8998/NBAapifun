def kc(odds, reward, stake):
    
    #odds % expected
    #rward expected value upside
    #stake expected downside risk
    ev = (reward * odds) + ((stake*-1) * (1-odds))
    risk = ev / reward #risk % of capital

    return risk

kc(.57,242,100)
