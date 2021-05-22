#Source: https://en.wikipedia.org/wiki/Pythagorean_expectation


def pyt_exp(points_for, points_against):
    win = (points_for ** 13.91)/ ((points_for ** 13.91) + points_against ** 13.91)
    return win

pyt_exp(921,640)


