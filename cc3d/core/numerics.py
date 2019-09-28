def hill(x, l, n):
    """
    Implements Hill Function - https://en.wikipedia.org/wiki/Hill_equation_(biochemistry)
    :param x:
    :param l:
    :param n:
    :return:
    """
    return 1. / (1. + (x / l) ** n)


def mm(x, k, d):
    """
    Implements Michaelis Menten Function - https://en.wikipedia.org/wiki/Hill_equation_(biochemistry)
    :param x:
    :param k:
    :param d:
    :return:
    """
    return k * x / (d + x)
