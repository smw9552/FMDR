import numpy as np
from spyder_kernels.utils.lazymodules import scipy
import math


def Weibull_four(value, a, b, c, d, mode):
    if mode == 'ec':
        return 10 ** ((np.log(d) + np.log(-np.log(1 - (value - b) / (c - b)))) / a)
    elif mode == 'ce':
        return b + ((c - b) * (1 - np.exp(-np.exp(a * np.log10(value) - np.log(d)))))
    else:
        raise ValueError("Invalid mode. Use 'ec' or 'ce'.")


def Asymptotic_three(value, a, b, c, mode):
    if mode == 'ec':
        return -c * np.log(1 - ((value - a) / (b - a)))
    elif mode == 'ce':
        return a + (b - a) * (1 - np.exp(-value / c))
    else:
        raise ValueError("Invalid mode. Use 'ec' or 'ce'.")


def Michaelis_Menten_three(value, a, b, c, mode):
    if mode == 'ec':
        return -((c * value - a * c) / (value - b))
    elif mode == 'ce':
        return a + ((b - a) / (1 + (c / value)))
    else:
        raise ValueError("Invalid mode. Use 'ec' or 'ce'.")


def Gompertz_four(value, a, b, c, d, mode):
    if mode == 'ec':
        return (np.log(-np.log((value - b) / (c - b))) / a) + d
    elif mode == 'ce':
        return b + ((c - b) * (np.exp(-np.exp((a) * value - d))))
    else:
        raise ValueError("Invalid mode. Use 'ec' or 'ce'.")


def Logistic_five(value, a, b, c, d, e, mode):
    if mode == 'ec':
        return (np.log((((c - b) / (value - b)) ** (1 / e)) - 1) / a) + d
    elif mode == 'ce':
        return b + ((c - b) / (1 + np.exp((a) * (value - d))) ** (e))
    else:
        raise ValueError("Invalid mode. Use 'ec' or 'ce'.")


def Hill(value, a, b, c, mode):
    if mode == 'ec':
        return a / ((c / value - 1) ** (1 / b))
    elif mode == 'ce':
        return 1 / (1 + (a / value) ** b)
    else:
        raise ValueError("Invalid mode. Use 'ec' or 'ce'.")


def Hill_two(value, a, b, mode):
    if mode == 'ec':
        return a * value / (b - value)
    elif mode == 'ce':
        return b * value / (a + value)
    else:
        raise ValueError("Invalid mode. Use 'ec' or 'ce'.")


def Hill_three(value, a, b, c, mode):
    if mode == 'ec':
        return c / ((a / value - 1)**(1 / b))
        #return c - b * np.log(a / value - 1)
    elif mode == 'ce':
        return a / (1 + (c / value) ** b)
        #return a / (1 + np.exp(-(value - c) / b))
    else:
        raise ValueError("Invalid mode. Use 'ec' or 'ce'.")


def Hill_four(value, a, b, c, d, mode):
    if mode == 'ec':
        return a / (((b - c) / (value - c) - 1) ** (1 / d))
    elif mode == 'ce':
        return d + (c - d) / (1 + (a / value) ** b)
    else:
        raise ValueError("Invalid mode. Use 'ec' or 'ce'.")


def Weibull(value, a, b, mode):
    if mode == 'ec':
        return 10 ** ((np.log(-np.log(1 - value)) - a) / b)
    elif mode == 'ce':
        return 1 - np.exp(-np.exp(a + b * np.log10(value)))
    else:
        raise ValueError("Invalid mode. Use 'ec' or 'ce'.")


def Weibull_three(value, a, b, c, mode):
    if mode == 'ec':
        return np.exp(-(-np.log(np.log(c / (c - value))) + a) * np.log(10) / b)
    elif mode == 'ce':
        return c * (1 - np.exp(-np.exp(a + b * np.log10(value))))
    else:
        raise ValueError("Invalid mode. Use 'ec' or 'ce'.")


def Logit(value, a, b, mode):
    if mode == 'ec':
        return 10 ** ((np.log(value / (1 - value)) - a) / b)
    elif mode == 'ce':
        return 1 / (1 + np.exp(-a - b * np.log10(value)))
    else:
        raise ValueError("Invalid mode. Use 'ec' or 'ce'.")


def Logit_three(value, a, b, c, mode):
    if mode == 'ec':
        return np.exp(-np.log(10) * (a + np.log((c - value) / value)) / b)
    elif mode == 'ce':
        return c / (1 + np.exp((-a) - b * np.log10(value)))
    else:
        raise ValueError("Invalid mode. Use 'ec' or 'ce'.")


def Logit_four(value, a, b, c, d, mode):
    if mode == 'ec':
        return np.exp(-np.log(10) * (a + np.log(-((c - value) / (d - value)))) / b)
    elif mode == 'ce':
        return d + (c - d) / (1 + np.exp((-a) - b * np.log10(value)))
    else:
        raise ValueError("Invalid mode. Use 'ec' or 'ce'.")


def BCW(value, a, b, c, mode):
    if mode == 'ec':
        # return np.exp(np.log(-((a * c - b - np.log(-np.log(1 - y)) * c) / b)) / c) / c
        return (c / b * (np.log(-np.log(1 - value)) - a) + 1) ** (1 / c)
    elif mode == 'ce':
        # return 1 - np.exp(-np.exp(a + b * ((x ** c - 1) / c)))
        return 1 - np.exp(-np.exp((b / c) * (value ** c - 1) + a))
    else:
        raise ValueError("Invalid mode. Use 'ec' or 'ce'.")


def BCL(value, a, b, c, mode): #코드 검토 필요
    if mode == 'ec':
        return np.exp(np.log(-((a * c - b + np.log(-(-1 + value) / value) * c) / b)) / c) / c
        #return math.exp(math.log(-(a * c - b + math.log(-(-1 + value) / value) * c) / b) / c)
    elif mode == 'ce':
        return 1 / (1 + np.exp(-a - b * ((value ** c - 1) / c)))
        #return 1 / (1 + math.exp(-a - b * ((value**c - 1) / c)))
    else:
        raise ValueError("Invalid mode. Use 'ec' or 'ce'.")


def GL(value, a, b, c, mode):
    if mode == 'ec':
        return np.exp(-np.log(10) * (a + np.log(np.exp(-np.log(value) / c) - 1)) / b)
    elif mode == 'ce':
        return 1 / (1 + np.exp(-a - b * np.log10(value)) ** c)
    else:
        raise ValueError("Invalid mode. Use 'ec' or 'ce'.")


def Probit(value, a, b, mode):
    if mode == 'ec':
        return 10 ** ((scipy.stats.norm.ppf(value, loc=0, scale=1) - a) / b)
    elif mode == 'ce':
        return scipy.stats.norm.cdf(a + b * np.log10(value), loc=0, scale=1)
    else:
        raise ValueError("Invalid mode. Use 'ec' or 'ce'.")


def BCP(value, a, b, c, mode):
    if mode == 'ec':
        return (c / b * (scipy.stats.norm.ppf(value, loc=0, scale=1) - a) + 1) ** (1 / c)
    elif mode == 'ce':
        return scipy.stats.norm.cdf(a + b * (((value ** c) - 1) / c), loc=0, scale=1)
    else:
        raise ValueError("Invalid mode. Use 'ec' or 'ce'.")


def Sigmoid(value, a, b, c, mode):
    if mode == 'ec':
        return c - b * np.log((a - value) / value)
    elif mode == 'ce':
        return a / (1 + np.exp(-(value - c) / b))
    else:
        raise ValueError("Invalid mode. Use 'ec' or 'ce'.")


def Logistic(value, a, b, c, mode):
    if mode == 'ec':
        return c * ((a - value) / value) ** (1 / b)
    elif mode == 'ce':
        return a / (1 + (value / c) ** b)
    else:
        raise ValueError("Invalid mode. Use 'ec' or 'ce'.")


def Chapman(value, a, b, c, mode):
    if mode == 'ec':
        return -(np.log(1 - (value / a) ** (1 / c)) / b)
    elif mode == 'ce':
        return a * (1 - np.exp(-b * value)) ** c
    else:
        raise ValueError("Invalid mode. Use 'ec' or 'ce'.")


def Gompertz(value, a, b, c, mode):
    if mode == 'ec':
        return c - b * np.log(-np.log(value / a))
    elif mode == 'ce':
        return a * np.exp(-np.exp(-(value - c) / b))
    else:
        raise ValueError("Invalid mode. Use 'ec' or 'ce'.")


def Weibull_drc(value, a, b, mode):
    if mode == 'ec':
        return np.exp((np.log(-np.log(1 - value)) / a) + np.log(b))
    elif mode == 'ce':
        return 1 - np.exp(-np.exp(a * (np.log(value) - np.log(b))))
    else:
        raise ValueError("Invalid mode. Use 'ec' or 'ce'.")

