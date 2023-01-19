import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import os

plt.style.use('extensys')

wl = 650e-9  # the wavelength of the lase in m
n = 1.51  # index of refraction of the glass


def roundup(x):
    """Returns the input value rounded up to one significant figure."""
    if int(np.log10(x)) == np.log10(x):
        y = np.log10(x)
    elif np.log10(x) < 0:
        y = int(np.log10(x)) - 1
    else:
        y = int(np.log10(x))

    if int(x * (10 ** (-y))) * 10 ** y != x:
        return int(x * (10 ** (-y)) + 1) * 10 ** y
    else:
        return x


def fitfunction(theta, thickness, path, bg):
    """Returns the light intensity for a given angle and plate thickness"""
    theta = theta * 2.9 / 42
    return bg + np.sin(((1 - n) * thickness / np.cos(theta) + path) / wl)


for file in os.listdir('data/one_plate'):

    data = pd.read_csv('data/one_plate/' + file)
    data.rename(columns=)