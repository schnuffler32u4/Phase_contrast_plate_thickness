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


thick = []
thickerr = []

for file in os.listdir('data/one_plate'):
    if file[:4] != "data":
        data = pd.read_csv('data/one_plate/' + file)
    data.rename(columns={"Angle (rad) Run #1": "Angle", "Light Intensity, Ch 1 (lx) Run #1": "Light"}, inplace=True)
    data.dropna(inplace=True)
    angle = np.array(data.Angle)
    angle[0] = 0
    nans = []
    for i in range(len(angle)):
        if np.isnan(angle[i]) == False:
            nans.append(i)

    for j in range(len(nans) - 1):
        angle[nans[j]:nans[j+1]] = np.linspace(angle[nans[j]], angle[nans[j+1]], len(angle[nans[j]:nans[j+1]]))
    # print(angle)
    popt, pcov = curve_fit(fitfunction, angle, data.Light, maxfev=50000, p0=[1e-3, 0, 48])

    plt.plot(angle, data.Light, color='red')
    cal = np.zeros(len(angle))

    plt.plot(angle, fitfunction(angle, *popt), color='cyan')
    plt.show()
    # print(popt[0])
    thick.append(popt[0])
    thickerr.append(np.sqrt(pcov[0][0]))

thick = np.array(thick)
thickerr = np.array(thickerr)
print(np.sum(thick / thickerr ** 2) / np.sum(1 / thickerr ** 2))
print(np.sqrt(1 / np.sum(1 / thickerr ** 2)))