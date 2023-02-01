import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import os
import statistics

plt.style.use('extensys')

wl = 650e-9  # the wavelength of the lase in m
# n = 1.51  # index of refraction of the glass
thickness = 1e-3  # the thickness of one of the glass plates


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


def fitfunction(theta, n, path, bg, amp):
    """Returns the light intensity for a given angle and plate thickness"""
    theta = theta * (2.9 / 42) ** (-1)
    return bg + amp * np.sin(((1 - n) * thickness / np.cos(theta) + path) / wl)


def fitfunction2(theta, n, path, bg, amp):
    """Returns the light intensity for a given angle and plate thickness"""
    theta = theta * 2.9 / 42
    return bg + amp * np.sin(((1 - n) * 2 * thickness / np.cos(theta) + path) / wl)


ns = []

for file in os.listdir('data/one_plate'):
    # if file[:4] != "data": # this can be used if one wishes to only work with a portion of the data
    if True:
        data = pd.read_csv('data/one_plate/' + file)
    # print(file)
    data.rename(columns={"Angle (rad) Run #1": "Angle", "Light Intensity, Ch 1 (lx) Run #1": "Light"}, inplace=True)
    data.dropna(inplace=True)
    data.drop(data[data.Light < np.average(data.Light) - 1.3 * (np.amax(data.Light) - np.average(data.Light))].index,
              inplace=True)
    data.drop(data[data.Light > np.average(data.Light) + 1.3 * (-np.amin(data.Light) + np.average(data.Light))].index,
              inplace=True)
    angle = np.array(data.Angle)
    angle[0] = 0
    nans = []

    for i in range(len(angle)):
        if np.isnan(angle[i]) == False:
            nans.append(i)

    for j in range(len(nans) - 1):
        angle[nans[j]:nans[j + 1]] = np.linspace(angle[nans[j]], angle[nans[j + 1]], len(angle[nans[j]:nans[j + 1]]))
    # print(angle)
    popt, pcov = curve_fit(fitfunction, angle, data.Light, maxfev=500000,
                           p0=[1.5, 0, np.average(data.Light), np.amax(data.Light) - np.average(data.Light)])
    cal = np.zeros(len(angle))
    peaks = np.zeros(len(angle), dtype=int)
    light = np.array(data.Light)
    datanew = pd.DataFrame({"Angle": angle, "Light": light})
    datanew.sort_values(by=['Angle'], ascending=True, inplace=True)
    angle = np.array(datanew.Angle)
    light = np.array(datanew.Light)

    fig, ax = plt.subplots()
    m = 0
    for j in range(1, len(light) - 1):
        if light[j - 1] < light[j] > light[j + 1] and np.amin(light[peaks[m - 1]:j]) < np.average(light) and sum(
                [1 for x in light[peaks[m - 1]:j] if x < np.average(light)]) / len(light[peaks[m - 1]:j]) > 0.3 and light[j] > np.average(light) + 0.1 * (
                np.amax(light) - np.average(light)) and sum([1 for x in light[peaks[m - 1]:j]]) > 3:
            peaks[m] = j
            circle1 = plt.Circle((angle[j], light[j]), 0.05, color='blue')
            ax.add_patch(circle1)
            m = m + 1

    circle1 = plt.Circle((angle[peaks[m]], light[m]), 0.05, color='blue', label='Peak detection')
    ax.add_patch(circle1)
    # ax.plot(angle, fitfunction(angle, *popt), color='cyan')
    ax.plot(angle, light, color='red', label='Data')
    plt.legend()
    plt.title("Light intensity versus angle with the peak detection algorithm")
    plt.xlabel("Angle [rad]")
    plt.ylabel("Light intensity [lux]")
    # plt.show()
    # print(popt[0])
    thet = np.amax(np.abs([np.amax(angle), np.amin(angle)]) * 2.9 / 42)
    fringes = sum([1 for x in peaks if x != 0])
    alph = fringes * wl / (2 * thickness)
    nny = (alph ** 2 + 2 * (1 - np.cos(thet)) * (1 - alph)) / (2 * (1 - np.cos(thet) - alph))
    ns.append(nny)

print(np.average(ns))
print(np.std(ns))
np.savetxt("One_plate.txt", ns, fmt="%.2f")

# We now move on to work on the second type of data collected by the experiment, mainly the one

ns = []
for file in os.listdir('data/two_plates'):
    # if file[:4] != "data": # this can be used if one wishes to only work with a portion of the data
    if True:
        data = pd.read_csv('data/two_plates/' + file)
    # print(file)
    data.rename(columns={"Angle (rad) Run #1": "Angle", "Light Intensity, Ch 1 (lx) Run #1": "Light"}, inplace=True)
    data.dropna(inplace=True)
    data.drop(data[data.Light < np.average(data.Light) - 1.3 * (np.amax(data.Light) - np.average(data.Light))].index,
              inplace=True)
    data.drop(data[data.Light > np.average(data.Light) + 1.3 * (-np.amin(data.Light) + np.average(data.Light))].index,
              inplace=True)
    angle = np.array(data.Angle)
    angle[0] = 0
    nans = []

    for i in range(len(angle)):
        if np.isnan(angle[i]) == False:
            nans.append(i)

    for j in range(len(nans) - 1):
        angle[nans[j]:nans[j + 1]] = np.linspace(angle[nans[j]], angle[nans[j + 1]], len(angle[nans[j]:nans[j + 1]]))
    # print(angle)
    popt, pcov = curve_fit(fitfunction, angle, data.Light, maxfev=500000,
                           p0=[1.5, 0, np.average(data.Light), np.amax(data.Light) - np.average(data.Light)])
    cal = np.zeros(len(angle))
    peaks = np.zeros(len(angle), dtype=int)
    light = np.array(data.Light)
    datanew = pd.DataFrame({"Angle": angle, "Light": light})
    datanew.sort_values(by=['Angle'], ascending=True, inplace=True)
    angle = np.array(datanew.Angle)
    light = np.array(datanew.Light)

    fig, ax = plt.subplots()
    m = 0
    for j in range(1, len(light) - 1):
        if light[j - 1] < light[j] > light[j + 1] and np.amin(light[peaks[m - 1]:j]) < np.average(light) and sum(
                [1 for x in light[peaks[m - 1]:j] if x < np.average(light)]) / len(light[peaks[m - 1]:j]) > 0.3 and light[j] > np.average(light) + 0.1 * (
                np.amax(light) - np.average(light)) and sum([1 for x in light[peaks[m - 1]:j]]) > 3:
            peaks[m] = j
            circle1 = plt.Circle((angle[j], light[j]), 0.05, color='blue')
            ax.add_patch(circle1)
            m = m + 1

    circle1 = plt.Circle((angle[peaks[m]], light[m]), 0.05, color='blue', label='Peak detection')
    ax.add_patch(circle1)
    # ax.plot(angle, fitfunction(angle, *popt), color='cyan')
    ax.plot(angle, light, color='red', label='Data')
    plt.legend()
    plt.title("Light intensity versus angle with the peak detection algorithm")
    plt.xlabel("Angle [rad]")
    plt.ylabel("Light intensity [lux]")
    # p.show()
    # print(popt[0])
    thet = np.amax(np.abs([np.amax(angle), np.amin(angle)]) * 2.9 / 42)
    fringes = sum([1 for x in peaks if x != 0])
    alph = fringes * wl / (4 * thickness)
    nny = (alph ** 2 + 2 * (1 - np.cos(thet)) * (1 - alph)) / (2 * (1 - np.cos(thet) - alph))
    ns.append(nny)
np.savetxt("Two_plates.txt", ns, fmt="%.2f")
print(np.average(ns))
print(np.std(ns))
