import os
import scipy.interpolate as spi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import argrelextrema


def sifting(data):
    index = list(range(len(data)))

    max_peaks = list(argrelextrema(data, np.greater)[0])
    min_peaks = list(argrelextrema(data, np.less)[0])

    ipo3_max = spi.splrep(max_peaks, data[max_peaks], k=3)
    iy3_max = spi.splev(index, ipo3_max)

    ipo3_min = spi.splrep(min_peaks, data[min_peaks], k=3)
    iy3_min = spi.splev(index, ipo3_min)

    iy3_mean = (iy3_max + iy3_min) / 2
    return data - iy3_mean


def hasPeaks(data):
    max_peaks = list(argrelextrema(data, np.greater)[0])
    min_peaks = list(argrelextrema(data, np.less)[0])

    if len(max_peaks) > 3 and len(min_peaks) > 3:
        return True
    else:
        return False


def isIMFs(data):
    max_peaks = list(argrelextrema(data, np.greater)[0])
    min_peaks = list(argrelextrema(data, np.less)[0])

    if min(data[max_peaks]) < 0 or max(data[min_peaks]) > 0:
        return False
    else:
        return True


def getIMFs(data):
    while (not isIMFs(data)):
        data = sifting(data)
    return data


def EMD(data):
    IMFs = []
    while hasPeaks(data):
        data_imf = getIMFs(data)
        data = data - data_imf
        IMFs.append(data_imf)
    return IMFs

def main():
    data_dir = ''
    fname = os.path.join(data_dir, 'dataset.csv')

    f = open(fname)
    data = f.read()
    f.close()

    lines = data.split('\n')
    lines = lines[1:-1]

    x = []
    y = []

    for e in lines:
        x.append(float(e.split(',')[0]))
        y.append(float(e.split(',')[1]))
    #plt.plot(x, y)

    #y = np.array(y)

    data = np.array(y)
    data = data / max(data)
    index = list(range(len(data)))
    plt.plot(data)

    # get extrema: min and max
    max_peaks = list(argrelextrema(data, np.greater)[0])
    min_peaks = list(argrelextrema(data, np.less)[0])

    # Установите крайнюю точку на кривую
    ipo3_max = spi.splrep(max_peaks, data[max_peaks], k=3)  # Импорт точек образца, генерируют параметры
    iy3_max = spi.splev(index, ipo3_max)  # генерирует интерполяцию в соответствии с наблюдениями и параметрами сплайн

    ipo3_min = spi.splrep(min_peaks, data[min_peaks], k=3)  # Импорт точек образца, генерируют параметры
    iy3_min = spi.splev(index, ipo3_min)  # генерирует интерполяцию в соответствии с наблюдениями и параметрами сплайн

    # Рассчитать средний конверт
    iy3_mean = (iy3_max + iy3_min) / 2

    plt.plot(iy3_max)
    plt.plot(iy3_min)
    plt.plot(iy3_mean)
    plt.xlim(100, 200)
    plt.ylim(-4, -6)
    plt.show()

    plt.plot(iy3_mean)
    plt.plot(data)
    plt.xlim(100, 200)
    plt.show()


    IMFs = EMD(data)
    n = len(IMFs) + 1

    plt.figure(figsize=(18, 15))
    plt.subplot(n, 1, 1)
    plt.plot(data, label='Origin')
    plt.title("Origin ")

    for i in range(0, len(IMFs)):
        plt.subplot(n, 1, i + 2)
        plt.plot(IMFs[i])
        #plt.ylabel('Amplitude')
        #plt.title("IMFs " + str(i + 1))

    #plt.legend()
    plt.xlabel('time (s)')
    plt.ylabel('Amplitude')
    plt.show()
    a = IMFs[0] + IMFs[1] + IMFs[2] + IMFs[3] + IMFs[4] + IMFs[5]
    plt.plot(data)
    plt.plot(a)
    plt.show()












if __name__ == '__main__':
    main()
#plt.plot(x, y)
#
#plt.show()
#print(lines)
#print(len(lines))

