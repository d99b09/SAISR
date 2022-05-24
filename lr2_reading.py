import os
import numpy as np
from matplotlib import pyplot as plt
import csv
import pandas as pd


data_dir = ''
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]



float_data = np.zeros((len(lines), len(header) - 1))
for i, line, in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

temp = float_data[:, 1] #temp
#plt.plot(range(len(temp)), temp)
plt.plot(range(2880), temp[:2880])

plt.show()

df = pd.DataFrame(temp[:2880])
df.to_csv('dataset.csv')