import math
import os
import random
import time

import seaborn as sns
import numpy as np
from kernel_regression import KernelRegression
from matplotlib import pyplot as plt
import scipy.stats as ss




# def normal_dist(x , mean , sd):
#     prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
#     return prob_density
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.svm import SVR

x = np.linspace(1,50,200)
# print(x)
# mean = np.mean(x)
# sd = np.std(x)
#
# #Apply function to the data.
# pdf = normal_dist(x,mean,sd)
#
# #Plotting the Results
# plt.plot(x, pdf, color = 'red')
# plt.xlabel('Data points')
# plt.ylabel('Probability Density')
# plt.show()
#РЯД С НОРМАЛЬНЫМ РАСПРЕДЕЛЕНИЕМ SIN(x)

# y = []
# rand = []
# for i in x:
#     y.append(math.sin(i))
# mu = 0
# sigma = 0.3
# rand = np.random.normal(0, 0.3, len(x))
# y += rand
# plt.plot(x, y)
# plt.show()
#
# # sns.displot(data=rand)
# count, bins, ignored = plt.hist(rand, 30, density=True)
# plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
#                np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
#          linewidth=2, color='r')
# plt.show()
#
# krg = KernelRegression()
#
# krg.fit(x, y)
#
# X = np.linspace(1,50,50)
#
# Y = krg.predict(X)


np.random.seed(0)


###############################################################################
# Generate sample data
X = np.sort(50 * np.random.rand(1000, 1), axis=0)
y = np.sin(X).ravel()
print(X, y)
print(X.shape, y.shape)

###############################################################################
# # Add noise to targets
# y += 0.5 * (0.5 - np.random.rand(y.size))
#
# ###############################################################################
# # Fit regression models
#
# kr = KernelRegression(kernel="rbf", gamma=np.logspace(-2, 2, 10))
#
# y_kr = kr.fit(X, y).predict(X)


###############################################################################
# Visualize models
# plt.scatter(X, y, c='k', label='data')
# plt.plot(X, y_kr, c='g', label='Kernel Regression')
# plt.xlabel('data')
# plt.ylabel('target')
# plt.legend()
# plt.show()
print('lr3_2')
data_dir = ''
fname = os.path.join(data_dir, 'dataset.csv')
f = open(fname)
data = f.read()
f.close()
lines = data.split('\n')
lines = lines[1:-2500]
x = []
y = []
for e in lines:
    x.append([float(e.split(',')[0])])
    y.append(float(e.split(',')[1]))

kr = KernelRegression(kernel="rbf", gamma=np.logspace(-2, 2, 10))
x = np.array(x)
y = np.array(y)

print(x, y)
print(x.shape, y.shape)

y_kr = kr.fit(x, y).predict(x)

# Visualize models
plt.scatter(x, y, c='k', label='data')
plt.plot(x, y_kr, c='g', label='Kernel Regression')
plt.xlabel('data')
plt.ylabel('target')
plt.legend()
plt.show()



