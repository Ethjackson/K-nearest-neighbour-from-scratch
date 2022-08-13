from cgitb import small
from random import random
from tkinter import END
import numpy as np
import numpy.random
import numpy.matlib
import pandas as pd
import matplotlib.pyplot as plt

max_iterations = 10

mean1 = -4
std1 = 1.9

mean2 = -1
std2 = 2.0

samples = 250
import random as rand
dataset1 = numpy.random.normal(mean1, std1, (2, samples))
dataset2 = numpy.random.normal(mean2, std2, (2, samples))
dataset = np.concatenate((dataset1, dataset2), axis=1)

dataset1 = pd.DataFrame(dataset1)
dataset2 = pd.DataFrame(dataset2)
dataset = pd.DataFrame(dataset)

r,c = np.shape(dataset)

plt.scatter(dataset1.iloc[0], dataset1.iloc[1])
plt.scatter(dataset2.iloc[0], dataset2.iloc[1])
plt.show()

plt.scatter(dataset.iloc[0], dataset.iloc[1])
plt.show()

print(dataset)

kbest = []
for k in range(0,5):
    distances = 0
    # def knearest(dataset):
    centroid = pd.DataFrame(np.random.permutation(dataset))
    totaldistsum = []
    for l in range(0,5):
        
        assignednodearr = pd.DataFrame(np.zeros((1, len(dataset))))

        for i in range(0, c):
            smallestdist = np.sum((dataset.iloc[:,i] - centroid.iloc[:,1]) ** 2)
            print(smallestdist)

            y = 1

            for p in range(0,2):
                dist = np.sum((dataset.iloc[:,i] - centroid.iloc[:,p]) ** 2)
                print(dist) 

                if dist < smallestdist:
                    smallestdist = dist
                    y = p
                
                assignednodearr[i] = y
            
        print(assignednodearr)
        distances = 0
        for j in range(0,2):
            clusterselected = []

            for i in range(0,c):
                if j == int(assignednodearr.iloc[:,i]):
                    clusterselected.append(dataset.iloc[:,i])
            
            print(np.shape(clusterselected))
            
            centroid.iloc[:,j] = np.mean(clusterselected)
            
            totdistfromcent = (clusterselected - np.mean(clusterselected)) ** 2

            distances = distances + np.sum(np.sqrt(totdistfromcent[0,:] + totdistfromcent[1,:]))
                
        totaldistsum.append(distances)
        print(totaldistsum)
    kbest.append(distances)
print(kbest)

# plt.plot(dataset[1, assignednodearr == i], dataset[2, assignednodearr == i])
# plt.plot(centroids(1,:), centroids(2,:), 'x', LineWidth= 3, MarkerSize= 15)


for j in range(0,2):
    clusterselectedplot = []
    for i in range(0,c):
        if j == int(assignednodearr.iloc[:,i]):
            clusterselectedplot.append(dataset.iloc[:,i])

    clusterselectedplot = pd.DataFrame(clusterselectedplot).T
    print(clusterselectedplot)
    plt.scatter(clusterselectedplot.iloc[0], clusterselectedplot.iloc[1])
plt.show()
