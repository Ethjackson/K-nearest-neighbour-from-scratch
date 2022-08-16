import numpy as np
import numpy.random
import numpy.matlib
import pandas as pd
import matplotlib.pyplot as plt
# set means and standard deviation for 3 datasets along with 300 results in each 
mean1 = -5
std1 = 2

mean2 = 0
std2 = 2.5

mean3 = 5
std3 = 3.0

samples = 300

# create 3 different datasets using numpy.random.normal which 'draws random samples from a normal 
# (Gaussian) distribution.' with 2 rows in all datasets. 
dataset1 = numpy.random.normal(mean1, std1, (2, samples))
dataset2 = numpy.random.normal(mean2, std2, (2, samples))
dataset3 = numpy.random.normal(mean3, std3, (2, samples))

# concatenate the datasets together to make a main one with axis=1 to concatenate alomg columns
dataset = np.concatenate((dataset1, dataset2, dataset3), axis=1)

# converts all the datasets into pandas dataframes
dataset1 = pd.DataFrame(dataset1)
dataset2 = pd.DataFrame(dataset2)
dataset3 = pd.DataFrame(dataset3)
dataset = pd.DataFrame(dataset)

# plots all the datasets on a graph to show them inividually, then show the concatenated dataset
plt.scatter(dataset1.iloc[0], dataset1.iloc[1])
plt.scatter(dataset2.iloc[0], dataset2.iloc[1])
plt.scatter(dataset3.iloc[0], dataset3.iloc[1])
plt.show()
plt.scatter(dataset.iloc[0], dataset.iloc[1])
plt.show()
# print(dataset)

# r and c are the amount of rows and columns in the main dataset using np.shape for use in for loops
# max_iterations is the number of times to loop reasign the centroids and label which nodes they belong to improve accuracy but lower performance
# n is the number of clusters that will be created to be assigned nodes to, and will loop from 1 cluster to n to see what amount is most
# efficient, n = 4 will result in 3 clusters
r,c = np.shape(dataset)
max_iterations = 5
n = 5

# kbest is used to store the final distances to see which iteration was best
kbest = []
# the for k loop will iterate through n times, this will run to record and unerstand what is the best cluster amount 'n'
for k in range(0,n):
    # distances variable used to append into kbest for total distance of a clusters iteration to find lowest WSS cluster amount 
    distances = 0

    # centroid permutes the dataset to randomise the order and then gets the first n entries for the amount of centroids to be set 
    centroid = pd.DataFrame(np.random.permutation(dataset))
    # sets centroid to have k amount of columns/centroids
    if k == 0:
        centroid = pd.DataFrame(centroid.iloc[:,0])
    else:
        centroid = centroid.iloc[:,:k]

    # this is a list of all the distances for each iteration of a cluster
    totaldistsum = []
    # l loop runs max_iterations times to test multiple different outcomes 
    for l in range(0,max_iterations):
        # assignednodearr to assign each node to its closest centroid
        assignednodearr = pd.DataFrame(np.zeros((1, len(dataset))))
        # i loop will run to assign nodes to their closest centroid
        for i in range(0, c):
            # smallestdist calculates euclidian distance between i column and first centroid
            smallestdist = np.sqrt(np.sum((dataset.iloc[:,i] - centroid.iloc[:,0]) ** 2))
            # is the to be centroid that a node gets assigned to
            y = 0

            # for loop runs to check which centroid node is closest to by comparing k centroids to i column
            for p in range(0,k):
                # dist calculates euclidian distance between i column and all centroids to see which centroid node is closest to
                dist = np.sqrt(np.sum((dataset.iloc[:,i] - centroid.iloc[:,p]) ** 2))

                # if the distance for that node to centroid is less then previous centroid set smallestdist to that dist 
                # and assign node to that centroid via y = p
                if dist < smallestdist:
                    smallestdist = dist
                    y = p
                
                # assign that node in assigned node array for later use although may be ovewritten if new smallestdist found
                assignednodearr[i] = y
            
        # distance = 0 to wipe previous values to append to totaldistsum  
        distances = 0
        # j loop gets all nodes assigned to certain centroid and calculates total distance of nodes to centroid
        for j in range(0,k):
            # used to get all nodes assigned to certain centroid
            clusterselected = []

            # i loop goes through assignednodearr and if that assigned node num = centroid number it will append that node from the dataset 
            # into clusterselected 
            for i in range(0,c):
                if j == int(assignednodearr.iloc[:,i]):
                    clusterselected.append(dataset.iloc[:,i])
            
            # finds the mean of the cluster nodes to set the centroid into the center of it
            centroid.iloc[:,j] = np.mean(clusterselected)
            
            # calculates the total distance of all nodes in cluster to the centroid
            totdistfromcent = (clusterselected - np.mean(clusterselected)) ** 2

            # sums the total distance from centroid into number and += distances to distances 
            # this will result in the total distance of all nodes in a cluster to their respective centroids for k centroid amount
            # aka the total distance of nodes in cluster 1 to centroid 1 + total distance of nodes in cluster 2 to centroid 2 to k repeating amount
            distances = distances + np.sum(np.sqrt(totdistfromcent[0,:] + totdistfromcent[1,:]))
                
        totaldistsum.append(distances)
    # kbest appends distances to allow plotting on which k has the lowest WSS for best amount of clusters
    kbest.append(distances)

    # does not run on k=0 to avoid empty scatter plot on first run as k=0 means no clusters
    if k != 0:
        # j loop in range of k loops through k times 
        for j in range(0,k):
            # this is a dataframe to store nodes that are clustered to a centroid for k times
            clusterselectedplot = []

            # i loop goes through assignednodearr and if that assigned node num = centroid number it will append that node from the dataset 
            # into clusterselected... same as previous
            for i in range(0,c):
                if j == int(assignednodearr.iloc[:,i]):
                    clusterselectedplot.append(dataset.iloc[:,i])

            # turns clusterselectedplot into dataframe an transposes data 
            clusterselectedplot = pd.DataFrame(clusterselectedplot).T
           
            # the selected cluster is then scatter plotted 
            # this will repeat k times for each cluster to add them seperatly to a plot
            plt.scatter(clusterselectedplot.iloc[0], clusterselectedplot.iloc[1])

        # finally the centroids for all the clusters are added to make more clear an able to see problems
        plt.scatter(centroid.iloc[0], centroid.iloc[1], marker='x', linewidths=4, s=200)
        plt.show()

# this plots all the cluster amounts and shows each clusters WSS
# WSS is the total distance of the nodes from centroids centroids
plt.plot(kbest)
plt.title('K best Distances')
plt.xlabel('K')
plt.ylabel('WSS')
plt.show()