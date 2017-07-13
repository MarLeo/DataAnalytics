import os
import sys
import time
import numpy as np

from pyspark import SparkContext, SparkConf
from pyspark.mllib.random import RandomRDDs
from math import sqrt

# compute distance between cluster and point
def computeDistance(x, y):
  return sqrt(sum([(a - b)**2 for a, b in zip(x, y)]))

# Keep the closest point
def computeClosestPoint(dist_list):
    cluster = dist_list[0][0]
    min_dist = dist_list[0][1]
    for elem in dist_list:
        if elem[1] < min_dist:
            cluster = elem[0]
            min_dist = elem[1]
    return (cluster, min_dist)


def avg(x, y):
    return x / y


def computeSum(x, y):
    return [x[i]+y[i] for i in range(len(x))]

def computeMean(x, n):
    return [x[i]/n for i in range(len(x))]


def loadData(sc, path):
    data = sc.textFile(path) #"data/iris_clustering.dat"
    mappedData = data.map(lambda l: l.split(","))\
                    .filter(lambda l: len(l) == 5)\
                    .map(lambda l: [float(l[0]), float(l[1]), float(l[2]), float(l[3]), str(l[4])])\
                    .zipWithIndex()\
                    .map(lambda l: (l[1], l[0]))
    return mappedData

def initCentroids(sc, mappedData):
    centroids = sc.parallelize(mappedData.takeSample(False, clusters)).zipWithIndex().map(lambda l: (l[1], l[0][1][:-1]))
    return centroids

# assign to each cluster the closest point
def assignTocluster(mappedData, centroids):
    cartesian = mappedData.cartesian(centroids)
    assignTocluster = cartesian.map(lambda l: (l[0][0], (l[1][0], computeDistance(l[0][1][:-1], l[1][1]))))\
                               .groupByKey().mapValues(list)
    close_distance = assignTocluster.mapValues(computeClosestPoint)
    return close_distance

# update the centroids
def computeCentroids(close_distance, mappedData):
    close_points = close_distance.join(mappedData)
    new_clusters = close_points.map(lambda l: (l[1][0][0], l[1][1][:-1]))
    number_of_points_by_cluster = new_clusters.map(lambda l: (l[0], 1)).reduceByKey(lambda i,j: i+j)
    total_distance_by_cluster = new_clusters.reduceByKey(computeSum)
    new_centroids = total_distance_by_cluster.join(number_of_points_by_cluster).map(lambda l: (l[0], computeMean(l[1][0], l[1][1])))
    return new_centroids

def computeIntraClusterDistance(close_distance):
      count_points_dist = close_distance.map(lambda l: l[1]).map(lambda l: (l[0], 1)).reduceByKey(lambda x, y: x + y)
      points_distance =  close_distance.map(lambda l: l[1]).reduceByKey(lambda x, y: x + y)
      avg_distance = points_distance.join(count_points_dist).map(lambda l: (l[0], avg(l[1][0], l[1][1])))
      intra_cluster_distance = avg_distance.map(lambda l: l[1]).sum()
      return intra_cluster_distance



def kmeans(sc, path, clusters, max_iterations, hasConverge=False, moved=200):
    iterations = 0

    mappedData = loadData(sc, path)

     # Initialize centroids sample(True, 0.02, 1)\
    centroids = sc.parallelize(mappedData.takeSample(False, clusters)).zipWithIndex().map(lambda l: (l[1], l[0][1][:-1]))
    print("centroids: ")
    for centroid in centroids.collect():
        print centroid

    while not hasConverge and iterations <= max_iterations:
            close_distance = assignTocluster(mappedData, centroids)
            new_centroids = computeCentroids(close_distance, mappedData)
            intra_cluster_distance = computeIntraClusterDistance(close_distance)
            print("Intra cluster distance: " + str(intra_cluster_distance))

            if iterations > 0:
                moved = new_points.join(close_distance)\
                                .filter(lambda l: l[1][0][0] != l[1][1][0])\
                                .count()

            if moved == 0 or iterations == max_iterations:
                hasConverge = True
            else:
                centroids = new_centroids
                new_points = close_distance
                iterations += 1

    return (intra_cluster_distance, iterations)


if __name__ == "__main__":

    start_time = time.time()

    if len(sys.argv) != 4:
        print("Usage: kmeans.py <file> <k> <m>")
        exit(-1)

    # Create Spark conf
    conf = SparkConf().setAppName("kmeans").setMaster("local")
    sc = SparkContext(conf=conf)

    path = sys.argv[1]
    clusters = int(sys.argv[2])
    max_iterations = int(sys.argv[3])
    distances = []
    for i in range(0, 5):
        print("Iteration number: " + str(i))
        solution = kmeans(sc, path, clusters, max_iterations)
        distances.append(solution[0])
        print(solution)

    print("average intra cluster distance: " + str(np.mean(distances)))
    print("standard deviation: " + str(np.std(distances)))

    print("---- %s seconds ----" % (time.time() - start_time))
