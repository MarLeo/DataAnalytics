import os
import sys

from pyspark import SparkContext, SparkConf
from pyspark.mllib.random import RandomRDDs
from math import sqrt

# compute distance between cluster and point
def distance(x, y):
  return sqrt(sum([(a - b)**2 for a, b in zip(x, y)]))

# Keep the closest point
def closestPoint(dist_list):
    cluster = dist_list[0][0]
    #print("first distance: " + str(cluster))
    min_dist = dist_list[0][1]
    #print("min distance: " + str(min_dist))
    for elem in dist_list:
        if elem[1] < min_dist:
            cluster = elem[0]
            min_dist = elem[1]
    return (cluster, min_dist)


def avg(x, y):
    return x / y


def summ(x, y):
    return [x[i]+y[i] for i in range(len(x))]

def moyenne(x, n):
    return [x[i]/n for i in range(len(x))] 


def loadData(sc, path):
    data = sc.textFile(path) #"data/iris_clustering.dat"
    mappedData = data.map(lambda l: l.split(","))\
                    .filter(lambda l: len(l) == 5)\
                    .map(lambda l: [float(i) for i in l[:4]] + [str(l[4])])\
                    .zipWithIndex()\
                    .map(lambda l: (l[1], l[0]))        
    return mappedData

def initCentroids(sc, mappedData):
    centroids = sc.parallelize(mappedData.takeSample(False, clusters)).zipWithIndex().map(lambda l: (l[1], l[0][1][:-1]))
    
    return centroids

def assignTocluster(mappedData, centroids):
    cartesian = mappedData.cartesian(centroids)
    assignTocluster = cartesian.map(lambda l: (l[0][0], (l[1][0], distance(l[0][1][:-1], l[1][1]))))\
                               .groupByKey().mapValues(list)
    close_distance = assignTocluster.mapValues(closestPoint)
    return close_distance


def computeCentroids(close_distance, mappedData):
    close_points = close_distance.join(mappedData)
    new_clusters = close_points.map(lambda l: (l[1][0][0], l[1][1][:-1]))
    number_of_points_by_cluster = new_clusters.map(lambda l: (l[0], 1)).reduceByKey(lambda i,j: i+j)
    total_distance_by_cluster = new_clusters.reduceByKey(summ)
    new_centroids = total_distance_by_cluster.join(number_of_points_by_cluster).map(lambda l: (l[0], moyenne(l[1][0], l[1][1])))
    return new_centroids

def computeIntraClusterDistance(close_distance):
      count_points_dist = close_distance.map(lambda l: l[1]).map(lambda l: (l[0], 1)).reduceByKey(lambda x, y: x + y)
      points_distance =  close_distance.map(lambda l: l[1]).reduceByKey(lambda x, y: x + y)
      avg_distance = points_distance.join(count_points_dist).map(lambda l: (l[0], avg(l[1][0], l[1][1])))
      intra_cluster_distance = avg_distance.map(lambda l: l[1]).sum()
      return intra_cluster_distance



def kmeans(sc, path, clusters, max_iterations, hasConverge=False, moved=0):

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
                moved = prev_assignment.join(close_distance)\
                                        .filter(lambda l: l[1][0][0] != l[1][1][0])\
                                        .count()
            else:
                moved = 150
            if moved == 0 or iterations == 100:
                hasConverge = True
            else:
                centroids = new_centroids
                prev_assignment = close_distance
                iterations += 1

    return (intra_cluster_distance, iterations)
    



def customKmeans(mappedData, clusters, sc):
    number_of_steps = 0
    isFinish = False

    # Initialize centroids sample(True, 0.02, 1)\
    centroids = sc.parallelize(mappedData.takeSample(False, clusters)).zipWithIndex().map(lambda l: (l[1], l[0][1][:-1]))

                       
    print("centroids: ")
    for centroid in centroids.collect():
        print centroid 
    
    while not isFinish:
        # Assign each point to a cluster
        cartesian = mappedData.cartesian(centroids)
        '''
        print("Cartesian product: ")
        for cluster in cartesian.collect():
            print cluster
        '''
        assignTocluster = cartesian.map(lambda l: (l[0][0], (l[1][0], distance(l[0][1][:-1], l[1][1]))))
        '''
        print("Distance between a centroid and any point: ")
        for d in assignTocluster.collect():
            print d
        '''    
        dist_list = assignTocluster.groupByKey().mapValues(list)
        '''
        print("List all points for a specific cluster: ")
        for p in dist_list.collect():
            print p
        '''
        # Closest point to a specific centroid
        min_dist = dist_list.mapValues(closestPoint)
        '''
        print("List all closest point to a specific cluster: ")
        for elem in min_dist.collect():
            print elem
        '''
        # contains the datapoint, the id of the closest cluster and the distance of the point to the centroid
        assignment = min_dist.join(mappedData)
        '''
        print("The id of the closest cluster and the distance of the point to the centroid: ")
        for point in assignment.collect():
            print point
        '''
        # Compute the new centroid to each cluster
        new_clusters = assignment.map(lambda l: (l[1][0][0], l[1][1][:-1]))
        '''
        print("New clusters: ")
        for cl in new_clusters.collect():
            print cl
        '''
        # Number of points for each cluster
        count = new_clusters.map(lambda l: (l[0], 1)).reduceByKey(lambda i,j: i+j)
        '''
        print("count: ")
        for c in count.collect():
            print c
        print("somme: ")
        '''
        somme = new_clusters.reduceByKey(summ)
        '''
        for s in somme.collect():
            print s
        '''
        new_centroids = somme.join(count).map(lambda l: (l[0], moyenne(l[1][0], l[1][1])))
        '''
        print("New centroids are : ")
        for c in new_centroids.collect():
            print c
        '''
        #intra cluster distance
        count_points_dist = min_dist.map(lambda l: l[1]).map(lambda l: (l[0], 1)).reduceByKey(lambda x, y: x + y)
        #intra_cluster_distance = min_dist.join(new_centroids).map(lambda l: (l[0], distance(l[1][0], l[1][1])))      
        '''
        print("points in a cluster: ")
        for intra in count_points_dist.collect():
            print intra
        '''
        points_distance =  min_dist.map(lambda l: l[1]).reduceByKey(lambda x, y: x + y)
        '''
        print("total in a cluster: ")
        for intra in points_distance.collect():
            print intra  
        '''
        avg_distance = points_distance.join(count_points_dist).map(lambda l: (l[0], avg(l[1][0], l[1][1])))
        print("avg distance in a cluster: ")
        for intra in avg_distance.collect():
            print intra   
        
        intra_cluster_distance = avg_distance.map(lambda l: l[1]).sum()
        print intra_cluster_distance



        if number_of_steps > 0:
            switch = prev_assignment.join(min_dist)\
                                    .filter(lambda l: l[1][0][0] != l[1][1][0])\
                                    .count()
        else:
            switch = 150
        if switch == 0 or number_of_steps == 100:
            isFinish = True
            error = sqrt(min_dist.map(lambda l: l[1][1]).reduce(lambda x, y: x + y))/mappedData.count()
        else:
            centroids = new_centroids
            prev_assignment = min_dist
            number_of_steps += 1

    return (switch, error, number_of_steps)
      

if __name__ == "__main__":
    

    if len(sys.argv) != 4:
        print("Usage: kmeans.py <file> <k> <m>")
        exit(-1)

    # Create Spark conf
    conf = SparkConf().setAppName("kmeans").setMaster("local")
    sc = SparkContext(conf=conf)

    path = sys.argv[1]
    clusters = int(sys.argv[2])
    max_iterations = int(sys.argv[3])
    total = 0
    for i in range(0, 100):
        print("Iteration number: " + str(i))
        solution = kmeans(sc, path, clusters, max_iterations)
        total += solution[0]/100
        print(solution)

    print("average intra cluster distance: " + str(total))   

    '''
    mappedData = loadData(sc)
    clusters = 4
    clustering = customKmeans(mappedData, clusters, sc)

    print(clustering)
    '''




