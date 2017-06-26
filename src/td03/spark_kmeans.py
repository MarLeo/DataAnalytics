import os

from pyspark import SparkContext, SparkConf
from pyspark.mllib.random import RandomRDDs
from math import sqrt

# compute distance between cluster and point
def distance(x, y):
  return sqrt(sum([(a - b)**2 for a, b in zip(x, y)]))

# Keep the closest point
def closestPoint(dist_list):
    cluster = dist_list[0][0]
    min_dist = dist_list[0][1]
    for elem in dist_list:
        if elem[1] < min_dist:
            cluster = elem[0]
            min_dist = elem[1]
    return (cluster, min_dist)

def summ(x, y):
    return [x[i]+y[i] for i in range(len(x))]

def moyenne(x, n):
    return [x[i]/n for i in range(len(x))] 

def customKmeans(mappedData, clusters):
    number_of_steps = 0
    isFinish = False

    # Initialize centroids
    centroids = mappedData.sample(True, 0.02, 1)\
                        .zipWithIndex()\
                        .map(lambda l: (l[1], l[0][1][:-1]))

    print("centroids: ")
    for centroid in centroids.collect():
        print centroid 

    while not isFinish:
        # Assign each point to a cluster
        cartesian = mappedData.cartesian(centroids)
        print("Cartesian product: ")
        for cluster in cartesian.collect():
            print cluster

        assignTocluster = cartesian.map(lambda l: (l[0][0], (l[1][0], distance(l[0][1][:-1], l[1][1]))))
        print("Distance between a centroid and any point: ")
        for d in assignTocluster.collect():
            print d

        dist_list = assignTocluster.groupByKey().mapValues(list)
        #print("dist_list[0][0]: " + dist_list[0][0])
        #print("dist_list[0][1]: " + dist_list[0][1])

        print("List all points for a specific cluster: ")
        for p in dist_list.collect():
            print p

        # Closest point to a specific cluster
        min_dist = dist_list.mapValues(closestPoint)
        print("List all closest point to a specific cluster: ")
        for elem in min_dist.collect():
            print elem

        # contains the datapoint, the id of the closest cluster and the distance of the point to the centroid
        assignment = min_dist.join(mappedData)
        print("The id of the closest cluster and the distance of the point to the centroid: ")
        for point in assignment.collect():
            print point

        # Compute the new centroid to each cluster
        new_clusters = assignment.map(lambda l: (l[1][0][0], l[1][1][:-1]))
        print("New clusters: ")
        for cl in new_clusters.collect():
            print cl

        count = new_clusters.map(lambda l: (l[0], 1)).reduceByKey(lambda i,j: i+j)
        somme = new_clusters.reduceByKey(summ)
        new_centroids = somme.join(count).map(lambda l: (l[0], moyenne(l[1][0], l[1][1])))

        print("New centroids are : ")
        for c in new_centroids.collect():
            print c

        #intra cluster distance
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
            centroides = new_centroids
            prev_assignment = min_dist
            number_of_steps += 1

    return (assignment, error, number_of_steps)
      

if __name__ == "__main__":

    # Create Spark conf
    conf = SparkConf().setAppName("kmeans").setMaster("local")
    sc = SparkContext(conf=conf)

    # Load data into the rdd
    data = sc.textFile("data/iris_clustering.dat")
    mappedData = data.map(lambda l: l.split(","))\
                    .filter(lambda l: len(l) == 5)\
                    .map(lambda l: [float(i) for i in l[:4]] + [str(l[4])])\
                    .zipWithIndex()\
                    .map(lambda l: (l[1], l[0]))          
                                
    for x in mappedData.collect():
        print x

    clusters = 3
    clustering = customKmeans(mappedData, clusters)

    print(clustering)






