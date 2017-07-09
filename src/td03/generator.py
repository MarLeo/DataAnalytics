import os
import sys
import random
import csv

from pyspark import SparkConf, SparkContext
from pyspark.mllib.random import RandomRDDs
from pyspark.mllib.linalg import Vectors

from spark_kmeans import kmeans

def cluster_id(sc, partitions):
    return sc.parallelize(range(0, partitions))

def generate_random_vector(sc, path, numRows, numCols, partitions, standard_deviation):
    normalRDD = RandomRDDs.normalVectorRDD(sc, numRows, numCols, partitions, seed=1)\
                           .map(lambda l: 10 + standard_deviation * l)\
                           .map(lambda l: [l[0], l[1], l[2], l[3], random.randint(0, partitions - 1)])
    #normalRDD.saveAsTextFile(path)
    return normalRDD




if __name__ == "__main__":


    if len(sys.argv) != 6:
        print("Usage: generator.py out.csv n k p s")
        exit(-1)


    # Create Spark conf
    conf = SparkConf().setAppName("generator").setMaster("local")
    sc = SparkContext(conf=conf)

    path = sys.argv[1]
    numRows = int(sys.argv[2])
    partitions = int(sys.argv[3])
    numCols = int(sys.argv[4])
    standard_deviation = int(sys.argv[5])

    normalRDD = generate_random_vector(sc, path, numRows, numCols, partitions, standard_deviation)

    print 'Generate RDD of %d examples of length-4 vectors.'%normalRDD.count()
    for sample in normalRDD.collect():
        print str(sample).replace("[", "").replace("]", "")
    print

    with open(path, 'wb') as csvfile:
        for sample in normalRDD.collect():
            csvfile.write(str(sample).replace("[", "").replace("]", ""))
            csvfile.write('\n')
    csvfile.close()

    solution = kmeans(sc, path, partitions, max_iterations=100)
    print(solution)


    sc.stop()
