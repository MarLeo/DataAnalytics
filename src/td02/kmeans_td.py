import os
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SQLContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import VectorAssembler

from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import FloatType
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType

from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import col

from pyspark.ml.clustering import KMeans


# Create spark session
spark = SparkSession.builder.master("local").appName("kmeans").getOrCreate()
clusters = 3


# build the schema to load dataframe.
colnames = ["x1", "x2", "x3", "x4"]
schema = StructType([StructField(colname, FloatType(), False) for colname in colnames])
schema.add(StructField("label", StringType(), True))

assembler = VectorAssembler(inputCols=colnames, outputCol="features")

# load the data into the dataframe
training = spark.read.csv("data/iris_clustering.dat", schema = schema)
training = assembler.transform(training)
training.select(monotonically_increasing_id().alias("p_id"), "features", "label").show(truncate=False)

# Build the model (cluster the data)
kmeans = KMeans().setK(clusters).setSeed(1)
model = kmeans.fit(training)

centers = model.clusterCenters()
print(str(len(centers)))

cost = model.computeCost(training)
print(str(cost))

transformed = model.transform(training).select("features", "label", "prediction")
transformed.show(truncate = False)

print("entries for each cluster:")
clusters = transformed.groupby('prediction').count()
clusters.show(truncate = False)

print("number of iris in a cluster:")
labels_by_cluster = transformed.groupBy('label', 'prediction').count()
labels_by_cluster.show(truncate = False)

#rename columns
clusters_prediction = labels_by_cluster.select(col('prediction').alias('pred'), col('label').alias('label'), col('count').alias('counter'))
clusters_numbers = clusters.select(col('prediction').alias('predict'), col('count').alias('cnt'))

print("percentage of iris by cluster:")
result = clusters.join(clusters_prediction, clusters.prediction == clusters_prediction.pred).select(clusters_prediction.label, clusters.prediction, clusters_prediction.counter, ((col('counter')/col('count')*100)).alias('percentage')).groupBy('label', 'prediction').avg()
result.select('label', 'prediction', col('avg(percentage)').alias('percentage')).show(truncate = False)

#centroids
centroids = model.clusterCenters()
print("cluster centroids:")
for centroid in centroids:
        print(centroid)

spark.stop()
