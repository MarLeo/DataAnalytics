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
from pyspark.ml.feature import StringIndexer



# CREATE SPARK SESSION
spark = SparkSession.builder.master("local").appName("Kdd cup 1999").getOrCreate()

# build the schema to load dataframe.
schemadata = {}
schemadata[0] = FloatType()
schemadata[1] = StringType()
schemadata[2] = StringType()
schemadata[3] = StringType()
for i in range(4, 41):
    schemadata[i] = FloatType()

schemadata[41] = StringType()

schema = StructType([StructField("x"+str(colname), schemadata[colname], False) for colname in schemadata.keys()])

# assembler group all x0...x40 into single col call X
assembler = VectorAssembler(inputCols=["x"+str(index) for index in schemadata.keys()[:-1]], outputCol="features")

## TRAINING
training = spark.read.csv("hdfs:///user/bnegrevergne/data_analytics/kddcup.data", schema = schema)

indexers = {}

for i in range(1, 4):
    name = "x"+str(i)
    indexers[name] = StringIndexer(inputCol=name, outputCol="_"+name)
    training = indexers[name].fit(training).transform(training)
    training = training.drop(name)
    training = training.withColumnRenamed("_"+name, name)

training = assembler.transform(training) # group all x1...x4 into a single col called features

# Keep X and y only
training.select(training.x41).show()
training = training.withColumn("y", (training.x41=="normal.").cast(FloatType()))
training.printSchema()
training = training.select("features", "y")
training.printSchema()
training.show()

# Logistic Regression
lr = LogisticRegression(maxIter=100, regParam=0.03, elasticNetParam=0.8, featuresCol="features", labelCol="y")
#print("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

# Learn a LogisticRegression model using parameters in lr
model = lr.fit(training)


# ## TESTING
testing = spark.read.csv("hdfs:///user/bnegrevergne/data_analytics/corrected", schema = schema)
for i in range(1, 4):
    name = "x"+str(i)
    indexers[name] = StringIndexer(inputCol=name, outputCol="_"+name)
    testing = indexers[name].fit(testing).transform(testing)
    testing = testing.drop(name)
    testing = testing.withColumnRenamed("_"+name, name)

testing = assembler.transform(testing)

# Keep X and y only
testing.select(testing.x41).show()
testing = testing.withColumn("y", (testing.x41=="normal.").cast(FloatType()))
testing.printSchema()
testing = testing.select("features", "y")
testing.printSchema()
testing.show()

# Make prediction
prediction = model.transform(testing).select("features", "y", "probability", "prediction")

# Show prediction
prediction.select("y", "probability", "prediction").show()

# show some Statistics
print("Show some statistics: \n")
fraud = prediction.filter(prediction.y == 0).count()
fraud_prediction = prediction.filter(prediction.y == 0).filter(prediction.prediction == 0).count()

normal = prediction.filter(prediction.y == 1).count()
normal_prediction = prediction.filter(prediction.y == 1).filter(prediction.prediction == 1).count()

print("Number of fraud connections : " + str(fraud) + " Number of fraud prediction connections : " + str(fraud_prediction))
print("Number of normal connections : " + str(normal) + " Number of normal prediction connections : " + str(normal_prediction))

#print("Multinomial coefficients : " + str(model.coefficientMatrix()))

print("accuracy on fraud connection: " + str(fraud_prediction) + " / " + str(fraud) + " = " + str(float(fraud_prediction) / fraud))
print("accuracy on normal connection: " + str(normal_prediction) + " / " + str(normal) + " = " + str(float(normal_prediction) / normal))
