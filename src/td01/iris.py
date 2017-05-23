import os
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SQLContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import VectorAssembler

from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import FloatType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import when


# Create spark session
spark = SparkSession.builder.master("local").appName("iris").getOrCreate()


# build the schema to load dataframe.
colnames = ["x1", "x2", "x3", "x4", "y"]
schema = StructType([StructField(colname, FloatType(), False) for colname in colnames])

# assembler group all x1...x4 into single col call X
assembler = VectorAssembler(inputCols=colnames[:-1], outputCol="X")

## TRAINING FOR IRIS_BIN

# load the data into the dataframe
training = spark.read.csv("data/iris_bin.train", schema = schema)
testing = spark.read.csv("data/iris_bin.test", schema = schema)

training = assembler.transform(training) # group all x1...x4 into a single col called X
testing = assembler.transform(testing)

# keep X and y only
training = training.select("X", "y")
testing = testing.select("X", "y")

print("Training Schema: ")
training.printSchema()

print("Show first 20 Training Data: ")
print(training.show(truncate=False))

print("Testing Schema: ")
testing.printSchema()

print("Show first 20 Testing Data: ")
print(testing.show(truncate=False))

# LOGISTIC REGRESSION TO TRAIN THE DATA
lr = LogisticRegression(maxIter=10, regParam=0.03, elasticNetParam=0.8, featuresCol="X", labelCol="y")

# FIT THE MODEL
lrFit = lr.fit(training)

# MAKE PREDICTION
prediction = lrFit.transform(testing)

# SHOW PREDICTIONS
prediction.withColumn("y", when(prediction.y.isNotNull(), 1).otherwise(0))
prediction.select("y", "probability", "prediction").show()

## PRINT THE COEFFICIENTS AND INTERCEPT FOR LOGISTIC REGRESSION
print("Coefficients: " + str(lrFit.coefficientMatrix))
print("Intercept: " + str(lrFit.intercept))

accuracy = 0
lines = 0
for row in prediction.collect():
    print("Expected: " + str(row.y) + "; Prediction: " + str(row.prediction) + " (" + str(row.probability) + ")")
    if row.y == row.prediction:
        accuracy+= 1
    lines+= 1

print("Accuracy " + str(accuracy) + "/" + str(lines) + " = " + (str(float(accuracy)/lines)))
