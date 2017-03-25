import os
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SQLContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import VectorAssembler

from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import FloatType

# Create spark session
spark = SparkSession.builder.master("local").appName("iris").getOrCreate()


# build the schema to load dataframe.
colnames = ["x1", "x2", "x3", "x4", "y"]
schema = StructType([StructField(colname, FloatType(), False) for colname in colnames])

# assembler group all x1...x4 into single col call X
assembler = VectorAssembler(inputCols=colnames[:-1], outputCol="X")

## TRAINING

# load the data into the dataframe
training = spark.read.csv("data/iris_bin.train", schema = schema)
training = assembler.transform(training) # group all x1...x4 into a single col called X

# keep X and y only
training = training.select("X", "y")

print("Schema: ")
training.printSchema()

print("Data")
print(training.show(truncate=False))