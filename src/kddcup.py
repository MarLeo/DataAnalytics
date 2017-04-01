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

schema = StructType([StructField("x"+str(colname), schemadata, False) for colname in schemadata.keys()])

# assembler group all x0...x40 into single col call X
assembler = VectorAssembler(inputCols=["x"+str(index) for index in schemadata.keys()[:-1]], outputCol="features")

## TRAINING
training = spark.read.csv("data/kddcup.data", schema = schema)

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
training = training.withColumn("y", (training.x41=="normal.").cast(FloatType))
training.printSchema()
training = training.select("features", "y")
training.printSchema()
training.show()