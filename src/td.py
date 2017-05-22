from pyspark.sql import SQLContext
from pyspark.sql.session import SparkSession

from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import FloatType
from pyspark.sql.types import StringType
from pyspark.sql.types import DateType
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import month
from pyspark.sql.functions import date_format
from pyspark.sql.functions import col, udf

from datetime import datetime

# Create spark session
spark = SparkSession.builder.master("local").appName("td").getOrCreate()

# This function converts the string cell into a date:
func =  udf (lambda x: datetime.strptime(x, '%mm/%dd/%YYYY'), DateType())

# Create dataframe
df = spark.read.text("Customer.txt")

# Get lines and map them
lines = spark.sparkContext.textFile("Customer.txt")
parts = lines.map(lambda l : l.split(","))

# Each line is converted to a tuple
people = parts.map(lambda p: (p[0], p[1], p[2].strip()))

# Create string schema
#schemaString = "cid startDate name"
#fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split()] 
#schema = StructType(fields)

schema = StructType([
    StructField("cid", IntegerType(), True),
    StructField("startDate", DateType(), True),
    StructField("name", StringType(), True)])


# Apply the schema to RDD
schemaPeople = spark.createDataFrame(people, schema)

schemaPeople1 = schemaPeople #.withColumn('test', func(col('startDate')))

schemaPeople1.printSchema()

schemaPeople1.show(truncate = False)

# Create a temporary view using the dataframe filter(month(datetime.strptime('startDate', 'MM/dd/yyy') == 7))
schemaPeople1.createOrReplaceTempView("people")

