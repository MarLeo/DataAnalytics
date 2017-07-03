from pyspark.sql import SQLContext
from pyspark.sql.session import SparkSession

from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import FloatType
from pyspark.sql.types import StringType
from pyspark.sql.types import DateType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import DoubleType
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
lines = spark.sparkContext.textFile("data/Customer.txt")
parts = lines.map(lambda l : l.split(","))

# Each line is converted to a tuple
people = parts.map(lambda p: (p[0], p[1], p[2].strip()))

schema = StructType([
    StructField("cid", StringType(), True),
    StructField("startDate", StringType(), True),
    StructField("name", StringType(), True)])


# Apply the schema to RDD
schemaPeople = spark.createDataFrame(people, schema)
schemaPeople.printSchema()
changedTypedf = schemaPeople.withColumn("cid", schemaPeople["cid"].cast(DoubleType()))
new_df = changedTypedf.withColumn("startDate", (changedTypedf["startDate"].cast(DateType())))
new_df.printSchema()

new_df.createOrReplaceTempView("people")

query1 = spark.sql("select * from people").filter(month(col("startDate")) == 7)
query1.printSchema()
query1.show()

query2 = spark.sql("select distinct(name) from people").filter(month(col("startDate")) == 7)
query2.show()

schema_order = StructType([
    StructField("cid", StringType(), True),
    StructField("total", StringType(), True)])

orders = spark.sparkContext.textFile("data/Order.txt").map(lambda l: l.split(",")).map(lambda p: (p[0], p[1]))
schemaOrder = spark.createDataFrame(orders, schema_order)
order_df = schemaOrder.withColumn("cid", schemaOrder["cid"].cast(DoubleType()))
new_order_df = order_df.withColumn("total", order_df["total"].cast(IntegerType()))
new_order_df.createOrReplaceTempView("orders")
new_order_df.printSchema()

query3 = spark.sql("select cid, sum(total) as tot, count(distinct(total)) as cnt from orders group by cid")
query3.show()

query4 = spark.sql("select C.cid, O.total from people C, orders O where C.name like 'B%' and C.cid = O.cid")
query4.show()

