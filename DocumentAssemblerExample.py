from pyspark.sql import SparkSession
from sparknlp.base import *
import pyspark.sql.functions as F


def init_spark():
    spark = SparkSession\
        .builder\
        .appName("HelloWorld") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.5.3s") \
        .config("spark.kryoserializer.buffer.max", "1000M") \
        .getOrCreate()
    sc = spark.sparkContext
    return spark, sc


def main():
    spark, sc = init_spark()
    documentAssembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

    tweetsDf = spark.read \
        .load("C:\\sparkTmp\\why_i_wear_mask_tweets.csv",
              format="csv", sep=",", inferSchema="true",
              header="true", charset="UTF-8")\
        .select("text")

    doc_df = documentAssembler.transform(tweetsDf)
    # print(doc_df.select("document.result").take(1))
    doc_df.withColumn(
        "tmp",
        F.explode("document")) \
        .select("tmp.*"). \
        show()


if __name__ == '__main__':
    main()
