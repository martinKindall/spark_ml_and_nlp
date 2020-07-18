from pyspark.sql import SparkSession
from sparknlp.pretrained import PretrainedPipeline


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

    df = spark.read \
        .load("C:\\sparkTmp\\tweets.csv",
              format="csv", sep=",", inferSchema="true", header="true", charset="UTF-8")

    # Download a pre-trained pipeline
    pipeline = PretrainedPipeline('explain_document_dl', lang='en')

    # Your testing dataset
    text = """
    The Mona Lisa is a 16th century oil painting created by Leonardo.
    It's held at the Louvre in Paris.
    """

    # Annotate your testing dataset
    result = pipeline.annotate(text)

    # What's in the pipeline
    print(list(result.keys()))
    print(result['entities'])


if __name__ == '__main__':
    main()
