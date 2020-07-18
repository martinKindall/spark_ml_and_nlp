from pyspark.sql import SparkSession
from sparknlp.annotator import SentenceDetector, Tokenizer, Normalizer, WordEmbeddingsModel
from sparknlp.base import *
from pyspark.ml import Pipeline
from sparknlp.base import LightPipeline


def init_spark():
    spark = SparkSession \
        .builder \
        .appName("HelloWorld") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.5.3s") \
        .config("spark.kryoserializer.buffer.max", "1000M") \
        .getOrCreate()
    sc = spark.sparkContext
    return spark, sc


def main():
    spark, sc = init_spark()

    tweetsDf = spark.read \
        .load("C:\\sparkTmp\\why_i_wear_mask_tweets.csv",
              format="csv", sep=",", inferSchema="true",
              header="true", charset="UTF-8") \
        .select("text")

    pipeline = simplePipeline()
    pipelineModel = pipeline.fit(tweetsDf)
    # result = pipelineModel.transform(tweetsDf)
    # result.show()

    lightModel = LightPipeline(pipelineModel, parse_embeddings=True)
    print(lightModel.annotate("How did serfdom develop in and then leave Russia ?"))


def simplePipeline():
    document_assembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")
    sentenceDetector = SentenceDetector() \
        .setInputCols(["document"]) \
        .setOutputCol("sentences")
    tokenizer = Tokenizer() \
        .setInputCols(["sentences"]) \
        .setOutputCol("token")
    normalizer = Normalizer() \
        .setInputCols(["token"]) \
        .setOutputCol("normal")
    word_embeddings = WordEmbeddingsModel.pretrained() \
        .setInputCols(["document", "normal"]) \
        .setOutputCol("embeddings")
    nlpPipeline = Pipeline(stages=[
        document_assembler,
        sentenceDetector,
        tokenizer,
        normalizer,
        word_embeddings,
    ])
    return nlpPipeline


if __name__ == '__main__':
    main()
