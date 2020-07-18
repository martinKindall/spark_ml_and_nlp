from pyspark.sql import SparkSession
from sparknlp.annotator import RecursiveTokenizer, ContextSpellCheckerModel
from sparknlp.base import *
from pyspark.ml import Pipeline


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

    spellModel = ContextSpellCheckerModel \
        .pretrained() \
        .setInputCols("token") \
        .setOutputCol("checked")

    pipeline = spellCheckerPipeline(spellModel)
    lp = LightPipeline(pipeline.fit(tweetsDf))
    print(lp.annotate("Plaese alliow me tao introdduce myhelf, I am a man of waelth und tiaste"))
    print(spellModel.getWordClasses())


def spellCheckerPipeline(spellModel):
    documentAssembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")
    tokenizer = RecursiveTokenizer() \
        .setInputCols(["document"]) \
        .setOutputCol("token") \
        .setPrefixes(["\"", "(", "[", "\n"]) \
        .setSuffixes([".", ", ", "?", ")", "!", "‘s"])

    finisher = Finisher() \
        .setInputCols("checked")

    pipeline = Pipeline(stages=[
        documentAssembler,
        tokenizer,
        spellModel,
        finisher
    ])

    return pipeline


if __name__ == '__main__':
    main()
