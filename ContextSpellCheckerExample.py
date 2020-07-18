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

    empty_ds = spark.createDataFrame([[""]]).toDF("text")

    spellModel = ContextSpellCheckerModel \
        .pretrained() \
        .setInputCols("token") \
        .setOutputCol("checked")
    spellModel.updateVocabClass('_NAME_', ['Monika', 'Agnieszka', 'Inga', 'Jowita', 'Melania'], True)

    pipeline = spellCheckerPipeline(spellModel)
    fittedPipeline = pipeline.fit(empty_ds)

    applyModelToTweetsAndShowResult(fittedPipeline, tweetsDf)

    lp = LightPipeline(fittedPipeline)

    showExampleOfSpellChecker(lp)
    showExampleOfNameCheckingAndWordClasses(spellModel, lp)

    print(spellModel.getTradeoff())  # this is a hyperparam that can be tuned to balance
                                     # context information and word/subword information.


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


def applyModelToTweetsAndShowResult(fittedPipeline, tweetsDf):
    result = fittedPipeline.transform(tweetsDf)
    result.show()


def showExampleOfSpellChecker(lp):
    print(lp.annotate("Plaese alliow me tao introdduce myhelf, I am a man of waelth und tiaste"))


def showExampleOfNameCheckingAndWordClasses(spellModel, lp):
    print(spellModel.getWordClasses())
    foreignNameExample = 'We are going to meet Jowita at the city hall.'
    print(lp.annotate(foreignNameExample))


if __name__ == '__main__':
    main()
