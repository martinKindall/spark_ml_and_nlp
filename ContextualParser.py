from pyspark.sql import SparkSession
from sparknlp.annotator import SentenceDetector, Tokenizer
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

    pipeline = contextualParserPipeline()
    fittedPipeline = pipeline.fit(empty_ds)

    sampleText = '''
    A patient has liver metastases pT1bN0M0 and the T5 primary 
    site may be colon or lung. If the primary site is not clearly identified , 
    this case is cT4bcN2M1, Stage Grouping 88. N4 A child T?N3M1  
    has soft tissue aM3 sarcoma and the staging has been left unstaged. 
    Both clinical and pathologic staging would be coded pT1bN0M0 
    as unstageable cT3cN2.Medications started.
    '''

    light_model = LightPipeline(fittedPipeline)
    annotations = light_model.fullAnnotate(sampleText)[0]
    print(annotations)


def contextualParserPipeline():
    document_assembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

    sentence_detector = SentenceDetector() \
        .setInputCols(["document"]) \
        .setOutputCol("sentence")

    tokenizer = Tokenizer() \
        .setInputCols(["sentence"]) \
        .setOutputCol("token")

    stage_contextual_parser = ChunkEntityResolverApproach() \
        .setInputCols(["sentence", "token"]) \
        .setOutputCol("entity_stage") \
        .setJsonPath("data/Stage.json")

    parser_pipeline = Pipeline(stages=[
        document_assembler,
        sentence_detector,
        tokenizer,
        stage_contextual_parser])

    return parser_pipeline


if __name__ == '__main__':
    main()
