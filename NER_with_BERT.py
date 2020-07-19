from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from sparknlp.annotator import BertEmbeddings, NerDLApproach
from sparknlp.training import CoNLL


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
    training_data = CoNLL().readDataset(spark, './spark_nlp/eng.train.txt')
    # training_data.show()

    bert = BertEmbeddings.pretrained('bert_base_cased', 'en') \
        .setInputCols(["sentence", 'token']) \
        .setOutputCol("bert") \
        .setCaseSensitive(False) \
        .setPoolingLayer(0)

    nerTagger = NerDLApproach() \
        .setInputCols(["sentence", "token", "bert"]) \
        .setLabelColumn("label") \
        .setOutputCol("ner") \
        .setMaxEpochs(1) \
        .setRandomSeed(0) \
        .setVerbose(1) \
        .setValidationSplit(0.2) \
        .setEvaluationLogExtended(True) \
        .setEnableOutputLogs(True) \
        .setIncludeConfidence(True)
        #.setTestDataset("test_withEmbeds.parquet")

    '''
    test_data = CoNLL().readDataset(spark, './spark_nlp/eng.testa.txt')
    test_data = bert.transform(test_data)
    test_data.write.parquet("test_withEmbeds.parquet")
    '''

    ner_pipeline = Pipeline(stages=[bert, nerTagger])
    ner_model = ner_pipeline.fit(training_data)


if __name__ == '__main__':
    main()
