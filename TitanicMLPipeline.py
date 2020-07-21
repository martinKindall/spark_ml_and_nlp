from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnull, when, count
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Titanic Data") \
        .getOrCreate()
    sc = spark.sparkContext
    return spark, sc


def main():
    spark, sc = init_spark()

    df = (spark.read
          .format("csv")
          .option('header', 'true')
          .load("C:\\sparkTmp\\titanic_train.csv"))

    df.show(5)

    # How many rows we have
    print(df.count())

    # The names of our columns
    print(df.columns)

    # Types of our columns
    print(df.dtypes)

    print(df.describe())

    dataset = df.select(
        col("Survived").cast("float"),
        col("Pclass").cast("float"),
        col("Sex"),
        col("Age").cast("float"),
        col("Fare").cast("float"),
        col("Embarked"),
    )

    dataset.show()

    dataset.select([count(when(isnull(c), c)).alias(c) for c in dataset.columns]).show()
    dataset = dataset.dropna(how="any")
    dataset.select([count(when(isnull(c), c)).alias(c) for c in dataset.columns]).show()

    # We need to transform Sex and Embarked to numerical value
    dataset = StringIndexer(
        inputCol="Sex",
        outputCol="Gender",
        handleInvalid="keep"
    ).fit(dataset).transform(dataset)

    dataset = StringIndexer(
        inputCol="Embarked",
        outputCol="Boarded",
        handleInvalid="keep"
    ).fit(dataset).transform(dataset)

    # StringIndexer transforms not just to a plain double, but preserves category
    print(dataset.schema.fields[7].metadata)

    dataset = dataset.drop("Sex")
    dataset = dataset.drop("Embarked")

    dataset.show()

    required_features = [
        "Pclass",
        "Age",
        "Fare",
        "Gender",
        "Boarded"
    ]

    assembler = VectorAssembler(
        inputCols=required_features,
        outputCol='features'
    )

    transformed_data = assembler.transform(dataset)
    transformed_data.show()

    (training_data, test_data) = transformed_data.randomSplit([0.8, 0.2])
    rf = RandomForestClassifier(
        labelCol="Survived",
        featuresCol="features",
        maxDepth=5
    )
    model = rf.fit(training_data)
    predictions = model.transform(test_data)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="Survived",
        predictionCol="prediction",
        metricName="accuracy"
    )

    accuracy = evaluator.evaluate(predictions)
    print("Test Accuracy = ", accuracy)


if __name__ == '__main__':
    main()
