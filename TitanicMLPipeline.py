from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnull, when, count


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


if __name__ == '__main__':
    main()
