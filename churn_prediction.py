from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # ✅ Initialize Spark session
    logger.info("Initializing Spark session...")
    spark = SparkSession.builder \
        .appName("Customer Churn Prediction") \
        .getOrCreate()

    try:
        # ✅ Load the dataset
        logger.info("Loading dataset...")
        df = spark.read.csv("/home/hadoop/churn_dataset", header=True, inferSchema=True)

        # ✅ Drop rows with missing/null values
        logger.info("Dropping rows with missing values...")
        df = df.dropna()

        # ✅ Rename columns for easier access
        logger.info("Renaming columns...")
        df = df.withColumnRenamed("Gender", "gender") \
            .withColumnRenamed("Partner", "partner") \
            .withColumnRenamed("Dependents", "dependents") \
            .withColumnRenamed("Tenure Months", "tenure") \
            .withColumnRenamed("Phone Service", "phone_service") \
            .withColumnRenamed("Internet Service", "internet_service") \
            .withColumnRenamed("Contract", "contract") \
            .withColumnRenamed("Payment Method", "payment_method") \
            .withColumnRenamed("Monthly Charges", "monthly_charges") \
            .withColumnRenamed("Total Charges", "total_charges") \
            .withColumnRenamed("Churn Label", "churn")

        # ✅ Index categorical columns
        logger.info("Indexing categorical columns...")
        indexers = [
            StringIndexer(inputCol=col, outputCol=col + "_index") for col in ['gender', 'partner', 'dependents', 'phone_service', 'internet_service', 'contract', 'payment_method', 'churn']
        ]
        for indexer in indexers:
            df = indexer.fit(df).transform(df)

        # ✅ Clean the total_charges column (remove spaces and cast to double)
        logger.info("Cleaning 'total_charges' column...")
        df = df.withColumn("total_charges_clean", regexp_replace(col("total_charges"), " ", ""))
        df = df.withColumn("total_charges_clean", col("total_charges_clean").cast("double"))

        # Cast monthly_charges just in case
        df = df.withColumn("monthly_charges", col("monthly_charges").cast("double"))

        # ✅ Assemble features
        logger.info("Assembling features...")
        assembler = VectorAssembler(
            inputCols=['tenure', 'monthly_charges', 'total_charges_clean'] +
                      [col + "_index" for col in ['gender', 'partner', 'dependents', 'phone_service', 'internet_service', 'contract', 'payment_method']],
            outputCol="features"
        )
        df = assembler.transform(df)

        # ✅ Rename churn_index to label
        df = df.withColumnRenamed("churn_index", "label")

        # ✅ Split the data
        logger.info("Splitting data into train and test sets...")
        train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

        # ✅ Train logistic regression model
        logger.info("Training Logistic Regression model...")
        lr = LogisticRegression(featuresCol="features", labelCol="label")
        model = lr.fit(train_data)

        # ✅ Make predictions
        logger.info("Making predictions on test data...")
        predictions = model.transform(test_data)

        # ✅ Show prediction results
        logger.info("Showing prediction results...")
        predictions.select("label", "prediction", "probability").show(10)

        # ✅ Evaluate the model
        logger.info("Evaluating model...")
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        logger.info(f"Model Accuracy: {accuracy:.2f}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

    finally:
        # Stop Spark session
        logger.info("Stopping Spark session...")
        spark.stop()

if __name__ == "__main__":
    main()

