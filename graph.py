from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace
import logging

# ✅ For plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_confusion_matrix(predictions, output_path="churn_confusion_matrix.png"):
    y_true = predictions.select("label").rdd.flatMap(lambda x: x).collect()
    y_pred = predictions.select("prediction").rdd.flatMap(lambda x: x).collect()

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def main():
    logger.info("Starting Spark session...")
    spark = SparkSession.builder \
        .appName("Customer Churn Prediction with Graph") \
        .getOrCreate()

    try:
        logger.info("Loading dataset...")
        df = spark.read.csv("/home/hadoop/churn_dataset", header=True, inferSchema=True).dropna()

        # Rename and preprocess
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

        indexers = [StringIndexer(inputCol=col, outputCol=col + "_index") for col in
                    ['gender', 'partner', 'dependents', 'phone_service', 'internet_service', 'contract', 'payment_method', 'churn']]
        for indexer in indexers:
            df = indexer.fit(df).transform(df)

        df = df.withColumn("total_charges_clean", regexp_replace(col("total_charges"), " ", "").cast("double"))
        df = df.withColumn("monthly_charges", col("monthly_charges").cast("double"))

        assembler = VectorAssembler(
            inputCols=['tenure', 'monthly_charges', 'total_charges_clean'] +
                      [col + "_index" for col in ['gender', 'partner', 'dependents', 'phone_service', 'internet_service', 'contract', 'payment_method']],
            outputCol="features"
        )
        df = assembler.transform(df)
        df = df.withColumnRenamed("churn_index", "label")

        train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

        logger.info("Training Logistic Regression model...")
        lr = LogisticRegression(featuresCol="features", labelCol="label")
        model = lr.fit(train_data)

        logger.info("Making predictions...")
        predictions = model.transform(test_data)

        # ✅ Evaluate
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        logger.info(f"Model Accuracy: {accuracy:.2f}")

        # ✅ Plot and Save Confusion Matrix
        logger.info("Plotting confusion matrix...")
        plot_confusion_matrix(predictions, output_path="churn_confusion_matrix.png")

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")

    finally:
        logger.info("Stopping Spark session...")
        spark.stop()

if __name__ == "__main__":
    main()

