Predictive Analytics for Customer Churn

This project predicts customer churn using PySpark and Logistic Regression.
It also includes visualizations like confusion matrix, ROC curve, prediction distribution, and predicted probability histograms to evaluate model performance.
🔥 Project Overview

    Goal: Predict whether a customer will churn based on their service usage and personal information.

    Dataset: Customer churn data (loaded from a CSV file).

    Tech Stack:

        PySpark for large-scale data processing and modeling

        Logistic Regression for binary classification

        Matplotlib and Seaborn for data visualization

        Scikit-learn for evaluation metrics (confusion matrix, ROC curve)

📂 Project Structure

.
├── churn_prediction.py   # Main PySpark script (modeling + graph generation)
├── churn_confusion_matrix.png   # Saved confusion matrix
├── churn_probability_histogram.png  # Histogram of churn probabilities
├── roc_curve.png   # ROC curve plot
├── prediction_distribution.png   # Distribution of predictions
├── README.md   # Project documentation

🚀 How to Run

    Make sure Apache Spark and Python 3.x are installed.

    Install Python dependencies:

pip install pyspark matplotlib seaborn scikit-learn pandas

Place your churn dataset at the location /home/hadoop/churn_dataset or update the dataset path in the script.

Run the script:

    python churn_prediction.py

    Output graphs will be saved in the project folder.

📊 Visualizations

    Confusion Matrix: Shows how many customers were correctly/incorrectly classified.

    ROC Curve: Displays the trade-off between True Positive Rate and False Positive Rate.

    Prediction Distribution: Shows count of predicted churn vs no-churn customers.

    Probability Histogram: Displays the probability of churn for true labels.

📈 Model Performance

    Evaluated using Accuracy and ROC-AUC score.

    Supports further tuning with different classification algorithms if needed.

✨ Future Improvements

    Add hyperparameter tuning (GridSearchCV with PySpark)

    Test with other classifiers like Random Forest, Gradient Boosting

    Deploy the model as a REST API

📜 License

This project is licensed for educational and personal use.
Feel free to modify and enhance!
