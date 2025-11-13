Titanic Classification Pipeline

This project provides a complete machine learning pipeline to predict passenger survival on the Titanic dataset. The pipeline includes data preprocessing, outlier handling, feature scaling, encoding categorical features, and model training using Logistic Regression.

Features

Custom Outlier Handling: Automatically identifies skewed and normal numeric features and clips outliers based on IQR or standard deviation.

Data Preprocessing:

Imputation of missing numeric and categorical values.

Standard scaling of numeric features.

One-hot encoding for categorical features.

Visualization: Histogram and box plots for visual inspection of feature distribution and outliers (can be disabled for automated usage).

Model Training: Logistic Regression classifier.

Evaluation Metrics:

Accuracy

Precision

Recall

F1-score

Confusion matrix

Installation

Clone the repository:

git clone https://github.com/your-username/Titanic_Classification_Pipeline.git

cd Titanic_Classification_Pipeline

Install required packages:

pip install -r requirements.txt

Usage

Run the pipeline:

python Titanic_Classification_Pipeline.py

The script will preprocess the Titanic dataset, train a Logistic Regression model, and display evaluation metrics.

To run without interactive inputs, modify the script by disabling input prompts for rows/columns.

File Structure

Titanic_Classification_Pipeline/
├── Titanic_Classification_Pipeline.py
├── README.md
└── requirements.txt

Contributing

Contributions are welcome! Fork the repository and create a pull request.

License

MIT License.
