# Telecom Customer Churn Prediction

## Project Overview

### Business Context

A telecommunications company offers its customers high-speed Internet, landline and mobile phone services, streaming, technical support, and different subscription plans. The company has seen an increase in customer cancellations, reducing revenue and customer loyalty while increasing the cost of acquiring new customers.

To anticipate customer departures and support retention efforts, the company wants to use its historical data through data science.

### Business Problem

**How can historical customer data be used to identify customers at high risk of churn and help the company develop retention strategies?**

### Overall Objective

Develop a machine learning model that predicts whether a customer will cancel their subscription (`Churn = Yes`) or remain a customer (`Churn = No`).

### Specific Objectives

1. Analyze customer data.
2. Identify factors associated with customer churn.
3. Clean and prepare the data.
4. Perform exploratory data analysis (EDA).
5. Build several classification models.
6. Compare model performance.
7. Select the best-performing model.
8. Interpret predictions.
9. Create a Streamlit application to estimate churn risk.
10. Develop business recommendations to improve customer retention.

The project covers data exploration, cleaning, analysis, feature preparation, model training, and evaluation. An interactive Streamlit application provides a dashboard, individual customer risk predictions, batch predictions from a CSV file, and model performance metrics.

## Model Results

The final model is a **Gradient Boosting** pipeline with feature preprocessing. The results below were recalculated using the saved model (`models/churn_model.pkl`) and the project's test files (`data/processed/X_test.csv` and `data/processed/y_test.csv`), comprising 1,409 customers.

| Metric | Result |
| :--- | ---: |
| Accuracy | 78.42% |
| Precision | 57.61% |
| Recall | 70.86% |
| F1-score | 63.55% |
| ROC-AUC | 84.62% |

Classification uses a **decision threshold of 0.35**: a churn probability greater than or equal to 35% classifies a customer as likely to churn. Accuracy, precision, recall, and F1-score use this threshold; ROC-AUC is calculated from the predicted probabilities.

Confusion matrix (rows: actual class; columns: predicted class):

| Actual class | Predicted `No` | Predicted `Yes` |
| :--- | ---: | ---: |
| `No` | 840 | 195 |
| `Yes` | 109 | 265 |

## Application Features

- **Landing page**: concise overview of the solution, key model results, and shortcuts to the main workflows.
- **Dashboard**: customer counts, churn rate, contracts, payment methods, and tenure analysis.
- **Individual prediction**: enter a customer's details and receive a probability, decision, and risk level.
- **Batch prediction**: upload a CSV, identify at-risk customers, filter the results, and download an enriched file.
- **Model performance**: accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and ROC curve.

## Application Interface

The Streamlit interface is designed as a professional telecommunications analytics platform:

- Orange, charcoal, and teal palette inspired by telecom operators.
- Responsive layout for desktop, tablet, and mobile screens.
- Structured navigation with Material icons and sentence-case labels.
- Card-based KPIs and charts with consistent spacing and contrast.
- Centralized theme configuration in `.streamlit/config.toml`.

The application contains five sections:

1. **Accueil**: project presentation and quick access to the main functions.
2. **Dashboard**: descriptive analysis of customer churn.
3. **Prédiction individuelle**: probability estimation for one customer.
4. **Prédiction en masse**: CSV-based predictions and export.
5. **Performance du modèle**: test-set evaluation and diagnostic charts.

## Project Structure

Project files, excluding virtual environments and tool-generated internal files:

```text
IBM/
├── .gitignore
├── README.md
├── .streamlit/
│   └── config.toml
├── app/
│   └── streamlit_app.py
├── data/
│   ├── IBM-Telco-Customer-Churn.csv
│   ├── IBM-Telco-Customer-Churn_cleaned.csv
│   ├── raw/
│   │   └── telco_customer_churn.csv
│   ├── processed/
│   │   ├── X_test.csv
│   │   ├── y_test.csv
│   │   └── cleaned_data.csv
├── models/
│   └── churn_model.pkl
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   └── 06_model_evaluation.ipynb
├── reports/
│   ├── model_comparison.csv
│   └── figures/
│       ├── churn_distribution.png
│       ├── churn_by_contract.png
│       ├── churn_by_payment.png
│       ├── churn_by_MonthlyCharges.png
│       ├── churn_by_TechSupport.png
│       ├── churn_by_TotalCharges.png
│       ├── churm_by_gender.png
│       ├── churm_by_tenure.png
│       └── correlation_heatmap.png
└── requirements.txt
```

## Dataset

This project uses the public **IBM Telco Customer Churn** dataset, a reference dataset for studying customer churn in the telecommunications industry.

| Item | Description |
| :--- | :--- |
| Number of customers | 7,043 |
| Number of columns in the original dataset | 21, including the customer ID |
| Features used by the model | 19 |
| Target variable | `Churn` |
| Problem type | Binary classification |
| Target values | `Yes` / `No` |
| Format | CSV |

The dataset includes customer ID (`customerID`), gender (`gender`), senior-citizen status (`SeniorCitizen`), family status (`Partner`, `Dependents`), tenure (`tenure`), phone and Internet services (`PhoneService`, `InternetService`), online security (`OnlineSecurity`), technical support (`TechSupport`), contract type (`Contract`), paperless billing (`PaperlessBilling`), payment method (`PaymentMethod`), monthly and total charges (`MonthlyCharges`, `TotalCharges`), and the target `Churn`.

The `customerID` field is not used as a model input. Once it is excluded, the 19 features and the target produce 20 columns in the prepared dataset.

The observed class distribution is approximately:

- 73.46% retained customers (`No`).
- 26.54% churned customers (`Yes`).

The data is split, using stratification, into 80% for training (5,634 customers) and 20% for testing (1,409 customers).

## Installation

### 1. Create and activate a virtual environment

From the project directory in Windows PowerShell:

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The `requirements.txt` file lists the dependencies for the Streamlit application and notebooks: pandas, numpy, matplotlib, seaborn, scikit-learn, streamlit, plotly, joblib, and notebook.

## Usage

### Run the Streamlit application

```powershell
streamlit run app/streamlit_app.py
```

The application is normally available at:

```text
http://localhost:8501
```

If port `8501` is already in use, Streamlit may select another port or you can specify one explicitly:

```powershell
streamlit run app/streamlit_app.py --server.port 8502
```

### Run the notebooks

```powershell
jupyter notebook
```

Run the notebooks in order, from `01_data_exploration.ipynb` through `06_model_evaluation.ipynb`.

## Batch Predictions

The CSV uploaded to the application must contain the 19 features used by the model:

```text
gender
SeniorCitizen
Partner
Dependents
tenure
PhoneService
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaperlessBilling
PaymentMethod
MonthlyCharges
TotalCharges
```

The `Churn` column is not required. Additional columns, such as `customerID`, may be included in the input file and will be retained in the exported results.

## Modeling Pipeline

The main pipeline performs the following steps:

1. Imputes missing values.
2. Scales numerical features.
3. Encodes categorical features.
4. Trains a Gradient Boosting model.
5. Generates churn probabilities.
6. Classifies customers using a 35% decision threshold.

The final model is saved at `models/churn_model.pkl`. The test data used by the application is available in `data/processed/X_test.csv` and `data/processed/y_test.csv`.

## Copyright

© 2026 Telecom Churn Intelligence. Tous droits réservés.

## Auteur

Développé et maintenu par **diak224-dot**.
