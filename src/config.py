import os
from datetime import datetime, date

# Project root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Current date (for timestamping)
CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")

# Directory paths
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_PATH = os.path.join(ROOT_DIR, "models")
MLFLOW_DIR = os.path.join(ROOT_DIR, "mlflow")

# Delta table paths
DELTA_TABLES = {
    "raw_data": os.path.join(DATA_DIR, "raw", "delta_table"),
    "validated_data": os.path.join(DATA_DIR, "validated", "delta_table"),
    "feature_table": os.path.join(DATA_DIR, "features", "delta_table"),
    "training_data": os.path.join(DATA_DIR, "training", "delta_table"),
    "test_data": os.path.join(DATA_DIR, "test", "delta_table"),
    "validation_data": os.path.join(DATA_DIR, "validation", "delta_table"),
    "risk_scores": os.path.join(DATA_DIR, "scores", "delta_table"),
}

# MLflow configuration
MLFLOW_TRACKING_URI = f"file://{MLFLOW_DIR}"
MLFLOW_EXPERIMENT_NAME = "risk_score_model_training"

# Set to True to use real PostgreSQL database or False to use mock data
USE_POSTGRESQL = True

# PostgreSQL connection details
POSTGRESQL_CONFIG = {
    "url": os.environ.get("DATABASE_URL"),
    "properties": {
        "user": os.environ.get("PGUSER"),
        "password": os.environ.get("PGPASSWORD"),
        "driver": "org.postgresql.Driver",
    },
}

# Define data schema
# These would be defined more thoroughly in a real project
CATEGORICAL_FEATURES = [
    "employment_status",
    "loan_purpose",
    "home_ownership",
    "application_type",
]

NUMERICAL_FEATURES = [
    "loan_amount",
    "annual_income",
    "debt_to_income_ratio",
    "monthly_debt",
    "credit_score",
    "num_credit_inquiries",
    "num_late_payments_30d",
    "num_late_payments_60d",
    "num_late_payments_90d",
    "credit_line_age",
    "loan_term",
    "interest_rate",
]

DATE_FEATURES = [
    "application_date",
]

ID_FEATURES = [
    "application_id",
    "customer_id",
]

# Target column
TARGET_COLUMN = "default_flag"

# Risk score ranges
RISK_SCORE_RANGES = {
    "low": (0, 30),
    "medium": (31, 70),
    "high": (71, 100),
}

# Spark configuration
SPARK_CONFIG = {
    "app_name": "RiskScoreAnalysis",
    "master": "local[*]",
    "config": {
        "spark.sql.extensions": "io.delta.sql.DeltaSparkSessionExtension",
        "spark.sql.catalog.spark_catalog": "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        "spark.jars.packages": "io.delta:delta-core_2.12:2.2.0,org.postgresql:postgresql:42.5.1",
        "spark.sql.warehouse.dir": os.path.join(DATA_DIR, "warehouse"),
        "spark.driver.memory": "4g",
        "spark.executor.memory": "4g",
    },
}

# Mock data configuration (used when PostgreSQL is not available)
MOCK_DATA_CONFIG = {
    "num_records": 1000,
    "default_rate": 0.15,  # 15% default rate
    "random_seed": 42,
}

# Model training configuration
MODEL_TRAINING_CONFIG = {
    "train_ratio": 0.7,
    "validation_ratio": 0.15,
    "test_ratio": 0.15,
    "random_seed": 42,
}

# Model type configuration
MODEL_TYPES = ["spark_lr", "spark_rf", "xgboost"]

# Model-specific hyperparameters
MODEL_HYPERPARAMS = {
    "spark_lr": {
        "maxIter": 10,
        "regParam": 0.3,
        "elasticNetParam": 0.8,
    },
    "spark_rf": {
        "numTrees": 20,
        "maxDepth": 5,
        "seed": 42,
    },
    "xgboost": {
        "max_depth": 6,
        "eta": 0.3,
        "gamma": 0,
        "min_child_weight": 1,
        "subsample": 1,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "num_round": 100,
    },
}