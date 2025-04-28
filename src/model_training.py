import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.utils import SPARK_AVAILABLE

# Import PySpark classes if available
if SPARK_AVAILABLE:
    from pyspark.ml.classification import LogisticRegression as SparkLogisticRegression
    from pyspark.ml.classification import RandomForestClassifier as SparkRandomForestClassifier
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder as SparkOneHotEncoder
    from pyspark.ml.feature import StandardScaler as SparkStandardScaler
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    from pyspark.ml import Pipeline as SparkPipeline
    from pyspark.sql import SparkSession, DataFrame

from src.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    MODEL_PATH,
    MODEL_TRAINING_CONFIG,
    MODEL_HYPERPARAMS
)
from src.utils import (
    get_spark_session,
    create_directory_if_not_exists,
    setup_mlflow,
    save_model_metadata
)
from src.data_generator import generate_sample_data

logger = logging.getLogger(__name__)

def get_compatible_one_hot_encoder(drop='first', handle_unknown='ignore'):
    """
    Create a OneHotEncoder that's compatible with the installed scikit-learn version.
    Handles the parameter change from 'sparse' to 'sparse_output' in newer versions.
    
    Args:
        drop: Parameter to pass to OneHotEncoder constructor
        handle_unknown: Parameter to pass to OneHotEncoder constructor
        
    Returns:
        OneHotEncoder instance
    """
    try:
        # Try with newer scikit-learn (sparse_output parameter)
        return OneHotEncoder(drop=drop, sparse_output=False, handle_unknown=handle_unknown)
    except TypeError:
        # Fall back to older scikit-learn (sparse parameter)
        return OneHotEncoder(drop=drop, sparse=False, handle_unknown=handle_unknown)

class ModelTrainer:
    """
    Class for training machine learning models for risk scoring.
    """
    
    def __init__(self, spark=None):
        """
        Initialize the model trainer.
        
        Args:
            spark: SparkSession object (optional)
        """
        # Handle PySpark availability
        if SPARK_AVAILABLE:
            self.spark = spark or get_spark_session()
        else:
            self.spark = None
        
        # Set random seed
        np.random.seed(MODEL_TRAINING_CONFIG["random_seed"])
        
        # Initialize MLflow
        setup_mlflow()
    
    def load_training_data(self, csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load training data from a CSV file or generate sample data.
        
        Args:
            csv_path: Path to the CSV file (optional)
            
        Returns:
            DataFrame with training data
        """
        if csv_path and os.path.exists(csv_path):
            # Load data from CSV
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} records from {csv_path}")
        else:
            # Generate sample data
            logger.info("No CSV file provided or file does not exist. Generating sample data.")
            csv_path = generate_sample_data()
            df = pd.read_csv(csv_path)
            logger.info(f"Generated {len(df)} sample records")
        
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Prepare data for model training by splitting into train, validation, and test sets.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with prepared data splits and feature information
        """
        # Make a copy to avoid modifying the original DataFrame
        data = df.copy()
        
        # Select relevant columns
        feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        columns_to_use = feature_cols + [TARGET_COLUMN]
        
        # Check if all required columns are present
        missing_cols = [col for col in columns_to_use if col not in data.columns]
        if missing_cols:
            logger.warning(f"Missing columns in data: {missing_cols}")
            logger.warning(f"Available columns: {data.columns.tolist()}")
            raise ValueError(f"Required columns are missing in the data: {missing_cols}")
        
        # Select only the columns we need
        data = data[columns_to_use].copy()
        
        # Handle missing values
        for col in NUMERICAL_FEATURES:
            if col in data.columns:
                data[col].fillna(data[col].median(), inplace=True)
        
        for col in CATEGORICAL_FEATURES:
            if col in data.columns:
                data[col].fillna(data[col].mode()[0], inplace=True)
        
        # Split data into features and target
        X = data.drop(TARGET_COLUMN, axis=1)
        y = data[TARGET_COLUMN]
        
        # Split data into train, validation, and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=MODEL_TRAINING_CONFIG["test_ratio"],
            random_state=MODEL_TRAINING_CONFIG["random_seed"],
            stratify=y
        )
        
        # Further split train_val into train and validation
        test_ratio = MODEL_TRAINING_CONFIG["test_ratio"]
        val_ratio = MODEL_TRAINING_CONFIG["validation_ratio"] / (1 - test_ratio)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_ratio,
            random_state=MODEL_TRAINING_CONFIG["random_seed"],
            stratify=y_train_val
        )
        
        # Create DataFrames for each split
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        # Log split sizes
        logger.info(f"Training set: {len(train_df)} records")
        logger.info(f"Validation set: {len(val_df)} records")
        logger.info(f"Test set: {len(test_df)} records")
        
        # Create data dictionary
        data_dict = {
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test
        }
        
        # Create feature information dictionary
        feature_info = {
            "numerical_features": [col for col in NUMERICAL_FEATURES if col in X.columns],
            "categorical_features": [col for col in CATEGORICAL_FEATURES if col in X.columns],
            "feature_names": X.columns.tolist()
        }
        
        return data_dict, feature_info
    
    def train_xgboost_model(self, data_dict: Dict[str, Any], feature_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train an XGBoost model.
        
        Args:
            data_dict: Dictionary with prepared data
            feature_info: Dictionary with feature information
            
        Returns:
            Dictionary with trained model and evaluation results
        """
        logger.info("Training XGBoost model")
        
        # Extract data
        X_train = data_dict["X_train"]
        y_train = data_dict["y_train"]
        X_val = data_dict["X_val"]
        y_val = data_dict["y_val"]
        X_test = data_dict["X_test"]
        y_test = data_dict["y_test"]
        
        # Create a preprocessor for the features
        categorical_features = feature_info["categorical_features"]
        numerical_features = feature_info["numerical_features"]
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', get_compatible_one_hot_encoder(), categorical_features)
            ],
            remainder='drop'
        )
        
        # Fit the preprocessor on the training data
        preprocessor.fit(X_train)
        
        # Transform the data
        X_train_processed = preprocessor.transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        X_test_processed = preprocessor.transform(X_test)
        
        # Get hyperparameters
        xgb_params = MODEL_HYPERPARAMS["xgboost"].copy()
        
        # Create DMatrix objects for XGBoost
        dtrain = xgb.DMatrix(X_train_processed, label=y_train)
        dval = xgb.DMatrix(X_val_processed, label=y_val)
        dtest = xgb.DMatrix(X_test_processed, label=y_test)
        
        # Create a watchlist to monitor training
        watchlist = [(dtrain, 'train'), (dval, 'validation')]
        
        # Train the model
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=xgb_params.pop("num_round", 100),
            evals=watchlist,
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        # Make predictions
        y_train_pred = model.predict(dtrain)
        y_val_pred = model.predict(dval)
        y_test_pred = model.predict(dtest)
        
        # Calculate evaluation metrics
        train_metrics = self._calculate_metrics(y_train, y_train_pred)
        val_metrics = self._calculate_metrics(y_val, y_val_pred)
        test_metrics = self._calculate_metrics(y_test, y_test_pred)
        
        # Log metrics
        logger.info(f"XGBoost Training Metrics: {train_metrics}")
        logger.info(f"XGBoost Validation Metrics: {val_metrics}")
        logger.info(f"XGBoost Test Metrics: {test_metrics}")
        
        # Get feature importance
        feature_names = preprocessor.get_feature_names_out()
        feature_importance = model.get_score(importance_type='gain')
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("XGBoost Feature Importance:")
        for feature, importance in sorted_importance[:10]:
            logger.info(f"{feature}: {importance}")
        
        # Create result dictionary
        result = {
            "model": model,
            "preprocessor": preprocessor,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "feature_importance": feature_importance,
            "model_type": "xgboost"
        }
        
        return result
    
    def train_spark_lr_model(self, data_dict: Dict[str, Any], feature_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train a Spark Logistic Regression model.
        
        Args:
            data_dict: Dictionary with prepared data
            feature_info: Dictionary with feature information
            
        Returns:
            Dictionary with trained model and evaluation results
        """
        logger.info("Training Spark Logistic Regression model")
        
        # Check if Spark is available
        if not SPARK_AVAILABLE:
            logger.warning("PySpark is not available. Cannot train Spark Logistic Regression model.")
            logger.warning("Falling back to scikit-learn Logistic Regression model.")
            return self.train_sklearn_lr_model(data_dict, feature_info)
        
        # Convert pandas DataFrames to Spark DataFrames
        train_spark_df = self.spark.createDataFrame(data_dict["train_df"])
        val_spark_df = self.spark.createDataFrame(data_dict["val_df"])
        test_spark_df = self.spark.createDataFrame(data_dict["test_df"])
        
        # Create a feature pipeline
        categorical_features = feature_info["categorical_features"]
        numerical_features = feature_info["numerical_features"]
        
        # Initialize transformers
        indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_idx") 
                   for col in categorical_features]
        
        encoders = [SparkOneHotEncoder(inputCol=f"{col}_idx", outputCol=f"{col}_enc") 
                   for col in categorical_features]
        
        # Define the feature columns
        numerical_cols = numerical_features
        categorical_cols = [f"{col}_enc" for col in categorical_features]
        
        # Create the vector assembler
        assembler = VectorAssembler(
            inputCols=numerical_cols + categorical_cols,
            outputCol="features"
        )
        
        # Create the scaler
        scaler = SparkStandardScaler(
            inputCol="features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True
        )
        
        # Get hyperparameters
        lr_params = MODEL_HYPERPARAMS["spark_lr"].copy()
        
        # Create the logistic regression model
        lr = LogisticRegression(
            featuresCol="scaled_features",
            labelCol=TARGET_COLUMN,
            maxIter=lr_params.get("maxIter", 10),
            regParam=lr_params.get("regParam", 0.3),
            elasticNetParam=lr_params.get("elasticNetParam", 0.8)
        )
        
        # Create the pipeline
        pipeline = SparkPipeline(stages=indexers + encoders + [assembler, scaler, lr])
        
        # Fit the pipeline
        model = pipeline.fit(train_spark_df)
        
        # Make predictions
        train_predictions = model.transform(train_spark_df)
        val_predictions = model.transform(val_spark_df)
        test_predictions = model.transform(test_spark_df)
        
        # Extract the LR model from the pipeline
        lr_model = model.stages[-1]
        
        # Get predictions for evaluation
        train_pred_pd = train_predictions.select(TARGET_COLUMN, "prediction", "probability").toPandas()
        val_pred_pd = val_predictions.select(TARGET_COLUMN, "prediction", "probability").toPandas()
        test_pred_pd = test_predictions.select(TARGET_COLUMN, "prediction", "probability").toPandas()
        
        # Extract probability of positive class (class 1)
        train_pred_pd["prob"] = train_pred_pd["probability"].apply(lambda x: float(x[1]))
        val_pred_pd["prob"] = val_pred_pd["probability"].apply(lambda x: float(x[1]))
        test_pred_pd["prob"] = test_pred_pd["probability"].apply(lambda x: float(x[1]))
        
        # Calculate evaluation metrics
        train_metrics = self._calculate_metrics(
            train_pred_pd[TARGET_COLUMN], train_pred_pd["prob"])
        val_metrics = self._calculate_metrics(
            val_pred_pd[TARGET_COLUMN], val_pred_pd["prob"])
        test_metrics = self._calculate_metrics(
            test_pred_pd[TARGET_COLUMN], test_pred_pd["prob"])
        
        # Log metrics
        logger.info(f"Spark LR Training Metrics: {train_metrics}")
        logger.info(f"Spark LR Validation Metrics: {val_metrics}")
        logger.info(f"Spark LR Test Metrics: {test_metrics}")
        
        # Create result dictionary
        result = {
            "model": model,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "model_type": "spark_lr"
        }
        
        return result
    
    def train_spark_rf_model(self, data_dict: Dict[str, Any], feature_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train a Spark Random Forest model.
        
        Args:
            data_dict: Dictionary with prepared data
            feature_info: Dictionary with feature information
            
        Returns:
            Dictionary with trained model and evaluation results
        """
        logger.info("Training Spark Random Forest model")
        
        # Check if Spark is available
        if not SPARK_AVAILABLE:
            logger.warning("PySpark is not available. Cannot train Spark Random Forest model.")
            logger.warning("Falling back to scikit-learn Random Forest model.")
            return self.train_sklearn_rf_model(data_dict, feature_info)
        
        # Convert pandas DataFrames to Spark DataFrames
        train_spark_df = self.spark.createDataFrame(data_dict["train_df"])
        val_spark_df = self.spark.createDataFrame(data_dict["val_df"])
        test_spark_df = self.spark.createDataFrame(data_dict["test_df"])
        
        # Create a feature pipeline
        categorical_features = feature_info["categorical_features"]
        numerical_features = feature_info["numerical_features"]
        
        # Initialize transformers
        indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_idx") 
                   for col in categorical_features]
        
        encoders = [SparkOneHotEncoder(inputCol=f"{col}_idx", outputCol=f"{col}_enc") 
                   for col in categorical_features]
        
        # Define the feature columns
        numerical_cols = numerical_features
        categorical_cols = [f"{col}_enc" for col in categorical_features]
        
        # Create the vector assembler
        assembler = VectorAssembler(
            inputCols=numerical_cols + categorical_cols,
            outputCol="features"
        )
        
        # Get hyperparameters
        rf_params = MODEL_HYPERPARAMS["spark_rf"].copy()
        
        # Create the random forest model
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol=TARGET_COLUMN,
            numTrees=rf_params.get("numTrees", 20),
            maxDepth=rf_params.get("maxDepth", 5),
            seed=rf_params.get("seed", 42)
        )
        
        # Create the pipeline
        pipeline = SparkPipeline(stages=indexers + encoders + [assembler, rf])
        
        # Fit the pipeline
        model = pipeline.fit(train_spark_df)
        
        # Make predictions
        train_predictions = model.transform(train_spark_df)
        val_predictions = model.transform(val_spark_df)
        test_predictions = model.transform(test_spark_df)
        
        # Extract the RF model from the pipeline
        rf_model = model.stages[-1]
        
        # Get predictions for evaluation
        train_pred_pd = train_predictions.select(TARGET_COLUMN, "prediction", "probability").toPandas()
        val_pred_pd = val_predictions.select(TARGET_COLUMN, "prediction", "probability").toPandas()
        test_pred_pd = test_predictions.select(TARGET_COLUMN, "prediction", "probability").toPandas()
        
        # Extract probability of positive class (class 1)
        train_pred_pd["prob"] = train_pred_pd["probability"].apply(lambda x: float(x[1]))
        val_pred_pd["prob"] = val_pred_pd["probability"].apply(lambda x: float(x[1]))
        test_pred_pd["prob"] = test_pred_pd["probability"].apply(lambda x: float(x[1]))
        
        # Calculate evaluation metrics
        train_metrics = self._calculate_metrics(
            train_pred_pd[TARGET_COLUMN], train_pred_pd["prob"])
        val_metrics = self._calculate_metrics(
            val_pred_pd[TARGET_COLUMN], val_pred_pd["prob"])
        test_metrics = self._calculate_metrics(
            test_pred_pd[TARGET_COLUMN], test_pred_pd["prob"])
        
        # Log metrics
        logger.info(f"Spark RF Training Metrics: {train_metrics}")
        logger.info(f"Spark RF Validation Metrics: {val_metrics}")
        logger.info(f"Spark RF Test Metrics: {test_metrics}")
        
        # Get feature importance
        feature_importance = rf_model.featureImportances.toArray()
        
        # Create result dictionary
        result = {
            "model": model,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "feature_importance": dict(zip(numerical_cols + categorical_cols, feature_importance)),
            "model_type": "spark_rf"
        }
        
        return result
    
    def _calculate_metrics(self, y_true, y_pred_proba) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Convert probabilities to binary predictions
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_pred_proba)
        }
        
        return metrics
    
    def save_model(self, result: Dict[str, Any], model_id: str = None) -> str:
        """
        Save the trained model and metadata.
        
        Args:
            result: Dictionary with model and evaluation results
            model_id: Optional model ID (if None, a timestamp will be used)
            
        Returns:
            Path to the saved model
        """
        # Generate model ID if not provided
        if model_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_id = f"{result['model_type']}_{timestamp}"
        
        # Create model directory
        model_dir = os.path.join(MODEL_PATH, model_id)
        create_directory_if_not_exists(model_dir)
        
        # Save model
        model_type = result["model_type"]
        
        if model_type == "xgboost":
            # Save XGBoost model
            model_path = os.path.join(model_dir, "model.json")
            result["model"].save_model(model_path)
            
            # Save preprocessor
            preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
            joblib.dump(result["preprocessor"], preprocessor_path)
            logger.info(f"Saved XGBoost model to {model_path}")
            
        elif model_type.startswith("spark_"):
            # Save Spark model
            model_path = os.path.join(model_dir, "spark_model")
            result["model"].write().overwrite().save(model_path)
            logger.info(f"Saved Spark model to {model_path}")
        
        # Extract metrics for metadata
        metadata = {
            "model_id": model_id,
            "model_type": model_type,
            "created_at": datetime.now().isoformat(),
            "train_metrics": result["train_metrics"],
            "val_metrics": result["val_metrics"],
            "test_metrics": result["test_metrics"],
            "is_best": False  # Default to not best model
        }
        
        # Add feature importance if available
        if "feature_importance" in result:
            metadata["feature_importance"] = result["feature_importance"]
        
        # Save metadata
        save_model_metadata(model_dir, metadata)
        
        logger.info(f"Saved model metadata to {model_dir}")
        
        return model_dir
    
    def train_sklearn_lr_model(self, data_dict: Dict[str, Any], feature_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train a scikit-learn Logistic Regression model.
        
        Args:
            data_dict: Dictionary with prepared data
            feature_info: Dictionary with feature information
            
        Returns:
            Dictionary with trained model and evaluation results
        """
        logger.info("Training scikit-learn Logistic Regression model")
        
        # Extract data
        X_train = data_dict["X_train"]
        y_train = data_dict["y_train"]
        X_val = data_dict["X_val"]
        y_val = data_dict["y_val"]
        X_test = data_dict["X_test"]
        y_test = data_dict["y_test"]
        
        # Create a preprocessor for the features
        categorical_features = feature_info["categorical_features"]
        numerical_features = feature_info["numerical_features"]
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', get_compatible_one_hot_encoder(), 
                 categorical_features)
            ],
            remainder='drop'
        )
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, random_state=MODEL_TRAINING_CONFIG["random_seed"]))
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred_proba = pipeline.predict_proba(X_train)[:, 1]
        y_val_pred_proba = pipeline.predict_proba(X_val)[:, 1]
        y_test_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate evaluation metrics
        train_metrics = self._calculate_metrics(y_train, y_train_pred_proba)
        val_metrics = self._calculate_metrics(y_val, y_val_pred_proba)
        test_metrics = self._calculate_metrics(y_test, y_test_pred_proba)
        
        # Log metrics
        logger.info(f"LogisticRegression Training Metrics: {train_metrics}")
        logger.info(f"LogisticRegression Validation Metrics: {val_metrics}")
        logger.info(f"LogisticRegression Test Metrics: {test_metrics}")
        
        # Get feature importance (coefficients for logistic regression)
        feature_importance = {}
        if hasattr(pipeline['classifier'], 'coef_'):
            features = preprocessor.get_feature_names_out()
            coefficients = pipeline['classifier'].coef_[0]
            feature_importance = dict(zip(features, abs(coefficients)))
        
        # Create result dictionary
        result = {
            "model": pipeline,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "feature_importance": feature_importance,
            "model_type": "sklearn_lr"
        }
        
        return result
        
    def train_sklearn_rf_model(self, data_dict: Dict[str, Any], feature_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train a scikit-learn Random Forest model.
        
        Args:
            data_dict: Dictionary with prepared data
            feature_info: Dictionary with feature information
            
        Returns:
            Dictionary with trained model and evaluation results
        """
        logger.info("Training scikit-learn Random Forest model")
        
        # Extract data
        X_train = data_dict["X_train"]
        y_train = data_dict["y_train"]
        X_val = data_dict["X_val"]
        y_val = data_dict["y_val"]
        X_test = data_dict["X_test"]
        y_test = data_dict["y_test"]
        
        # Create a preprocessor for the features
        categorical_features = feature_info["categorical_features"]
        numerical_features = feature_info["numerical_features"]
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', get_compatible_one_hot_encoder(), 
                 categorical_features)
            ],
            remainder='drop'
        )
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=MODEL_TRAINING_CONFIG["random_seed"]
            ))
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred_proba = pipeline.predict_proba(X_train)[:, 1]
        y_val_pred_proba = pipeline.predict_proba(X_val)[:, 1]
        y_test_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate evaluation metrics
        train_metrics = self._calculate_metrics(y_train, y_train_pred_proba)
        val_metrics = self._calculate_metrics(y_val, y_val_pred_proba)
        test_metrics = self._calculate_metrics(y_test, y_test_pred_proba)
        
        # Log metrics
        logger.info(f"RandomForest Training Metrics: {train_metrics}")
        logger.info(f"RandomForest Validation Metrics: {val_metrics}")
        logger.info(f"RandomForest Test Metrics: {test_metrics}")
        
        # Get feature importance
        feature_importance = {}
        if hasattr(pipeline['classifier'], 'feature_importances_'):
            features = preprocessor.get_feature_names_out()
            importances = pipeline['classifier'].feature_importances_
            feature_importance = dict(zip(features, importances))
        
        # Create result dictionary
        result = {
            "model": pipeline,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "feature_importance": feature_importance,
            "model_type": "sklearn_rf"
        }
        
        return result

    def run_training_pipeline(self, model_type: str = "xgboost", data_path: str = None) -> Dict[str, Any]:
        """
        Run the complete model training pipeline.
        
        Args:
            model_type: Type of model to train ("xgboost", "spark_lr", "spark_rf", "sklearn_lr", "sklearn_rf")
            data_path: Path to the training data CSV file (optional)
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Starting model training pipeline for {model_type}")
        
        # Load data
        df = self.load_training_data(data_path)
        
        # Prepare data for model training
        data_dict, feature_info = self.prepare_data(df)
        
        # Handle model choice based on availability
        if not SPARK_AVAILABLE and model_type in ["spark_lr", "spark_rf"]:
            logger.warning(f"PySpark is not available. Falling back to scikit-learn equivalent of {model_type}")
            if model_type == "spark_lr":
                model_type = "sklearn_lr"
            elif model_type == "spark_rf":
                model_type = "sklearn_rf"
        
        # Train model based on type
        if model_type == "xgboost":
            result = self.train_xgboost_model(data_dict, feature_info)
        elif model_type == "spark_lr" and SPARK_AVAILABLE:
            result = self.train_spark_lr_model(data_dict, feature_info)
        elif model_type == "spark_rf" and SPARK_AVAILABLE:
            result = self.train_spark_rf_model(data_dict, feature_info)
        elif model_type == "sklearn_lr":
            result = self.train_sklearn_lr_model(data_dict, feature_info)
        elif model_type == "sklearn_rf":
            result = self.train_sklearn_rf_model(data_dict, feature_info)
        else:
            # Default to XGBoost for unsupported model types
            logger.warning(f"Unsupported model type: {model_type}. Falling back to XGBoost.")
            result = self.train_xgboost_model(data_dict, feature_info)
        
        # Save model
        model_dir = self.save_model(result)
        
        # Add model path to result
        result["model_path"] = model_dir
        
        logger.info(f"Model training pipeline completed for {model_type}")
        
        return result

def train_model(model_type: str = "xgboost", data_path: str = None) -> Dict[str, Any]:
    """
    Convenience function to train a model.
    
    Args:
        model_type: Type of model to train
        data_path: Path to the training data
        
    Returns:
        Dictionary with training results
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create model trainer
    trainer = ModelTrainer()
    
    # Run training pipeline
    return trainer.run_training_pipeline(model_type, data_path)

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a risk score model")
    parser.add_argument("--model-type", type=str, default="xgboost",
                        choices=["xgboost", "spark_lr", "spark_rf"],
                        help="Type of model to train")
    parser.add_argument("--data-path", type=str, help="Path to training data CSV")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Train model
    result = train_model(args.model_type, args.data_path)
    
    # Print results
    print(f"Model trained and saved to: {result['model_path']}")
    print(f"Validation metrics: {result['val_metrics']}")