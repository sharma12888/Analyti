import os
import logging
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

from src.config import (
    DELTA_TABLES,
    MODEL_PATH,
    TARGET_COLUMN
)
from src.utils import (
    read_delta_table,
    load_model_metadata
)

# Configure logging
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Class for evaluating machine learning models
    """
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """
        Initialize the model evaluator
        
        Args:
            spark: SparkSession object (optional)
        """
        self.spark = spark
    
    def load_evaluation_data(self) -> DataFrame:
        """
        Load data for model evaluation
        
        Returns:
            DataFrame containing evaluation data
        """
        try:
            # Try to load test data first
            test_df = read_delta_table(self.spark, DELTA_TABLES["test_data"])
            
            if test_df is not None and not test_df.rdd.isEmpty():
                logger.info(f"Loaded {test_df.count()} records from test data")
                return test_df
            
            # If test data not available, try validation data
            val_df = read_delta_table(self.spark, DELTA_TABLES["validation_data"])
            
            if val_df is not None and not val_df.rdd.isEmpty():
                logger.info(f"Loaded {val_df.count()} records from validation data")
                return val_df
            
            # If validation data not available, use feature data directly
            feature_df = read_delta_table(self.spark, DELTA_TABLES["feature_table"])
            
            if feature_df is not None and not feature_df.rdd.isEmpty():
                # Use a small sample for evaluation
                sample_df = feature_df.sample(withReplacement=False, fraction=0.2, seed=42)
                logger.info(f"Loaded {sample_df.count()} records from feature data (sampled)")
                return sample_df
            
            logger.warning("No evaluation data available")
            return self.spark.createDataFrame([], [])
        except Exception as e:
            logger.error(f"Error loading evaluation data: {e}")
            raise
    
    def load_model_and_pipeline(self, model_path: str) -> Tuple[Any, Optional[PipelineModel]]:
        """
        Load a trained model and its associated pipeline
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            Tuple of (model, pipeline_model)
        """
        try:
            # Check if the path exists
            if not os.path.exists(model_path):
                logger.warning(f"Model path does not exist: {model_path}")
                return None, None
            
            # Load model metadata to determine model type
            metadata = load_model_metadata(model_path)
            model_type = metadata.get("model_type") if metadata else None
            
            if not model_type:
                # Try to infer model type from directory name
                model_type = os.path.basename(model_path).split("_")[0]
            
            # Load model based on type
            model = None
            
            if model_type in ["spark_lr", "spark_rf"]:
                from pyspark.ml.classification import LogisticRegressionModel, RandomForestClassificationModel
                
                model_file = os.path.join(model_path, "model")
                
                if os.path.exists(model_file):
                    if model_type == "spark_lr":
                        model = LogisticRegressionModel.load(model_file)
                    elif model_type == "spark_rf":
                        model = RandomForestClassificationModel.load(model_file)
            
            elif model_type == "xgboost":
                import xgboost as xgb
                
                model_file = os.path.join(model_path, "model.json")
                
                if os.path.exists(model_file):
                    model = xgb.Booster()
                    model.load_model(model_file)
            
            # Load pipeline model if available
            pipeline_model = None
            pipeline_file = os.path.join(model_path, "pipeline")
            
            if os.path.exists(pipeline_file):
                pipeline_model = PipelineModel.load(pipeline_file)
            
            if model:
                logger.info(f"Loaded model of type {model_type} from {model_path}")
            else:
                logger.warning(f"Failed to load model from {model_path}")
            
            return model, pipeline_model
        except Exception as e:
            logger.error(f"Error loading model and pipeline: {e}")
            return None, None
    
    def prepare_data_for_evaluation(
        self, 
        df: DataFrame, 
        pipeline_model: Optional[PipelineModel] = None
    ) -> DataFrame:
        """
        Prepare data for model evaluation
        
        Args:
            df: Input DataFrame
            pipeline_model: Optional pipeline model for feature preparation
            
        Returns:
            DataFrame ready for model evaluation
        """
        try:
            if df.rdd.isEmpty():
                logger.warning("Empty DataFrame provided for evaluation")
                return df
            
            # Apply pipeline model if provided
            if pipeline_model:
                prepared_df = pipeline_model.transform(df)
                logger.info("Applied pipeline model to prepare data for evaluation")
                return prepared_df
            
            # Check if data already has features column
            if "features" in df.columns:
                return df
            
            logger.warning("No pipeline model provided and no features column found")
            return df
        except Exception as e:
            logger.error(f"Error preparing data for evaluation: {e}")
            return df
    
    def evaluate_spark_model(
        self, 
        model, 
        df: DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate a Spark ML model
        
        Args:
            model: Trained Spark ML model
            df: DataFrame with features and target
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            if df.rdd.isEmpty() or TARGET_COLUMN not in df.columns:
                logger.warning("Empty DataFrame or missing target column")
                return {}
            
            # Make predictions
            predictions = model.transform(df)
            
            # Evaluate the model
            binary_evaluator = BinaryClassificationEvaluator(
                labelCol=TARGET_COLUMN,
                rawPredictionCol="rawPrediction",
                metricName="areaUnderROC"
            )
            
            auc = binary_evaluator.evaluate(predictions)
            
            # Calculate other metrics
            multi_evaluator = MulticlassClassificationEvaluator(
                labelCol=TARGET_COLUMN,
                predictionCol="prediction"
            )
            
            accuracy = multi_evaluator.setMetricName("accuracy").evaluate(predictions)
            precision = multi_evaluator.setMetricName("weightedPrecision").evaluate(predictions)
            recall = multi_evaluator.setMetricName("weightedRecall").evaluate(predictions)
            f1 = multi_evaluator.setMetricName("f1").evaluate(predictions)
            
            metrics = {
                "auc": auc,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
            
            logger.info(f"Spark model evaluation metrics: {metrics}")
            
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating Spark model: {e}")
            return {}
    
    def evaluate_sklearn_model(
        self, 
        model, 
        df: DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate a scikit-learn based model (including XGBoost)
        
        Args:
            model: Trained scikit-learn based model
            df: DataFrame with features and target
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            if df.rdd.isEmpty() or TARGET_COLUMN not in df.columns or "features" not in df.columns:
                logger.warning("Empty DataFrame, missing target column, or missing features")
                return {}
            
            # Convert to Pandas DataFrame
            pdf = df.select(TARGET_COLUMN, "features").toPandas()
            
            # Extract features from vector column
            from pyspark.ml.linalg import Vectors
            
            def extract_features(row):
                return Vectors.dense(row.features).toArray()
            
            X = np.array([extract_features(row) for _, row in pdf.iterrows()])
            y = pdf[TARGET_COLUMN].values
            
            # Generate predictions
            import xgboost as xgb
            
            if isinstance(model, xgb.Booster):
                dtest = xgb.DMatrix(X, label=y)
                y_pred_proba = model.predict(dtest)
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
                y_pred = model.predict(X)
            
            # Calculate metrics
            from sklearn.metrics import (
                roc_auc_score, 
                accuracy_score, 
                precision_score, 
                recall_score, 
                f1_score
            )
            
            metrics = {
                "accuracy": accuracy_score(y, y_pred),
                "precision": precision_score(y, y_pred),
                "recall": recall_score(y, y_pred),
                "f1": f1_score(y, y_pred)
            }
            
            if y_pred_proba is not None:
                metrics["auc"] = roc_auc_score(y, y_pred_proba)
            
            logger.info(f"Scikit-learn model evaluation metrics: {metrics}")
            
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating scikit-learn model: {e}")
            return {}
    
    def generate_confusion_matrix(
        self, 
        model, 
        df: DataFrame, 
        model_type: str
    ) -> np.ndarray:
        """
        Generate confusion matrix for model evaluation
        
        Args:
            model: Trained model
            df: DataFrame with features and target
            model_type: Type of model ('spark' or 'sklearn')
            
        Returns:
            Confusion matrix as numpy array
        """
        try:
            if df.rdd.isEmpty() or TARGET_COLUMN not in df.columns:
                logger.warning("Empty DataFrame or missing target column")
                return np.array([[0, 0], [0, 0]])
            
            if model_type == 'spark':
                # Make predictions
                predictions = model.transform(df)
                
                # Get actual and predicted labels
                pred_and_labels = predictions.select(
                    F.col("prediction").cast("double"), 
                    F.col(TARGET_COLUMN).cast("double")
                )
                
                # Convert to Pandas DataFrame
                pdf = pred_and_labels.toPandas()
                y_true = pdf[TARGET_COLUMN].values
                y_pred = pdf["prediction"].values
                
            elif model_type == 'sklearn':
                # Convert to Pandas DataFrame
                pdf = df.select(TARGET_COLUMN, "features").toPandas()
                
                # Extract features from vector column
                from pyspark.ml.linalg import Vectors
                
                def extract_features(row):
                    return Vectors.dense(row.features).toArray()
                
                X = np.array([extract_features(row) for _, row in pdf.iterrows()])
                y_true = pdf[TARGET_COLUMN].values
                
                # Generate predictions
                import xgboost as xgb
                
                if isinstance(model, xgb.Booster):
                    dtest = xgb.DMatrix(X, label=y_true)
                    y_pred_proba = model.predict(dtest)
                    y_pred = (y_pred_proba > 0.5).astype(int)
                else:
                    y_pred = model.predict(X)
            
            else:
                logger.warning(f"Unknown model type: {model_type}")
                return np.array([[0, 0], [0, 0]])
            
            # Generate confusion matrix
            from sklearn.metrics import confusion_matrix
            
            cm = confusion_matrix(y_true, y_pred)
            
            logger.info(f"Generated confusion matrix: {cm}")
            
            return cm
        except Exception as e:
            logger.error(f"Error generating confusion matrix: {e}")
            return np.array([[0, 0], [0, 0]])
    
    def evaluate_models(
        self, 
        model_paths: List[str], 
        data_df: Optional[DataFrame] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate multiple models and compare their performance
        
        Args:
            model_paths: List of paths to model directories
            data_df: Optional DataFrame for evaluation (if None, will load from Delta)
            
        Returns:
            Dictionary with evaluation results for each model
        """
        try:
            # Load evaluation data if not provided
            eval_df = data_df if data_df is not None else self.load_evaluation_data()
            
            if eval_df.rdd.isEmpty():
                logger.warning("No data available for model evaluation")
                return {}
            
            # Evaluate each model
            evaluation_results = {}
            
            for model_path in model_paths:
                # Load model and pipeline
                model, pipeline_model = self.load_model_and_pipeline(model_path)
                
                if model is None:
                    logger.warning(f"Failed to load model from {model_path}")
                    continue
                
                # Get model type
                metadata = load_model_metadata(model_path)
                model_type = metadata.get("model_type") if metadata else None
                
                if not model_type:
                    # Try to infer model type from directory name
                    model_type = os.path.basename(model_path).split("_")[0]
                
                # Prepare data for evaluation
                prepared_df = self.prepare_data_for_evaluation(eval_df, pipeline_model)
                
                # Evaluate model based on type
                metrics = {}
                
                if model_type in ["spark_lr", "spark_rf"]:
                    metrics = self.evaluate_spark_model(model, prepared_df)
                    confusion_matrix = self.generate_confusion_matrix(model, prepared_df, "spark")
                elif model_type == "xgboost":
                    metrics = self.evaluate_sklearn_model(model, prepared_df)
                    confusion_matrix = self.generate_confusion_matrix(model, prepared_df, "sklearn")
                
                # Store results
                if metrics:
                    model_name = os.path.basename(model_path)
                    
                    evaluation_results[model_name] = {
                        "model_type": model_type,
                        "metrics": metrics,
                        "confusion_matrix": confusion_matrix.tolist() if confusion_matrix is not None else None,
                        "path": model_path
                    }
            
            return evaluation_results
        except Exception as e:
            logger.error(f"Error evaluating models: {e}")
            return {}
    
    def find_best_model(self, model_dir: str = MODEL_PATH) -> Optional[str]:
        """
        Find the best performing model in the models directory
        
        Args:
            model_dir: Directory containing model subdirectories
            
        Returns:
            Path to the best model or None if no models found
        """
        try:
            # Get list of model directories
            if not os.path.exists(model_dir):
                logger.warning(f"Model directory does not exist: {model_dir}")
                return None
            
            model_paths = []
            for item in os.listdir(model_dir):
                item_path = os.path.join(model_dir, item)
                if os.path.isdir(item_path):
                    model_paths.append(item_path)
            
            if not model_paths:
                logger.warning(f"No models found in directory: {model_dir}")
                return None
            
            # Load metrics for each model
            models_metrics = []
            
            for model_path in model_paths:
                # Check for metrics file
                metrics_file = os.path.join(model_path, "metrics.json")
                
                if os.path.exists(metrics_file):
                    try:
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                        
                        # Add to list with path
                        models_metrics.append({
                            "path": model_path,
                            "metrics": metrics
                        })
                    except Exception as e:
                        logger.warning(f"Error loading metrics from {metrics_file}: {e}")
            
            if not models_metrics:
                logger.warning("No models with metrics found")
                return None
            
            # Find best model by AUC
            best_model = max(models_metrics, key=lambda m: m["metrics"].get("auc", 0))
            best_path = best_model["path"]
            
            logger.info(f"Found best model at path: {best_path} with AUC: {best_model['metrics'].get('auc')}")
            
            return best_path
        except Exception as e:
            logger.error(f"Error finding best model: {e}")
            return None
    
    def run_evaluation_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete model evaluation pipeline
        
        Returns:
            Dictionary with evaluation results
        """
        try:
            # Load evaluation data
            eval_df = self.load_evaluation_data()
            
            if eval_df.rdd.isEmpty():
                logger.warning("No data available for model evaluation")
                return {
                    "status": "error",
                    "message": "No data available for model evaluation"
                }
            
            # Get list of models to evaluate
            if not os.path.exists(MODEL_PATH):
                logger.warning(f"Model directory does not exist: {MODEL_PATH}")
                return {
                    "status": "error",
                    "message": f"Model directory does not exist: {MODEL_PATH}"
                }
            
            model_paths = []
            for item in os.listdir(MODEL_PATH):
                item_path = os.path.join(MODEL_PATH, item)
                if os.path.isdir(item_path):
                    model_paths.append(item_path)
            
            if not model_paths:
                logger.warning(f"No models found in directory: {MODEL_PATH}")
                return {
                    "status": "error",
                    "message": f"No models found in directory: {MODEL_PATH}"
                }
            
            # Evaluate models
            evaluation_results = self.evaluate_models(model_paths, eval_df)
            
            if not evaluation_results:
                logger.warning("No models could be evaluated")
                return {
                    "status": "error",
                    "message": "No models could be evaluated"
                }
            
            # Find best model
            best_model_name = max(
                evaluation_results.keys(), 
                key=lambda name: evaluation_results[name]["metrics"].get("auc", 0)
            )
            best_model = evaluation_results[best_model_name]
            
            logger.info(f"Best model: {best_model_name} with AUC: {best_model['metrics'].get('auc')}")
            
            return {
                "status": "success",
                "models": evaluation_results,
                "best_model": best_model
            }
        except Exception as e:
            logger.error(f"Error running evaluation pipeline: {e}")
            return {
                "status": "error",
                "message": str(e)
            }