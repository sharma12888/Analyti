import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Union, TypeVar

# Type for DataFrame (either pandas or Spark)
DataFrameType = TypeVar('DataFrameType')

import xgboost as xgb

from src.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    RISK_SCORE_RANGES,
    MODEL_PATH,
    DELTA_TABLES
)
from src.utils import (
    get_spark_session,
    read_delta_table,
    save_to_delta_table,
    load_model_metadata,
    SPARK_AVAILABLE
)

# Import required classes if PySpark is available
if SPARK_AVAILABLE:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.ml import PipelineModel
else:
    # Define DataFrame for type hinting when PySpark is not available
    class DataFrame:
        pass

from src.data_generator import generate_sample_data, SampleDataGenerator

logger = logging.getLogger(__name__)

class RiskScorer:
    """
    Class for generating risk scores using trained models.
    """
    
    def __init__(self, spark=None):
        """
        Initialize the risk scorer.
        
        Args:
            spark: SparkSession object (optional)
        """
        # Handle PySpark availability
        if SPARK_AVAILABLE:
            self.spark = spark or get_spark_session()
        else:
            self.spark = None
    
    def load_model(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a trained model.
        
        Args:
            model_path: Path to the model directory (if None, the best model will be used)
            
        Returns:
            Dictionary with loaded model and metadata
        """
        # If no model path is provided, find the best model
        if model_path is None:
            # Find all model directories
            if not os.path.exists(MODEL_PATH):
                raise ValueError(f"Model directory does not exist: {MODEL_PATH}")
            
            model_dirs = [os.path.join(MODEL_PATH, d) for d in os.listdir(MODEL_PATH) 
                         if os.path.isdir(os.path.join(MODEL_PATH, d))]
            
            if not model_dirs:
                raise ValueError(f"No models found in {MODEL_PATH}")
            
            # Find the best model (one with is_best=True)
            best_model_dir = None
            for model_dir in model_dirs:
                metadata = load_model_metadata(model_dir)
                if metadata and metadata.get("is_best", False):
                    best_model_dir = model_dir
                    break
            
            # If no best model is found, use the most recent one
            if best_model_dir is None:
                # Sort by creation time (newest first)
                model_dirs.sort(key=lambda d: os.path.getctime(d), reverse=True)
                best_model_dir = model_dirs[0]
            
            model_path = best_model_dir
            logger.info(f"Using model: {os.path.basename(model_path)}")
        
        # Load metadata
        metadata = load_model_metadata(model_path)
        if metadata is None:
            raise ValueError(f"No metadata found for model at {model_path}")
        
        model_type = metadata.get("model_type")
        if model_type not in ["xgboost", "spark_lr", "spark_rf", "sklearn_lr", "sklearn_rf"]:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Load model
        if model_type == "xgboost":
            # Load XGBoost model
            model_file = os.path.join(model_path, "model.json")
            preprocessor_file = os.path.join(model_path, "preprocessor.joblib")
            
            if not os.path.exists(model_file) or not os.path.exists(preprocessor_file):
                raise ValueError(f"Model files not found in {model_path}")
            
            model = xgb.Booster()
            model.load_model(model_file)
            preprocessor = joblib.load(preprocessor_file)
            
            logger.info(f"Loaded XGBoost model from {model_file}")
            
            result = {
                "model": model,
                "preprocessor": preprocessor,
                "model_type": model_type,
                "metadata": metadata
            }
        
        elif model_type.startswith("sklearn_"):
            # Load scikit-learn models (Random Forest, etc.)
            model_file = os.path.join(model_path, "model.joblib")
            preprocessor_file = os.path.join(model_path, "preprocessor.joblib")
            
            if not os.path.exists(model_file) or not os.path.exists(preprocessor_file):
                raise ValueError(f"Model files not found in {model_path}")
            
            model = joblib.load(model_file)
            preprocessor = joblib.load(preprocessor_file)
            
            logger.info(f"Loaded scikit-learn model from {model_file}")
            
            result = {
                "model": model,
                "preprocessor": preprocessor,
                "model_type": model_type,
                "metadata": metadata
            }
            
        elif model_type.startswith("spark_"):
            if not SPARK_AVAILABLE:
                # We can't load Spark models without Java/Spark
                logger.warning(f"Cannot load Spark model at {model_path} because PySpark is not available.")
                # Return metadata but no actual model
                result = {
                    "model": None,
                    "model_type": model_type,
                    "metadata": metadata,
                    "spark_unavailable": True
                }
            else:
                # Load Spark model
                model_dir = os.path.join(model_path, "spark_model")
                
                if not os.path.exists(model_dir):
                    raise ValueError(f"Spark model directory not found: {model_dir}")
                
                model = PipelineModel.load(model_dir)
                
                logger.info(f"Loaded Spark model from {model_dir}")
                
                result = {
                    "model": model,
                    "model_type": model_type,
                    "metadata": metadata
                }
        
        return result
    
    def score_dataframe(self, df: pd.DataFrame, model_info: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate risk scores for a pandas DataFrame.
        
        Args:
            df: Input DataFrame
            model_info: Dictionary with loaded model and metadata
            
        Returns:
            DataFrame with risk scores
        """
        # Make a copy to avoid modifying the original DataFrame
        data = df.copy()
        
        # Prepare feature columns
        feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        
        # Check if all required columns are present
        missing_cols = [col for col in feature_cols if col not in data.columns]
        if missing_cols:
            logger.warning(f"Missing columns in data: {missing_cols}")
            logger.warning(f"Available columns: {data.columns.tolist()}")
            raise ValueError(f"Required columns are missing in the data: {missing_cols}")
        
        # Handle missing values
        for col in NUMERICAL_FEATURES:
            if col in data.columns:
                data[col].fillna(data[col].median(), inplace=True)
        
        for col in CATEGORICAL_FEATURES:
            if col in data.columns:
                data[col].fillna(data[col].mode()[0], inplace=True)
        
        # Extract features
        X = data[feature_cols]
        
        # Generate predictions based on model type
        model_type = model_info["model_type"]
        
        if model_type == "xgboost":
            # Preprocess features
            preprocessor = model_info["preprocessor"]
            X_processed = preprocessor.transform(X)
            
            # Create DMatrix
            dmatrix = xgb.DMatrix(X_processed)
            
            # Predict probabilities
            probabilities = model_info["model"].predict(dmatrix)
            
        elif model_type.startswith("sklearn_"):
            # Preprocess features
            preprocessor = model_info["preprocessor"]
            X_processed = preprocessor.transform(X)
            
            # Predict probabilities
            probabilities = model_info["model"].predict_proba(X_processed)[:, 1]
            
        elif model_type.startswith("spark_"):
            if not SPARK_AVAILABLE:
                logger.warning("PySpark is not available. Falling back to random scoring.")
                # Generate random probabilities as a fallback
                import random
                probabilities = [random.uniform(0, 0.5) for _ in range(len(data))]
            else:
                # Convert to Spark DataFrame
                spark_df = self.spark.createDataFrame(data)
                
                # Make predictions
                predictions = model_info["model"].transform(spark_df)
                
                # Extract probabilities
                pred_df = predictions.select("*").toPandas()
                probabilities = pred_df["probability"].apply(lambda x: float(x[1]))
        
        # Add predictions to the DataFrame
        data["default_probability"] = probabilities
        
        # Convert probabilities to risk scores (scale from 0-100)
        data["risk_score"] = (data["default_probability"] * 100).round().astype(int)
        
        # Ensure risk score is in range 0-100
        data["risk_score"] = data["risk_score"].clip(0, 100)
        
        # Determine risk category based on score
        data["risk_category"] = "low"  # Default
        
        for category, (lower, upper) in RISK_SCORE_RANGES.items():
            mask = (data["risk_score"] >= lower) & (data["risk_score"] <= upper)
            data.loc[mask, "risk_category"] = category
        
        return data
    
    def score_spark_dataframe(self, df: DataFrameType, model_info: Dict[str, Any]) -> DataFrameType:
        """
        Generate risk scores for a Spark DataFrame.
        
        Args:
            df: Input Spark DataFrame
            model_info: Dictionary with loaded model and metadata
            
        Returns:
            Spark DataFrame with risk scores
        """
        if not SPARK_AVAILABLE:
            logger.warning("PySpark is not available. Cannot score Spark DataFrame.")
            return None
            
        # Extract model type
        model_type = model_info["model_type"]
        
        if model_type.startswith("spark_"):
            # Make predictions using the Spark model
            predictions = model_info["model"].transform(df)
            
            # Add risk score column (scale from 0-100)
            predictions = predictions.withColumn(
                "risk_score", 
                (predictions["probability"].getItem(1) * 100).cast("integer")
            )
            
            # Ensure risk score is in range 0-100
            predictions = predictions.withColumn(
                "risk_score",
                predictions.withColumn("risk_score", 
                                      predictions["risk_score"] < 0, 0),
                predictions.withColumn("risk_score",
                                      predictions["risk_score"] > 100, 100)
            )
            
            # Add risk category column based on score ranges
            for category, (lower, upper) in RISK_SCORE_RANGES.items():
                predictions = predictions.withColumn(
                    "risk_category",
                    (predictions["risk_score"] >= lower) & 
                    (predictions["risk_score"] <= upper),
                    category
                )
            
            return predictions
            
        else:
            # Convert Spark DataFrame to pandas
            pandas_df = df.toPandas()
            
            # Score using pandas method
            scored_df = self.score_dataframe(pandas_df, model_info)
            
            # Convert back to Spark DataFrame
            return self.spark.createDataFrame(scored_df)
    
    def score_manual_application(self, 
                              application_data: Dict[str, Any], 
                              model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a risk score for a single application.
        
        Args:
            application_data: Dictionary with application data
            model_info: Dictionary with loaded model and metadata
            
        Returns:
            Dictionary with risk score results
        """
        # Create a DataFrame with the application data
        df = pd.DataFrame([application_data])
        
        # Generate risk score
        scored_df = self.score_dataframe(df, model_info)
        
        # Extract risk score results
        result = {
            "application_id": application_data.get("application_id", "MANUAL"),
            "risk_score": int(scored_df["risk_score"].iloc[0]),
            "risk_category": scored_df["risk_category"].iloc[0],
            "default_probability": float(scored_df["default_probability"].iloc[0])
        }
        
        # Add risk reasons based on most important features
        risk_factors = []
        
        # Credit score is low
        if application_data.get("credit_score", 0) < 650:
            risk_factors.append("Low credit score")
        
        # High DTI ratio
        if application_data.get("debt_to_income_ratio", 0) > 0.4:
            risk_factors.append("High debt-to-income ratio")
        
        # Late payments
        late_30d = application_data.get("num_late_payments_30d", 0)
        late_60d = application_data.get("num_late_payments_60d", 0)
        late_90d = application_data.get("num_late_payments_90d", 0)
        
        if late_30d > 0 or late_60d > 0 or late_90d > 0:
            risk_factors.append(f"History of late payments (30d: {late_30d}, 60d: {late_60d}, 90d: {late_90d})")
        
        # High loan amount relative to income
        loan_amount = application_data.get("loan_amount", 0)
        annual_income = application_data.get("annual_income", 1)
        
        if loan_amount / annual_income > 0.5:
            risk_factors.append("High loan amount relative to income")
        
        # Employment status
        if application_data.get("employment_status") == "Unemployed":
            risk_factors.append("Unemployed status")
        
        # Add risk factors to result
        result["risk_reasons"] = risk_factors
        
        return result
    
    def run_scoring_pipeline(self, 
                           model_path: Optional[str] = None, 
                           output_table: Optional[str] = None,
                           score_date: Optional[str] = None,
                           data_source: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the risk scoring pipeline.
        
        Args:
            model_path: Path to the model to use (if None, use the best model)
            output_table: Path to the output Delta table (if None, use default)
            score_date: Date for scoring in format 'YYYY-MM-DD' (if None, use all data)
            data_source: Path to data source (if None, generate sample data)
            
        Returns:
            Dictionary with scoring results
        """
        # Load model
        model_info = self.load_model(model_path)
        model_id = os.path.basename(model_path) if model_path else "best_model"
        
        # Set output table
        if output_table is None:
            output_table = DELTA_TABLES["risk_scores"]
        
        # Load data
        if data_source and os.path.exists(data_source):
            # Load from file
            if data_source.endswith(".csv"):
                df = pd.read_csv(data_source)
                logger.info(f"Loaded {len(df)} records from {data_source}")
            else:
                # Assume Delta table
                df = read_delta_table(self.spark, data_source)
                if df is None:
                    raise ValueError(f"Failed to read Delta table: {data_source}")
                logger.info(f"Loaded data from Delta table: {data_source}")
        else:
            # Generate sample data
            logger.info("No data source provided. Generating sample data.")
            csv_path = generate_sample_data(num_records=500)
            df = pd.read_csv(csv_path)
            logger.info(f"Generated {len(df)} sample records")
        
        # Filter by date if specified
        if score_date and isinstance(df, pd.DataFrame) and "application_date" in df.columns:
            df["application_date"] = pd.to_datetime(df["application_date"])
            df = df[df["application_date"].dt.strftime("%Y-%m-%d") == score_date]
            logger.info(f"Filtered to {len(df)} records for date {score_date}")
        
        # Generate risk scores
        if isinstance(df, pd.DataFrame):
            scored_df = self.score_dataframe(df, model_info)
        else:
            scored_df = self.score_spark_dataframe(df, model_info)
        
        # Save results to Delta table if Spark is available
        if SPARK_AVAILABLE:
            if isinstance(scored_df, pd.DataFrame):
                # Add timestamp
                scored_df["scoring_timestamp"] = datetime.now()
                
                # Convert to Spark DataFrame
                spark_df = self.spark.createDataFrame(scored_df)
                
                # Save to Delta table
                save_to_delta_table(spark_df, output_table, mode="append")
            else:
                # Add timestamp
                scored_df = scored_df.withColumn("scoring_timestamp", 
                                               self.spark.sql("current_timestamp()"))
                
                # Save to Delta table
                save_to_delta_table(scored_df, output_table, mode="append")
        else:
            # Save to CSV if Spark is not available
            if isinstance(scored_df, pd.DataFrame):
                # Add timestamp
                scored_df["scoring_timestamp"] = datetime.now()
                
                # Save to CSV
                os.makedirs(os.path.dirname(output_table), exist_ok=True)
                csv_path = f"{output_table}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                scored_df.to_csv(csv_path, index=False)
                logger.info(f"Saved scores to CSV: {csv_path} (Delta tables not available without Spark)")
            else:
                logger.error("Cannot save Spark DataFrame when Spark is not available")
        
        # Count records by risk category
        if isinstance(scored_df, pd.DataFrame):
            risk_counts = scored_df["risk_category"].value_counts().to_dict()
            total_count = len(scored_df)
        else:
            # Spark DataFrame
            risk_counts_df = scored_df.groupBy("risk_category").count().toPandas()
            risk_counts = dict(zip(risk_counts_df["risk_category"], risk_counts_df["count"]))
            total_count = scored_df.count()
        
        # Create result summary
        result = {
            "status": "success",
            "model_used": model_id,
            "model_type": model_info["model_type"],
            "scored_count": total_count,
            "output_table": output_table,
            "risk_counts": risk_counts,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Scored {total_count} records using model {model_id}")
        
        return result

def generate_risk_score(application_data: Dict[str, Any], model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to generate a risk score for a single application.
    
    Args:
        application_data: Dictionary with application data
        model_path: Path to the model to use (if None, use the best model)
        
    Returns:
        Dictionary with risk score results
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create risk scorer
    scorer = RiskScorer()
    
    # Load model
    model_info = scorer.load_model(model_path)
    
    # Generate risk score
    return scorer.score_manual_application(application_data, model_info)

def run_scoring_pipeline(model_path: Optional[str] = None, 
                        output_table: Optional[str] = None,
                        score_date: Optional[str] = None,
                        data_source: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to run the risk scoring pipeline.
    
    Args:
        model_path: Path to the model to use (if None, use the best model)
        output_table: Path to the output Delta table (if None, use default)
        score_date: Date for scoring in format 'YYYY-MM-DD' (if None, use all data)
        data_source: Path to data source (if None, generate sample data)
        
    Returns:
        Dictionary with scoring results
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create risk scorer
    scorer = RiskScorer()
    
    # Run scoring pipeline
    return scorer.run_scoring_pipeline(model_path, output_table, score_date, data_source)

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate risk scores")
    parser.add_argument("--model-path", type=str, help="Path to the model to use")
    parser.add_argument("--output-table", type=str, help="Path to the output Delta table")
    parser.add_argument("--score-date", type=str, help="Date for scoring (YYYY-MM-DD)")
    parser.add_argument("--data-source", type=str, help="Path to data source")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run scoring pipeline
    result = run_scoring_pipeline(
        args.model_path, 
        args.output_table, 
        args.score_date, 
        args.data_source
    )
    
    # Print results
    print(f"Scoring complete. Scored {result['scored_count']} records.")