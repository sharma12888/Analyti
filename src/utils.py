import os
import json
import logging
from typing import Dict, Any, Optional, List, Tuple, Union, TypeVar

import numpy as np
import pandas as pd

# Check if Java is available before importing PySpark
SPARK_AVAILABLE = False
try:
    # First check if JAVA_HOME is set
    java_home = os.environ.get('JAVA_HOME')
    if not java_home:
        logging.warning("JAVA_HOME is not set, PySpark functionality will be disabled")
    else:
        from pyspark.sql import SparkSession, DataFrame
        from pyspark.sql import functions as F
        from pyspark.sql.types import StructType
        SPARK_AVAILABLE = True
except Exception as e:
    logging.warning(f"Error importing PySpark: {e}")
    logging.warning("PySpark functionality will be disabled")
    
    # Define DataFrame for type hinting when PySpark is not available
    class DataFrame:
        pass

from src.config import (
    SPARK_CONFIG,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    ROOT_DIR,
    DATA_DIR
)

# Configure logging
logger = logging.getLogger(__name__)

def get_spark_session():
    """
    Create or get a SparkSession with Delta Lake support.
    
    Returns:
        SparkSession object or None if Spark is not available
    """
    if not SPARK_AVAILABLE:
        logger.warning("Cannot create SparkSession because PySpark is not available")
        return None
        
    # Create SparkSession with Delta Lake support
    builder = SparkSession.builder.appName(SPARK_CONFIG["app_name"]) \
        .master(SPARK_CONFIG["master"])
    
    # Add configs
    for key, value in SPARK_CONFIG["config"].items():
        builder = builder.config(key, value)
    
    # Get or create session
    spark = builder.getOrCreate()
    
    logger.info(f"Created SparkSession with app name: {SPARK_CONFIG['app_name']}")
    
    return spark

def get_or_create_spark():
    """
    Alias for get_spark_session() for backward compatibility.
    
    Returns:
        SparkSession object or None if Spark is not available
    """
    return get_spark_session()

def create_directory_if_not_exists(directory_path: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory_path: Path to the directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        logger.info(f"Created directory: {directory_path}")

def setup_mlflow() -> None:
    """
    Set up MLflow tracking.
    """
    try:
        import mlflow
        
        # Create MLflow directory if it doesn't exist
        create_directory_if_not_exists(os.path.dirname(MLFLOW_TRACKING_URI.replace("file://", "")))
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Set experiment
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        
        logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")
        logger.info(f"MLflow experiment set to: {MLFLOW_EXPERIMENT_NAME}")
    except ImportError:
        logger.warning("MLflow not available, skipping MLflow setup")
    except Exception as e:
        logger.error(f"Error setting up MLflow: {e}")

def save_to_delta_table(df, table_path: str, mode: str = "overwrite") -> None:
    """
    Save a DataFrame to a Delta table.
    
    Args:
        df: DataFrame to save (Spark DataFrame)
        table_path: Path to the Delta table
        mode: Write mode (overwrite or append)
    """
    if not SPARK_AVAILABLE:
        # If Spark is not available, save as CSV instead
        if isinstance(df, pd.DataFrame):
            csv_path = f"{table_path}.csv"
            create_directory_if_not_exists(os.path.dirname(csv_path))
            df.to_csv(csv_path, index=False)
            logger.info(f"Spark not available. Saved DataFrame with {len(df)} rows to CSV at: {csv_path}")
        else:
            logger.error("Cannot save non-pandas DataFrame when Spark is not available")
        return
        
    try:
        # Create directory if not exists
        create_directory_if_not_exists(os.path.dirname(table_path))
        
        # Save to Delta table
        df.write.format("delta").mode(mode).save(table_path)
        
        logger.info(f"Saved DataFrame with {df.count()} rows to Delta table at: {table_path}")
    except Exception as e:
        logger.error(f"Error saving to Delta table at {table_path}: {e}")
        raise

# Create a type for DataFrame (either pandas or Spark)
DataFrameType = TypeVar('DataFrameType')

def read_delta_table(spark, table_path: str) -> Optional[Any]:
    """
    Read a Delta table into a DataFrame.
    
    Args:
        spark: SparkSession (can be None if Spark is not available)
        table_path: Path to the Delta table
        
    Returns:
        DataFrame or None if table doesn't exist or if Spark is not available
    """
    if not SPARK_AVAILABLE:
        # Try to read as CSV if Spark is not available
        csv_path = f"{table_path}.csv"
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                logger.info(f"Spark not available. Read {len(df)} rows from CSV at: {csv_path}")
                return df
            except Exception as e:
                logger.error(f"Error reading from CSV at {csv_path}: {e}")
                return None
        else:
            logger.warning(f"Neither Delta table nor CSV exists at path: {table_path}")
            return None
        
    try:
        if not os.path.exists(table_path):
            logger.warning(f"Delta table does not exist at path: {table_path}")
            return None
        
        # Read from Delta table
        df = spark.read.format("delta").load(table_path)
        
        logger.info(f"Read {df.count()} rows from Delta table at: {table_path}")
        
        return df
    except Exception as e:
        logger.error(f"Error reading from Delta table at {table_path}: {e}")
        return None

def save_model_metadata(model_path: str, metadata: Dict[str, Any]) -> None:
    """
    Save model metadata to a JSON file.
    
    Args:
        model_path: Path to the model directory
        metadata: Dictionary of metadata
    """
    try:
        # Create directory if not exists
        create_directory_if_not_exists(model_path)
        
        # Convert numpy/pandas types to Python types
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj.to_dict()
            elif isinstance(obj, (set, frozenset)):
                return list(obj)
            else:
                return obj
        
        # Convert metadata to serializable format
        serializable_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, dict):
                serializable_metadata[key] = {k: convert_to_serializable(v) for k, v in value.items()}
            else:
                serializable_metadata[key] = convert_to_serializable(value)
        
        # Save metadata to JSON file
        metadata_path = os.path.join(model_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(serializable_metadata, f, indent=2)
        
        logger.info(f"Saved model metadata to: {metadata_path}")
    except Exception as e:
        logger.error(f"Error saving model metadata: {e}")

def load_model_metadata(model_path: str) -> Optional[Dict[str, Any]]:
    """
    Load model metadata from a JSON file.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Dictionary of metadata or None if file doesn't exist
    """
    try:
        metadata_path = os.path.join(model_path, "metadata.json")
        
        if not os.path.exists(metadata_path):
            logger.warning(f"Model metadata file does not exist at path: {metadata_path}")
            return None
        
        # Load metadata from JSON file
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded model metadata from: {metadata_path}")
        
        return metadata
    except Exception as e:
        logger.error(f"Error loading model metadata: {e}")
        return None

def json_serial(obj):
    """
    Helper function for JSON serialization of objects like datetime.
    """
    if isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    raise TypeError(f"Type {type(obj)} not serializable")