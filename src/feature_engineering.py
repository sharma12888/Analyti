import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import PipelineModel

from src.config import (
    DELTA_TABLES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    DATE_FEATURES,
    ID_FEATURES,
    TARGET_COLUMN
)
from src.utils import (
    save_to_delta_table,
    read_delta_table
)

# Configure logging
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Class for feature engineering and data preparation
    """
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """
        Initialize the feature engineer
        
        Args:
            spark: SparkSession object (optional)
        """
        self.spark = spark
    
    def join_datasets(self) -> DataFrame:
        """
        Join different datasets from Delta tables into a unified dataset
        
        Returns:
            Joined DataFrame
        """
        try:
            # Read validated data
            validated_data_path = DELTA_TABLES["validated_data"]
            df = read_delta_table(self.spark, validated_data_path)
            
            if df is None or df.rdd.isEmpty():
                # Try reading from raw data table
                raw_data_path = DELTA_TABLES["raw_data"]
                df = read_delta_table(self.spark, raw_data_path)
                
                if df is None or df.rdd.isEmpty():
                    logger.warning("No data found in validated or raw data tables")
                    return self.spark.createDataFrame([], StructType([]))
            
            return df
        except Exception as e:
            logger.error(f"Error joining datasets: {e}")
            raise
    
    def handle_missing_values(self, df: DataFrame) -> DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        try:
            if df.rdd.isEmpty():
                return df
            
            # Fill missing values for numerical features
            num_filled_df = df
            
            for col in NUMERICAL_FEATURES:
                if col in df.columns:
                    # Fill with median value
                    median_value = df.approxQuantile(col, [0.5], 0.25)[0]
                    num_filled_df = num_filled_df.withColumn(
                        col,
                        F.when(F.col(col).isNull(), median_value).otherwise(F.col(col))
                    )
            
            # Fill missing values for categorical features
            cat_filled_df = num_filled_df
            
            for col in CATEGORICAL_FEATURES:
                if col in df.columns:
                    # Fill with mode (most frequent value)
                    mode_value = (
                        df.groupBy(col)
                        .count()
                        .orderBy(F.desc("count"))
                        .filter(F.col(col).isNotNull())
                        .first()
                    )
                    
                    mode_value = mode_value[col] if mode_value else "Unknown"
                    
                    cat_filled_df = cat_filled_df.withColumn(
                        col,
                        F.when(F.col(col).isNull(), mode_value).otherwise(F.col(col))
                    )
            
            return cat_filled_df
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            raise
    
    def create_date_features(self, df: DataFrame) -> DataFrame:
        """
        Create features from date columns
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new date-based features
        """
        try:
            if df.rdd.isEmpty() or not any(col in df.columns for col in DATE_FEATURES):
                return df
            
            date_df = df
            
            # Process each date column
            for date_col in DATE_FEATURES:
                if date_col in df.columns:
                    # Extract components from date
                    date_df = date_df.withColumn(
                        f"{date_col}_year",
                        F.year(F.col(date_col))
                    ).withColumn(
                        f"{date_col}_month",
                        F.month(F.col(date_col))
                    ).withColumn(
                        f"{date_col}_day",
                        F.dayofmonth(F.col(date_col))
                    ).withColumn(
                        f"{date_col}_quarter",
                        F.quarter(F.col(date_col))
                    ).withColumn(
                        f"{date_col}_dayofweek",
                        F.dayofweek(F.col(date_col))
                    )
            
            return date_df
        except Exception as e:
            logger.error(f"Error creating date features: {e}")
            raise
    
    def create_numerical_features(self, df: DataFrame) -> DataFrame:
        """
        Create additional numerical features and transformations
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new numerical features
        """
        try:
            if df.rdd.isEmpty():
                return df
            
            # Create derived numerical features
            transformed_df = df
            
            # Create debt to income ratio buckets
            if "debt_to_income_ratio" in df.columns:
                transformed_df = transformed_df.withColumn(
                    "dti_bucket",
                    F.when(F.col("debt_to_income_ratio") <= 10, "very_low")
                    .when(F.col("debt_to_income_ratio") <= 20, "low")
                    .when(F.col("debt_to_income_ratio") <= 30, "medium")
                    .when(F.col("debt_to_income_ratio") <= 40, "high")
                    .otherwise("very_high")
                )
            
            # Create age buckets
            if "age" in df.columns:
                def age_category(age):
                    if age < 25:
                        return "young"
                    elif age < 35:
                        return "young_adult"
                    elif age < 45:
                        return "adult"
                    elif age < 55:
                        return "middle_age"
                    elif age < 65:
                        return "senior"
                    else:
                        return "retired"
                
                age_category_udf = F.udf(age_category)
                transformed_df = transformed_df.withColumn(
                    "age_category",
                    age_category_udf(F.col("age"))
                )
            
            # Create credit score buckets
            if "credit_score" in df.columns:
                def credit_category(score):
                    if score < 580:
                        return "very_poor"
                    elif score < 670:
                        return "fair"
                    elif score < 740:
                        return "good"
                    elif score < 800:
                        return "very_good"
                    else:
                        return "excellent"
                
                credit_category_udf = F.udf(credit_category)
                transformed_df = transformed_df.withColumn(
                    "credit_category",
                    credit_category_udf(F.col("credit_score"))
                )
            
            return transformed_df
        except Exception as e:
            logger.error(f"Error creating numerical features: {e}")
            raise
    
    def create_categorical_features(self, df: DataFrame) -> DataFrame:
        """
        Create additional categorical features and transformations
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new categorical features
        """
        try:
            if df.rdd.isEmpty():
                return df
            
            transformed_df = df
            
            # Combine late payment features if available
            late_payment_cols = [
                "num_late_payments_30d",
                "num_late_payments_60d",
                "num_late_payments_90d"
            ]
            
            if all(col in df.columns for col in late_payment_cols):
                transformed_df = transformed_df.withColumn(
                    "has_late_payments",
                    (F.col("num_late_payments_30d") > 0) | 
                    (F.col("num_late_payments_60d") > 0) | 
                    (F.col("num_late_payments_90d") > 0)
                )
                
                transformed_df = transformed_df.withColumn(
                    "late_payment_severity",
                    F.when(
                        F.col("num_late_payments_90d") > 0, "severe"
                    ).when(
                        F.col("num_late_payments_60d") > 0, "moderate"
                    ).when(
                        F.col("num_late_payments_30d") > 0, "mild"
                    ).otherwise("none")
                )
            
            # Create a loan size category
            if "loan_amount" in df.columns:
                transformed_df = transformed_df.withColumn(
                    "loan_size_category",
                    F.when(F.col("loan_amount") <= 5000, "micro")
                    .when(F.col("loan_amount") <= 15000, "small")
                    .when(F.col("loan_amount") <= 50000, "medium")
                    .when(F.col("loan_amount") <= 100000, "large")
                    .otherwise("very_large")
                )
            
            return transformed_df
        except Exception as e:
            logger.error(f"Error creating categorical features: {e}")
            raise
    
    def prepare_features_for_modeling(
        self, 
        df: DataFrame, 
        is_training: bool = True
    ) -> Tuple[DataFrame, PipelineModel]:
        """
        Prepare features for modeling, including encoding, scaling, etc.
        
        Args:
            df: Input DataFrame with raw features
            is_training: Whether the DataFrame is for training (True) or inference (False)
            
        Returns:
            Tuple of (DataFrame with prepared features, Pipeline model)
        """
        try:
            if df.rdd.isEmpty():
                return df, None
            
            all_feature_cols = []
            pipeline_stages = []
            
            # Process categorical features
            cat_feature_output_cols = []
            
            for cat_col in CATEGORICAL_FEATURES:
                if cat_col in df.columns:
                    # Create string indexer for the categorical column
                    indexer = StringIndexer(
                        inputCol=cat_col,
                        outputCol=f"{cat_col}_index",
                        handleInvalid="keep"
                    )
                    pipeline_stages.append(indexer)
                    
                    # Create one-hot encoder for the indexed column
                    encoder = OneHotEncoder(
                        inputCol=f"{cat_col}_index",
                        outputCol=f"{cat_col}_vec"
                    )
                    pipeline_stages.append(encoder)
                    
                    cat_feature_output_cols.append(f"{cat_col}_vec")
            
            # Process numerical features
            num_feature_output_cols = []
            
            for num_col in NUMERICAL_FEATURES:
                if num_col in df.columns:
                    # Numerical features just need to be added to the feature vector
                    num_feature_output_cols.append(num_col)
            
            # Add any additional features generated earlier
            extra_numerical_cols = [
                col for col in df.columns 
                if col.endswith("_year") or 
                   col.endswith("_month") or 
                   col.endswith("_day") or 
                   col.endswith("_quarter") or 
                   col.endswith("_dayofweek")
            ]
            
            num_feature_output_cols.extend(extra_numerical_cols)
            
            # Combine all features into a single vector
            all_feature_cols = cat_feature_output_cols + num_feature_output_cols
            
            # Create vector assembler
            assembler = VectorAssembler(
                inputCols=all_feature_cols,
                outputCol="features_raw",
                handleInvalid="keep"
            )
            pipeline_stages.append(assembler)
            
            # Scale numerical features
            scaler = StandardScaler(
                inputCol="features_raw",
                outputCol="features",
                withStd=True,
                withMean=True
            )
            pipeline_stages.append(scaler)
            
            # Create and fit the pipeline
            pipeline = Pipeline(stages=pipeline_stages)
            
            if is_training:
                # Fit the pipeline on the training data
                pipeline_model = pipeline.fit(df)
                
                # Transform the data using the fitted pipeline
                prepared_df = pipeline_model.transform(df)
                
                return prepared_df, pipeline_model
            else:
                # Use the provided pipeline model to transform the data
                logger.warning("No pipeline model provided for inference")
                return df, None
        except Exception as e:
            logger.error(f"Error preparing features for modeling: {e}")
            raise
    
    def analyze_feature_importance(self, df: DataFrame) -> Dict[str, float]:
        """
        Analyze feature importance using correlation with target
        
        Args:
            df: Input DataFrame with features and target
            
        Returns:
            Dictionary of feature importance scores
        """
        try:
            if df.rdd.isEmpty() or TARGET_COLUMN not in df.columns:
                return {}
            
            feature_importance = {}
            
            # Analyze importance for numerical features
            for num_col in NUMERICAL_FEATURES:
                if num_col in df.columns:
                    # Calculate correlation with target
                    correlation = df.stat.corr(num_col, TARGET_COLUMN)
                    feature_importance[num_col] = abs(correlation)
            
            # Analyze importance for categorical features
            for cat_col in CATEGORICAL_FEATURES:
                if cat_col in df.columns:
                    # Calculate Cramer's V (approximation through chi-square)
                    # Not implemented here for simplicity
                    feature_importance[cat_col] = 0.0
            
            return feature_importance
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {e}")
            return {}
    
    def run_feature_engineering_pipeline(self, input_df: Optional[DataFrame] = None) -> DataFrame:
        """
        Run the complete feature engineering pipeline
        
        Args:
            input_df: Optional input DataFrame (if None, will join from Delta tables)
            
        Returns:
            DataFrame with engineered features
        """
        try:
            # Get input data
            df = input_df if input_df is not None else self.join_datasets()
            
            if df.rdd.isEmpty():
                logger.warning("No data available for feature engineering")
                return df
            
            # Handle missing values
            df = self.handle_missing_values(df)
            
            # Create date-based features
            df = self.create_date_features(df)
            
            # Create additional numerical features
            df = self.create_numerical_features(df)
            
            # Create additional categorical features
            df = self.create_categorical_features(df)
            
            # Prepare features for modeling
            prepared_df, pipeline_model = self.prepare_features_for_modeling(df)
            
            # Save engineered features to Delta table
            feature_table_path = DELTA_TABLES["feature_table"]
            save_to_delta_table(prepared_df, feature_table_path)
            
            logger.info(f"Saved {prepared_df.count()} records to feature table")
            
            return prepared_df
        except Exception as e:
            logger.error(f"Error running feature engineering pipeline: {e}")
            raise