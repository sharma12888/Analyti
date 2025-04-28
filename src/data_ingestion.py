import os
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, DateType

from src.config import (
    DELTA_TABLES,
    RAW_DATA_PATH,
    DATABASE_URL,
    CURRENT_DATE,
    YESTERDAY
)
from src.utils import (
    save_to_delta_table,
    read_delta_table,
    create_directory_if_not_exists
)

# Configure logging
logger = logging.getLogger(__name__)

# Define schemas for various data sources
LOAN_APPLICATION_SCHEMA = StructType([
    StructField("application_id", StringType(), False),
    StructField("customer_id", StringType(), False),
    StructField("loan_amount", DoubleType(), False),
    StructField("loan_term", IntegerType(), False),
    StructField("interest_rate", DoubleType(), False),
    StructField("application_date", DateType(), False),
    StructField("loan_purpose", StringType(), False),
    StructField("credit_score", IntegerType(), False),
    StructField("annual_income", DoubleType(), False),
    StructField("debt_to_income_ratio", DoubleType(), False),
    StructField("employment_status", StringType(), False),
    StructField("home_ownership", StringType(), False),
    StructField("verification_status", StringType(), False),
    StructField("application_type", StringType(), False),
    StructField("loan_grade", StringType(), False)
])

CUSTOMER_SCHEMA = StructType([
    StructField("customer_id", StringType(), False),
    StructField("age", IntegerType(), False),
    StructField("num_credit_lines", IntegerType(), False),
    StructField("utilization_rate", DoubleType(), False),
    StructField("num_late_payments_30d", IntegerType(), False),
    StructField("num_late_payments_60d", IntegerType(), False),
    StructField("num_late_payments_90d", IntegerType(), False)
])

LOAN_PERFORMANCE_SCHEMA = StructType([
    StructField("application_id", StringType(), False),
    StructField("customer_id", StringType(), False),
    StructField("default_flag", IntegerType(), True),
    StructField("last_payment_date", DateType(), True),
    StructField("days_past_due", IntegerType(), True)
])


class DataIngestionPipeline:
    """
    Pipeline for ingesting data from various sources
    """
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """
        Initialize the data ingestion pipeline
        
        Args:
            spark: SparkSession object (optional)
        """
        self.spark = spark
        
        # Create raw data directory if it doesn't exist
        create_directory_if_not_exists(RAW_DATA_PATH)
        
    def ingest_csv_data(self, file_path: str, schema: StructType) -> DataFrame:
        """
        Ingest data from CSV file
        
        Args:
            file_path: Path to the CSV file
            schema: Schema of the data
            
        Returns:
            Spark DataFrame with the loaded data
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"CSV file not found: {file_path}")
                # Return empty DataFrame with the specified schema
                return self.spark.createDataFrame([], schema)
            
            df = (
                self.spark.read.format("csv")
                .option("header", "true")
                .option("inferSchema", "false")
                .schema(schema)
                .load(file_path)
            )
            
            logger.info(f"Loaded {df.count()} rows from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error ingesting CSV data from {file_path}: {e}")
            raise
    
    def ingest_historical_data(self) -> Dict[str, DataFrame]:
        """
        Ingest historical data from various sources and combine into a single dataset
        
        Returns:
            Dictionary of DataFrames with the loaded data
        """
        try:
            # Define paths to historical data files
            loan_applications_path = os.path.join(RAW_DATA_PATH, "historical", "loan_applications.csv")
            customer_data_path = os.path.join(RAW_DATA_PATH, "historical", "customers.csv")
            loan_performance_path = os.path.join(RAW_DATA_PATH, "historical", "loan_performance.csv")
            
            # Load data from CSV files
            loan_applications_df = self.ingest_csv_data(loan_applications_path, LOAN_APPLICATION_SCHEMA)
            customer_df = self.ingest_csv_data(customer_data_path, CUSTOMER_SCHEMA)
            loan_performance_df = self.ingest_csv_data(loan_performance_path, LOAN_PERFORMANCE_SCHEMA)
            
            # Return dictionary of DataFrames
            return {
                "loan_applications": loan_applications_df,
                "customers": customer_df,
                "loan_performance": loan_performance_df
            }
        except Exception as e:
            logger.error(f"Error ingesting historical data: {e}")
            raise
    
    def ingest_postgresql_data(self) -> Dict[str, DataFrame]:
        """
        Ingest data from PostgreSQL
        
        Returns:
            Dictionary of DataFrames with the loaded data
        """
        try:
            if not DATABASE_URL:
                logger.warning("DATABASE_URL not set, skipping PostgreSQL ingestion")
                return {
                    "loan_applications": self.spark.createDataFrame([], LOAN_APPLICATION_SCHEMA),
                    "customers": self.spark.createDataFrame([], CUSTOMER_SCHEMA),
                    "loan_performance": self.spark.createDataFrame([], LOAN_PERFORMANCE_SCHEMA)
                }
            
            # JDBC connection properties
            jdbc_url = f"jdbc:postgresql://{DATABASE_URL.split('@')[1]}"
            connection_properties = {
                "user": os.environ.get("PGUSER"),
                "password": os.environ.get("PGPASSWORD"),
                "driver": "org.postgresql.Driver"
            }
            
            # Load data from PostgreSQL tables
            loan_applications_df = (
                self.spark.read.jdbc(
                    url=jdbc_url,
                    table="loan_applications",
                    properties=connection_properties
                )
            )
            
            customer_df = (
                self.spark.read.jdbc(
                    url=jdbc_url,
                    table="customers",
                    properties=connection_properties
                )
            )
            
            loan_performance_df = (
                self.spark.read.jdbc(
                    url=jdbc_url,
                    table="loan_performance",
                    properties=connection_properties
                )
            )
            
            # Return dictionary of DataFrames
            return {
                "loan_applications": loan_applications_df,
                "customers": customer_df,
                "loan_performance": loan_performance_df
            }
        except Exception as e:
            logger.error(f"Error ingesting PostgreSQL data: {e}")
            # Return empty DataFrames in case of error
            return {
                "loan_applications": self.spark.createDataFrame([], LOAN_APPLICATION_SCHEMA),
                "customers": self.spark.createDataFrame([], CUSTOMER_SCHEMA),
                "loan_performance": self.spark.createDataFrame([], LOAN_PERFORMANCE_SCHEMA)
            }
    
    def ingest_daily_data(self, date_str: Optional[str] = None) -> Dict[str, DataFrame]:
        """
        Ingest daily data updates
        
        Args:
            date_str: Date string in format 'YYYY-MM-DD' (defaults to yesterday)
            
        Returns:
            Dictionary of DataFrames with the loaded data
        """
        try:
            # Use yesterday's date if not specified
            date_str = date_str if date_str else YESTERDAY
            
            # Define paths to daily data files
            daily_dir = os.path.join(RAW_DATA_PATH, "daily", date_str)
            loan_applications_path = os.path.join(daily_dir, "loan_applications.csv")
            customer_data_path = os.path.join(daily_dir, "customers.csv")
            loan_performance_path = os.path.join(daily_dir, "loan_performance.csv")
            
            # Load data from CSV files
            loan_applications_df = self.ingest_csv_data(loan_applications_path, LOAN_APPLICATION_SCHEMA)
            customer_df = self.ingest_csv_data(customer_data_path, CUSTOMER_SCHEMA)
            loan_performance_df = self.ingest_csv_data(loan_performance_path, LOAN_PERFORMANCE_SCHEMA)
            
            # Return dictionary of DataFrames
            return {
                "loan_applications": loan_applications_df,
                "customers": customer_df,
                "loan_performance": loan_performance_df
            }
        except Exception as e:
            logger.error(f"Error ingesting daily data for {date_str}: {e}")
            # Return empty DataFrames in case of error
            return {
                "loan_applications": self.spark.createDataFrame([], LOAN_APPLICATION_SCHEMA),
                "customers": self.spark.createDataFrame([], CUSTOMER_SCHEMA),
                "loan_performance": self.spark.createDataFrame([], LOAN_PERFORMANCE_SCHEMA)
            }
    
    def combine_and_save_data(self, data_frames: Dict[str, DataFrame], mode: str = "append") -> DataFrame:
        """
        Combine data from various sources and save to Delta Lake
        
        Args:
            data_frames: Dictionary of DataFrames to combine
            mode: Write mode for Delta table (append or overwrite)
            
        Returns:
            Combined DataFrame
        """
        try:
            # Combine loan applications data
            loan_applications_combined = data_frames.get("loan_applications")
            if loan_applications_combined and not loan_applications_combined.rdd.isEmpty():
                loan_apps_table_path = os.path.join(DELTA_TABLES["raw_data"], "loan_applications")
                save_to_delta_table(loan_applications_combined, loan_apps_table_path, mode=mode)
                logger.info(f"Saved {loan_applications_combined.count()} loan applications to Delta table")
            
            # Combine customer data
            customers_combined = data_frames.get("customers")
            if customers_combined and not customers_combined.rdd.isEmpty():
                customers_table_path = os.path.join(DELTA_TABLES["raw_data"], "customers")
                save_to_delta_table(customers_combined, customers_table_path, mode=mode)
                logger.info(f"Saved {customers_combined.count()} customers to Delta table")
            
            # Combine loan performance data
            loan_performance_combined = data_frames.get("loan_performance")
            if loan_performance_combined and not loan_performance_combined.rdd.isEmpty():
                loan_perf_table_path = os.path.join(DELTA_TABLES["raw_data"], "loan_performance")
                save_to_delta_table(loan_performance_combined, loan_perf_table_path, mode=mode)
                logger.info(f"Saved {loan_performance_combined.count()} loan performance records to Delta table")
            
            # Return combined DataFrame for all data types
            # Join loan applications with customers and loan performance
            if (loan_applications_combined and not loan_applications_combined.rdd.isEmpty() and
                customers_combined and not customers_combined.rdd.isEmpty()):
                
                combined_df = loan_applications_combined.join(
                    customers_combined,
                    on="customer_id",
                    how="left"
                )
                
                if loan_performance_combined and not loan_performance_combined.rdd.isEmpty():
                    combined_df = combined_df.join(
                        loan_performance_combined.select("application_id", "default_flag"),
                        on="application_id",
                        how="left"
                    )
                
                return combined_df
            else:
                logger.warning("No data to combine")
                return self.spark.createDataFrame([], LOAN_APPLICATION_SCHEMA)
        except Exception as e:
            logger.error(f"Error combining and saving data: {e}")
            raise
    
    def run_ingestion_pipeline(self, 
                              ingest_historical: bool = False,
                              ingest_postgresql: bool = True,
                              ingest_daily: bool = True,
                              daily_date: Optional[str] = None) -> None:
        """
        Run the complete data ingestion pipeline
        
        Args:
            ingest_historical: Whether to ingest historical data
            ingest_postgresql: Whether to ingest data from PostgreSQL
            ingest_daily: Whether to ingest daily data
            daily_date: Specific date for daily ingestion
        """
        try:
            all_data_frames = {}
            
            # Historical data ingestion
            if ingest_historical:
                logger.info("Ingesting historical data")
                historical_data = self.ingest_historical_data()
                for data_type, df in historical_data.items():
                    all_data_frames[data_type] = df
            
            # PostgreSQL data ingestion
            if ingest_postgresql:
                logger.info("Ingesting PostgreSQL data")
                postgresql_data = self.ingest_postgresql_data()
                for data_type, df in postgresql_data.items():
                    # Append to existing data if available, otherwise use this data
                    if data_type in all_data_frames:
                        all_data_frames[data_type] = all_data_frames[data_type].union(df)
                    else:
                        all_data_frames[data_type] = df
            
            # Daily data ingestion
            if ingest_daily:
                logger.info(f"Ingesting daily data for {daily_date if daily_date else YESTERDAY}")
                daily_data = self.ingest_daily_data(daily_date)
                for data_type, df in daily_data.items():
                    # Append to existing data if available, otherwise use this data
                    if data_type in all_data_frames:
                        all_data_frames[data_type] = all_data_frames[data_type].union(df)
                    else:
                        all_data_frames[data_type] = df
            
            # Combine and save all data
            logger.info("Combining and saving all data")
            self.combine_and_save_data(all_data_frames)
            
            logger.info("Data ingestion pipeline completed successfully")
        except Exception as e:
            logger.error(f"Error running ingestion pipeline: {e}")
            raise