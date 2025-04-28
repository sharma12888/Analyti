import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType

from src.config import (
    DELTA_TABLES,
    SCHEMA_VALIDATION
)
from src.utils import (
    save_to_delta_table,
    read_delta_table
)

# Configure logging
logger = logging.getLogger(__name__)

class DataValidator:
    """
    Class for validating data quality
    """
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """
        Initialize data validator
        
        Args:
            spark: SparkSession object (optional)
        """
        self.spark = spark
    
    def check_schema_compliance(self, df: DataFrame, expected_schema: StructType) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if DataFrame complies with expected schema
        
        Args:
            df: DataFrame to validate
            expected_schema: Expected schema
            
        Returns:
            Tuple of (is_compliant, issues_dict)
        """
        try:
            # Get actual schema
            actual_schema = df.schema
            
            # Compare schema fields
            missing_fields = []
            type_mismatches = []
            
            for field in expected_schema:
                field_name = field.name
                field_type = field.dataType
                
                # Check if field exists in actual schema
                if not any(f.name == field_name for f in actual_schema):
                    missing_fields.append(field_name)
                else:
                    # Check if field type matches
                    actual_field = next(f for f in actual_schema if f.name == field_name)
                    if str(actual_field.dataType) != str(field_type):
                        type_mismatches.append({
                            "field": field_name,
                            "expected_type": str(field_type),
                            "actual_type": str(actual_field.dataType)
                        })
            
            is_compliant = len(missing_fields) == 0 and len(type_mismatches) == 0
            
            return is_compliant, {
                "missing_fields": missing_fields,
                "type_mismatches": type_mismatches
            }
        except Exception as e:
            logger.error(f"Error checking schema compliance: {e}")
            return False, {"error": str(e)}
    
    def check_nulls_and_missing(self, df: DataFrame) -> Tuple[bool, Dict[str, float]]:
        """
        Check for null and missing values in DataFrame
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_compliant, null_percentage_dict)
        """
        try:
            # Count total rows
            total_rows = df.count()
            if total_rows == 0:
                return True, {}
            
            # Check null counts for each column
            null_counts = {}
            null_percentages = {}
            
            for col_name in df.columns:
                # Get validation rules for this column
                validation_rules = SCHEMA_VALIDATION.get(col_name, {})
                nullable = validation_rules.get("nullable", True)
                
                # Count nulls
                null_count = df.filter(F.col(col_name).isNull()).count()
                null_counts[col_name] = null_count
                null_percentages[col_name] = (null_count / total_rows) * 100
                
                # Check if non-nullable field has nulls
                if not nullable and null_count > 0:
                    return False, null_percentages
            
            # Check if any column has more than 50% nulls (general rule)
            is_compliant = all(pct < 50 for pct in null_percentages.values())
            
            return is_compliant, null_percentages
        except Exception as e:
            logger.error(f"Error checking nulls and missing values: {e}")
            return False, {"error": str(e)}
    
    def check_data_ranges(self, df: DataFrame) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
        """
        Check for data values outside expected ranges
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_compliant, range_issues_dict)
        """
        try:
            range_issues = {}
            is_compliant = True
            
            for col_name in df.columns:
                # Get validation rules for this column
                validation_rules = SCHEMA_VALIDATION.get(col_name, {})
                
                # Skip if no range constraints defined
                if "min" not in validation_rules and "max" not in validation_rules:
                    continue
                
                # Get column statistics
                stats = df.select(
                    F.min(col_name).alias("min"),
                    F.max(col_name).alias("max"),
                    F.count(F.when(F.col(col_name).isNotNull(), 1)).alias("count")
                ).collect()[0].asDict()
                
                col_issues = {}
                
                # Check min value
                if "min" in validation_rules and stats["min"] is not None:
                    expected_min = validation_rules["min"]
                    actual_min = stats["min"]
                    
                    if actual_min < expected_min:
                        col_issues["min_violation"] = {
                            "expected_min": expected_min,
                            "actual_min": actual_min
                        }
                        is_compliant = False
                
                # Check max value
                if "max" in validation_rules and stats["max"] is not None:
                    expected_max = validation_rules["max"]
                    actual_max = stats["max"]
                    
                    if actual_max > expected_max:
                        col_issues["max_violation"] = {
                            "expected_max": expected_max,
                            "actual_max": actual_max
                        }
                        is_compliant = False
                
                # Add to issues dict if any issues found
                if col_issues:
                    range_issues[col_name] = col_issues
            
            return is_compliant, range_issues
        except Exception as e:
            logger.error(f"Error checking data ranges: {e}")
            return False, {"error": str(e)}
    
    def check_categorical_values(self, df: DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if categorical values are within expected sets
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_compliant, category_issues_dict)
        """
        try:
            category_issues = {}
            is_compliant = True
            
            for col_name in df.columns:
                # Get validation rules for this column
                validation_rules = SCHEMA_VALIDATION.get(col_name, {})
                
                # Skip if no allowed values defined
                if "allowed_values" not in validation_rules:
                    continue
                
                # Get unique values
                actual_values = set([row[0] for row in df.select(col_name).distinct().collect() if row[0] is not None])
                allowed_values = set(validation_rules["allowed_values"])
                
                # Find invalid values
                invalid_values = actual_values - allowed_values
                
                if invalid_values:
                    category_issues[col_name] = {
                        "allowed_values": list(allowed_values),
                        "invalid_values": list(invalid_values)
                    }
                    is_compliant = False
            
            return is_compliant, category_issues
        except Exception as e:
            logger.error(f"Error checking categorical values: {e}")
            return False, {"error": str(e)}
    
    def check_duplicate_records(self, df: DataFrame, key_columns: List[str]) -> Tuple[bool, Dict[str, Any]]:
        """
        Check for duplicate records based on key columns
        
        Args:
            df: DataFrame to validate
            key_columns: List of columns that should uniquely identify a record
            
        Returns:
            Tuple of (is_compliant, duplicate_info_dict)
        """
        try:
            # Count total rows
            total_rows = df.count()
            
            # Count distinct rows
            distinct_rows = df.select(*key_columns).distinct().count()
            
            # Calculate duplicate count
            duplicate_count = total_rows - distinct_rows
            duplicate_percentage = (duplicate_count / total_rows) * 100 if total_rows > 0 else 0
            
            is_compliant = duplicate_count == 0
            
            return is_compliant, {
                "total_rows": total_rows,
                "distinct_rows": distinct_rows,
                "duplicate_count": duplicate_count,
                "duplicate_percentage": duplicate_percentage
            }
        except Exception as e:
            logger.error(f"Error checking duplicate records: {e}")
            return False, {"error": str(e)}
    
    def generate_data_quality_report(self, df: DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive data quality report
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with data quality metrics
        """
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "row_count": df.count(),
                "column_count": len(df.columns),
                "columns": {}
            }
            
            # Analyze each column
            for col_name in df.columns:
                col_type = str(df.schema[col_name].dataType)
                
                # Basic statistics
                stats = df.select(
                    F.count(F.when(F.col(col_name).isNotNull(), 1)).alias("count"),
                    F.count(F.when(F.col(col_name).isNull(), 1)).alias("null_count")
                ).collect()[0].asDict()
                
                null_percentage = (stats["null_count"] / report["row_count"]) * 100 if report["row_count"] > 0 else 0
                
                col_report = {
                    "type": col_type,
                    "count": stats["count"],
                    "null_count": stats["null_count"],
                    "null_percentage": null_percentage
                }
                
                # Additional stats for numeric columns
                if col_type in ["IntegerType", "DoubleType", "LongType", "FloatType"]:
                    num_stats = df.select(
                        F.min(col_name).alias("min"),
                        F.max(col_name).alias("max"),
                        F.mean(col_name).alias("mean"),
                        F.stddev(col_name).alias("stddev")
                    ).collect()[0].asDict()
                    
                    col_report.update(num_stats)
                
                # Stats for string columns
                elif col_type == "StringType":
                    # Count distinct values
                    distinct_count = df.select(col_name).distinct().count()
                    col_report["distinct_count"] = distinct_count
                    
                    # Calculate cardinality ratio
                    cardinality_ratio = distinct_count / report["row_count"] if report["row_count"] > 0 else 0
                    col_report["cardinality_ratio"] = cardinality_ratio
                    
                    # Infer if categorical based on cardinality ratio
                    col_report["is_categorical"] = cardinality_ratio < 0.1
                
                report["columns"][col_name] = col_report
            
            return report
        except Exception as e:
            logger.error(f"Error generating data quality report: {e}")
            return {"error": str(e)}
    
    def validate_data_pipeline(self, df: DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Run all validation checks and determine if data can proceed through the pipeline
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (can_proceed, validation_results)
        """
        try:
            validation_results = {}
            
            # Check for nulls in required fields
            nulls_compliant, null_results = self.check_nulls_and_missing(df)
            validation_results["nulls_check"] = {
                "compliant": nulls_compliant,
                "details": null_results
            }
            
            # Check data ranges
            ranges_compliant, range_results = self.check_data_ranges(df)
            validation_results["ranges_check"] = {
                "compliant": ranges_compliant,
                "details": range_results
            }
            
            # Check categorical values
            categories_compliant, category_results = self.check_categorical_values(df)
            validation_results["categories_check"] = {
                "compliant": categories_compliant,
                "details": category_results
            }
            
            # Check for duplicates by application ID
            duplicates_compliant, duplicate_results = self.check_duplicate_records(df, ["application_id"])
            validation_results["duplicates_check"] = {
                "compliant": duplicates_compliant,
                "details": duplicate_results
            }
            
            # Determine if data can proceed
            can_proceed = all([
                nulls_compliant,
                ranges_compliant,
                categories_compliant,
                duplicates_compliant
            ])
            
            validation_results["can_proceed"] = can_proceed
            
            return can_proceed, validation_results
        except Exception as e:
            logger.error(f"Error validating data pipeline: {e}")
            return False, {"error": str(e)}
    
    def run_validation_on_delta_table(self, table_path: str) -> Dict[str, Any]:
        """
        Run validation on a Delta table
        
        Args:
            table_path: Path to the Delta table
            
        Returns:
            Validation report
        """
        try:
            # Read Delta table
            df = read_delta_table(self.spark, table_path)
            
            if df is None:
                return {
                    "can_proceed": False,
                    "validation_issues": "Delta table not found"
                }
            
            # Generate data quality report
            quality_report = self.generate_data_quality_report(df)
            
            # Run validation checks
            can_proceed, validation_results = self.validate_data_pipeline(df)
            
            # Save validated data if compliant
            if can_proceed:
                validated_table_path = DELTA_TABLES["validated_data"]
                save_to_delta_table(df, validated_table_path, mode="overwrite")
                logger.info(f"Saved validated data to {validated_table_path}")
            
            # Combine results
            full_report = {
                "timestamp": datetime.now().isoformat(),
                "table_path": table_path,
                "row_count": df.count(),
                "column_count": len(df.columns),
                "can_proceed": can_proceed,
                "quality_report": quality_report,
                "validation_issues": validation_results
            }
            
            return full_report
        except Exception as e:
            logger.error(f"Error running validation on Delta table: {e}")
            return {
                "can_proceed": False,
                "validation_issues": str(e)
            }