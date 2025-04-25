# Risk Score Analysis Pipeline

A comprehensive data pipeline for risk score analysis using PySpark, Delta Lake, and machine learning models. This project provides end-to-end capabilities for ingesting, processing, modeling, and scoring data to assess risk.

## Overview

This risk scoring system ingests loan application data, customer information, and loan performance data, applies various data processing and feature engineering techniques, and uses machine learning models to generate risk scores. The pipeline is built with scalability in mind, using PySpark for distributed processing and Delta Lake for efficient data storage.

## Features

- **Data Ingestion**: Import data from multiple sources including CSV files and PostgreSQL
- **Data Validation**: Validate data quality and completeness before processing
- **Feature Engineering**: Create and transform features for optimal model performance
- **Model Training**: Train and evaluate multiple ML models (Logistic Regression, Random Forest, XGBoost)
- **Model Evaluation**: Comprehensive model evaluation metrics and visualizations
- **Risk Scoring**: Generate risk scores and risk categories for loan applications
- **MLflow Integration**: Track experiments, metrics, and model artifacts
- **Delta Lake Storage**: Store data in reliable, performant Delta Lake format

## Project Structure

