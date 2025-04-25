#!/usr/bin/env python3
"""
Demo script for the Risk Score Analysis System.
This script generates sample data, trains models, and runs the risk scoring pipeline.
"""

import os
import logging
import argparse
import pandas as pd
from datetime import datetime

from src.data_generator import generate_sample_data, SampleDataGenerator
from src.model_training import train_model
from src.risk_scoring import generate_risk_score, run_scoring_pipeline
from src.config import MODEL_PATH, DELTA_TABLES
from src.utils import create_directory_if_not_exists

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories for the demo."""
    # Create model directory
    create_directory_if_not_exists(MODEL_PATH)
    
    # Create data directories
    for path in DELTA_TABLES.values():
        create_directory_if_not_exists(os.path.dirname(path))
    
    logger.info("Directory setup complete")

def generate_data(num_records=1000, output_path=None):
    """Generate sample data for the demo."""
    logger.info(f"Generating {num_records} sample records")
    
    if output_path:
        # Generate and save to specific path
        data_path = generate_sample_data(num_records=num_records, output_file=output_path)
    else:
        # Use default path
        data_path = generate_sample_data(num_records=num_records)
    
    logger.info(f"Sample data saved to: {data_path}")
    return data_path

def train_demo_models(data_path, model_types=None):
    """Train demo models."""
    if model_types is None:
        model_types = ["xgboost"]  # Default to XGBoost only
    
    logger.info(f"Training models: {', '.join(model_types)}")
    
    model_paths = {}
    
    for model_type in model_types:
        try:
            logger.info(f"Training {model_type} model")
            result = train_model(model_type=model_type, data_path=data_path)
            model_path = result.get("model_path")
            model_paths[model_type] = model_path
            logger.info(f"{model_type} model trained and saved to: {model_path}")
            
            # Log validation metrics
            val_metrics = result.get("val_metrics", {})
            logger.info(f"Validation metrics for {model_type}:")
            for metric, value in val_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
                
        except Exception as e:
            logger.error(f"Error training {model_type} model: {e}")
    
    return model_paths

def run_demo_scoring(model_path=None, data_path=None):
    """Run demo risk scoring."""
    logger.info("Running risk scoring pipeline")
    
    try:
        result = run_scoring_pipeline(
            model_path=model_path,
            data_source=data_path
        )
        
        logger.info(f"Scoring complete. Scored {result['scored_count']} records.")
        logger.info(f"Risk counts: {result['risk_counts']}")
        
        return result
    except Exception as e:
        logger.error(f"Error running scoring pipeline: {e}")
        return None

def generate_manual_score():
    """Generate a risk score for a manually entered application."""
    logger.info("Generating risk score for sample application")
    
    # Sample application data
    application_data = {
        "loan_amount": 25000,
        "annual_income": 85000,
        "debt_to_income_ratio": 0.35,
        "monthly_debt": 2000,
        "credit_score": 720,
        "num_credit_inquiries": 2,
        "num_late_payments_30d": 0,
        "num_late_payments_60d": 0,
        "num_late_payments_90d": 0,
        "credit_line_age": 60,
        "loan_term": 36,
        "interest_rate": 7.5,
        "employment_status": "Employed",
        "loan_purpose": "Debt consolidation",
        "home_ownership": "Mortgage",
        "application_type": "Individual"
    }
    
    try:
        result = generate_risk_score(application_data)
        
        logger.info(f"Manual scoring complete.")
        logger.info(f"Risk score: {result['risk_score']}")
        logger.info(f"Risk category: {result['risk_category']}")
        logger.info(f"Risk factors: {result['risk_reasons']}")
        
        return result
    except Exception as e:
        logger.error(f"Error generating manual risk score: {e}")
        return None

def run_full_demo():
    """Run the full demo pipeline."""
    logger.info("Starting full demo pipeline")
    
    # Setup directories
    setup_directories()
    
    # Generate sample data
    data_path = generate_data(num_records=1000)
    
    # Train models
    model_paths = train_demo_models(data_path, model_types=["xgboost"])
    
    # Get the first model path
    model_path = next(iter(model_paths.values()), None)
    
    # Run scoring pipeline
    scoring_result = run_demo_scoring(model_path, data_path)
    
    # Generate manual score
    manual_result = generate_manual_score()
    
    logger.info("Full demo pipeline complete")
    
    return {
        "data_path": data_path,
        "model_paths": model_paths,
        "scoring_result": scoring_result,
        "manual_result": manual_result
    }

def save_demo_results(results, output_dir="./demo_results"):
    """Save demo results to files."""
    create_directory_if_not_exists(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save scoring result
    if results.get("scoring_result"):
        scoring_file = os.path.join(output_dir, f"scoring_result_{timestamp}.txt")
        with open(scoring_file, "w") as f:
            for key, value in results["scoring_result"].items():
                f.write(f"{key}: {value}\n")
    
    # Save manual result
    if results.get("manual_result"):
        manual_file = os.path.join(output_dir, f"manual_result_{timestamp}.txt")
        with open(manual_file, "w") as f:
            for key, value in results["manual_result"].items():
                f.write(f"{key}: {value}\n")
    
    logger.info(f"Demo results saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Risk Score Analysis demo")
    parser.add_argument("--data-only", action="store_true", help="Generate sample data only")
    parser.add_argument("--train-only", action="store_true", help="Train models only (requires data)")
    parser.add_argument("--score-only", action="store_true", help="Run scoring only (requires model and data)")
    parser.add_argument("--manual-only", action="store_true", help="Generate manual score only (requires model)")
    parser.add_argument("--data-path", type=str, help="Path to sample data CSV")
    parser.add_argument("--model-path", type=str, help="Path to trained model")
    parser.add_argument("--num-records", type=int, default=1000, help="Number of sample records to generate")
    parser.add_argument("--save-results", action="store_true", help="Save demo results to files")
    
    args = parser.parse_args()
    
    if args.data_only:
        data_path = generate_data(num_records=args.num_records, output_path=args.data_path)
        print(f"Sample data saved to: {data_path}")
        
    elif args.train_only:
        if not args.data_path:
            print("Error: --data-path is required for --train-only")
            exit(1)
        model_paths = train_demo_models(args.data_path, model_types=["xgboost"])
        print(f"Models trained and saved to: {model_paths}")
        
    elif args.score_only:
        if not args.model_path:
            print("Error: --model-path is required for --score-only")
            exit(1)
        scoring_result = run_demo_scoring(args.model_path, args.data_path)
        print(f"Scoring complete. Result: {scoring_result}")
        
    elif args.manual_only:
        manual_result = generate_manual_score()
        print(f"Manual scoring complete. Result: {manual_result}")
        
    else:
        # Run full demo
        results = run_full_demo()
        
        if args.save_results:
            save_demo_results(results)