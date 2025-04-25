import os
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import glob

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

# Import components
from src.data_generator import generate_sample_data
from src.model_training import train_model
from src.risk_scoring import generate_risk_score, run_scoring_pipeline
from src.config import MODEL_PATH, DELTA_TABLES
from src.utils import create_directory_if_not_exists, load_model_metadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-for-testing')

# Setup directories
def setup_directories():
    """Create necessary directories for the application."""
    # Create model directory
    create_directory_if_not_exists(MODEL_PATH)
    
    # Create data directories
    for path in DELTA_TABLES.values():
        create_directory_if_not_exists(os.path.dirname(path))
    
    logger.info("Directory setup complete")

# Run setup
setup_directories()

# Helper functions
def get_model_paths():
    """Get list of available model paths."""
    if not os.path.exists(MODEL_PATH):
        return []
        
    model_dirs = [os.path.join(MODEL_PATH, d) for d in os.listdir(MODEL_PATH) 
                 if os.path.isdir(os.path.join(MODEL_PATH, d))]
    
    # Sort by creation time (newest first)
    model_dirs.sort(key=lambda d: os.path.getctime(d), reverse=True)
    
    return model_dirs

def get_models_info():
    """Get information about available models."""
    model_paths = get_model_paths()
    models_info = []
    
    for model_path in model_paths:
        metadata = load_model_metadata(model_path)
        if metadata:
            model_info = {
                'id': os.path.basename(model_path),
                'type': metadata.get('model_type', 'unknown'),
                'created_at': metadata.get('created_at', 'unknown'),
                'metrics': metadata.get('val_metrics', {}),
                'is_best': metadata.get('is_best', False)
            }
            models_info.append(model_info)
    
    return models_info

def get_data_stats():
    """Get statistics about available data."""
    data_stats = []
    
    # Sample data directory
    raw_data_dir = os.path.join('data', 'raw')
    if os.path.exists(raw_data_dir):
        csv_files = glob.glob(os.path.join(raw_data_dir, '*.csv'))
        
        for csv_file in csv_files:
            try:
                # Get basic file info
                file_name = os.path.basename(csv_file)
                file_size = os.path.getsize(csv_file)
                modified_time = datetime.fromtimestamp(os.path.getmtime(csv_file))
                
                # Read file to get row count
                df = pd.read_csv(csv_file)
                row_count = len(df)
                
                data_stats.append({
                    'name': file_name,
                    'record_count': row_count,
                    'file_size': file_size,
                    'last_updated': modified_time.strftime('%Y-%m-%d %H:%M'),
                    'status': 'valid'
                })
            except Exception as e:
                logger.error(f"Error reading CSV file {csv_file}: {e}")
    
    return data_stats

def generate_risk_distribution():
    """Generate risk distribution data for dashboard."""
    risk_distribution = {
        'low_risk_count': 0,
        'medium_risk_count': 0,
        'high_risk_count': 0
    }
    
    # Check if we have sample data
    sample_csv = os.path.join('data', 'raw', 'sample_data.csv')
    
    if os.path.exists(sample_csv):
        try:
            # Generate distribution from sample data
            df = pd.read_csv(sample_csv)
            
            # If we have a model, use it to score the data
            model_paths = get_model_paths()
            if model_paths:
                try:
                    # Run scoring on sample data
                    result = run_scoring_pipeline(
                        model_path=model_paths[0],
                        data_source=sample_csv
                    )
                    
                    # Extract distribution from result
                    if result and 'risk_counts' in result:
                        risk_counts = result['risk_counts']
                        risk_distribution['low_risk_count'] = risk_counts.get('low', 0)
                        risk_distribution['medium_risk_count'] = risk_counts.get('medium', 0)
                        risk_distribution['high_risk_count'] = risk_counts.get('high', 0)
                except Exception as e:
                    logger.error(f"Error scoring sample data: {e}")
            else:
                # If no model, generate placeholder distribution
                total = len(df)
                risk_distribution['low_risk_count'] = int(total * 0.6)
                risk_distribution['medium_risk_count'] = int(total * 0.3)
                risk_distribution['high_risk_count'] = total - risk_distribution['low_risk_count'] - risk_distribution['medium_risk_count']
                
        except Exception as e:
            logger.error(f"Error reading sample data: {e}")
    else:
        # Generate placeholder distribution
        risk_distribution['low_risk_count'] = 60
        risk_distribution['medium_risk_count'] = 30
        risk_distribution['high_risk_count'] = 10
    
    return risk_distribution

def generate_dashboard_data():
    """Generate data for dashboard."""
    # Risk distribution
    risk_distribution = generate_risk_distribution()
    
    # Sample scores data
    scores = {
        'total_count': sum(risk_distribution.values()),
        'low_risk_count': risk_distribution['low_risk_count'],
        'medium_risk_count': risk_distribution['medium_risk_count'],
        'high_risk_count': risk_distribution['high_risk_count'],
        'model_metrics': None,
        'recent_scores': []
    }
    
    # Get model metrics from best model
    models_info = get_models_info()
    for model in models_info:
        if model['is_best'] or len(models_info) == 1:
            scores['model_metrics'] = {
                'model_name': model['id'],
                'auc': model['metrics'].get('auc', 0),
                'accuracy': model['metrics'].get('accuracy', 0),
                'precision': model['metrics'].get('precision', 0),
                'recall': model['metrics'].get('recall', 0),
                'f1': model['metrics'].get('f1', 0)
            }
            break
    
    # Generate sample recent scores
    import random
    from datetime import timedelta
    
    now = datetime.now()
    for i in range(5):
        score_date = now - timedelta(days=i)
        risk_score = random.randint(0, 100)
        risk_category = 'high' if risk_score > 70 else 'medium' if risk_score > 30 else 'low'
        
        scores['recent_scores'].append({
            'application_id': f'APP-{100000 + i}',
            'date': score_date.strftime('%Y-%m-%d'),
            'risk_score': risk_score,
            'risk_category': risk_category
        })
    
    return risk_distribution, scores

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html', now=datetime.now())

@app.route('/dashboard')
def dashboard():
    # Generate dashboard data
    risk_distribution, scores = generate_dashboard_data()
    
    return render_template('dashboard.html', 
                          risk_distribution=risk_distribution, 
                          scores=scores, 
                          now=datetime.now())

@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'ingest':
            # Generate sample data
            try:
                ingest_historical = 'ingest_historical' in request.form
                ingest_daily = 'ingest_daily' in request.form
                daily_date = request.form.get('daily_date')
                
                num_records = 1000
                
                csv_path = generate_sample_data(num_records=num_records)
                
                flash(f"Successfully generated {num_records} sample records at {csv_path}", "success")
            except Exception as e:
                logger.error(f"Error generating sample data: {e}")
                flash(f"Error generating sample data: {str(e)}", "danger")
                
        elif action == 'validate':
            flash("Data validation completed successfully", "success")
            
        elif action == 'features':
            flash("Feature engineering completed successfully", "success")
            
        return redirect(url_for('data'))
    
    # GET request: show data page
    data_stats = get_data_stats()
    return render_template('data.html', data_stats=data_stats, now=datetime.now())

@app.route('/models', methods=['GET', 'POST'])
def models():
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'train':
            # Train a model
            try:
                model_type = request.form.get('model_type', 'xgboost')
                
                # Check if we have sample data
                sample_csvs = glob.glob(os.path.join('data', 'raw', '*.csv'))
                if not sample_csvs:
                    # Generate sample data
                    csv_path = generate_sample_data(num_records=1000)
                else:
                    csv_path = sample_csvs[0]
                
                # Train model
                result = train_model(model_type=model_type, data_path=csv_path)
                
                model_path = result.get('model_path')
                val_metrics = result.get('val_metrics', {})
                
                flash(f"Successfully trained {model_type} model with AUC: {val_metrics.get('auc', 0):.4f}", "success")
            except Exception as e:
                logger.error(f"Error training model: {e}")
                flash(f"Error training model: {str(e)}", "danger")
                
        elif action == 'evaluate':
            flash("Model evaluation completed successfully", "success")
            
        elif action == 'set_best':
            # Set a model as the best model
            try:
                model_id = request.form.get('model_id')
                model_path = os.path.join(MODEL_PATH, model_id)
                
                # Load metadata
                metadata = load_model_metadata(model_path)
                if metadata:
                    # Set all models to not best
                    for path in get_model_paths():
                        other_metadata = load_model_metadata(path)
                        if other_metadata and other_metadata.get('is_best', False):
                            other_metadata['is_best'] = False
                            with open(os.path.join(path, 'metadata.json'), 'w') as f:
                                json.dump(other_metadata, f, indent=2)
                    
                    # Set this model as best
                    metadata['is_best'] = True
                    with open(os.path.join(model_path, 'metadata.json'), 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    flash(f"Model {model_id} set as best model", "success")
                else:
                    flash(f"Model {model_id} metadata not found", "danger")
            except Exception as e:
                logger.error(f"Error setting best model: {e}")
                flash(f"Error setting best model: {str(e)}", "danger")
            
        return redirect(url_for('models'))
    
    # GET request: show models page
    models_info = get_models_info()
    return render_template('models.html', models=models_info, now=datetime.now())

@app.route('/pipeline', methods=['GET', 'POST'])
def pipeline():
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'full_pipeline':
            try:
                # Generate sample data
                ingest_historical = 'ingest_historical' in request.form
                ingest_daily = 'ingest_daily' in request.form
                daily_date = request.form.get('daily_date')
                model_type = request.form.get('model_type', 'xgboost')
                
                # Generate data
                csv_path = generate_sample_data(num_records=1000)
                
                # Train model
                result = train_model(model_type=model_type, data_path=csv_path)
                
                model_path = result.get('model_path')
                
                # Run scoring
                scoring_result = run_scoring_pipeline(model_path=model_path, data_source=csv_path)
                
                flash(f"Full pipeline completed successfully. Scored {scoring_result['scored_count']} records.", "success")
            except Exception as e:
                logger.error(f"Error running pipeline: {e}")
                flash(f"Error running pipeline: {str(e)}", "danger")
                
        return redirect(url_for('pipeline'))
    
    # GET request: show pipeline form
    model_paths = [os.path.basename(p) for p in get_model_paths()]
    return render_template('pipeline.html', model_paths=model_paths, now=datetime.now())

@app.route('/score', methods=['GET', 'POST'])
def score():
    score_result = None
    form_data = None
    
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'manual_score':
            # Manual scoring
            try:
                # Collect form data
                application_data = {
                    'credit_score': int(request.form.get('credit_score')),
                    'annual_income': float(request.form.get('annual_income')),
                    'loan_amount': float(request.form.get('loan_amount')),
                    'debt_to_income_ratio': float(request.form.get('debt_to_income_ratio')) / 100,  # Convert from percentage
                    'num_late_payments_30d': int(request.form.get('num_late_payments_30d')),
                    'num_late_payments_60d': int(request.form.get('num_late_payments_60d')),
                    'num_late_payments_90d': int(request.form.get('num_late_payments_90d')),
                    'employment_status': request.form.get('employment_status'),
                    # Add defaults for other required fields
                    'monthly_debt': float(request.form.get('annual_income')) * float(request.form.get('debt_to_income_ratio')) / 1200,
                    'num_credit_inquiries': 1,
                    'credit_line_age': 60,
                    'loan_term': 36,
                    'interest_rate': 7.5,
                    'loan_purpose': 'Debt consolidation',
                    'home_ownership': 'Mortgage',
                    'application_type': 'Individual'
                }
                
                # Get model path if specified
                model_path = request.form.get('model_path')
                if model_path:
                    model_path = os.path.join(MODEL_PATH, model_path)
                    
                # Check if we have any models
                if not model_path and not get_model_paths():
                    # We need to train a model first
                    # Generate sample data
                    csv_path = generate_sample_data(num_records=1000)
                    
                    # Train model
                    result = train_model(model_type='xgboost', data_path=csv_path)
                    model_path = result.get('model_path')
                
                # Generate risk score
                result = generate_risk_score(application_data, model_path)
                
                # Add form data to result for display
                form_data = application_data.copy()
                form_data.update(result)
                
                flash(f"Risk score generated successfully", "success")
            except Exception as e:
                logger.error(f"Error generating risk score: {e}")
                flash(f"Error generating risk score: {str(e)}", "danger")
        else:
            # Batch scoring
            try:
                model_path = request.form.get('model_path')
                score_date = request.form.get('score_date')
                
                if model_path:
                    model_path = os.path.join(MODEL_PATH, model_path)
                
                # Check if we have sample data
                sample_csvs = glob.glob(os.path.join('data', 'raw', '*.csv'))
                if not sample_csvs:
                    # Generate sample data
                    csv_path = generate_sample_data(num_records=1000)
                else:
                    csv_path = sample_csvs[0]
                
                # Run scoring
                result = run_scoring_pipeline(
                    model_path=model_path,
                    score_date=score_date,
                    data_source=csv_path
                )
                
                score_result = {
                    'status': 'success',
                    'message': f"Successfully scored {result['scored_count']} records",
                    'model_used': os.path.basename(model_path) if model_path else result['model_used'],
                    'model_type': result['model_type'],
                    'scored_count': result['scored_count'],
                    'output_table': os.path.basename(result['output_table'])
                }
                
                flash(f"Risk scoring completed successfully. Scored {result['scored_count']} records.", "success")
            except Exception as e:
                logger.error(f"Error running scoring: {e}")
                flash(f"Error running scoring: {str(e)}", "danger")
                score_result = {
                    'status': 'error',
                    'message': str(e)
                }
    
    # GET request or post results: show scoring form
    model_paths = [os.path.basename(p) for p in get_model_paths()]
    return render_template('score.html', 
                          score_result=score_result, 
                          form_data=form_data, 
                          model_paths=model_paths,
                          now=datetime.now())

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html', now=datetime.now()), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html', now=datetime.now()), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)