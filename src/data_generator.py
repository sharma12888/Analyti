import os
import random
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from src.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    DATE_FEATURES,
    ID_FEATURES,
    TARGET_COLUMN,
    MOCK_DATA_CONFIG,
    DATA_DIR
)

logger = logging.getLogger(__name__)

class SampleDataGenerator:
    """
    Generate sample data for testing the risk score analysis system.
    This data generation is for demonstration purposes only.
    """
    
    def __init__(self, num_records: int = 1000, random_seed: int = 42, default_rate: float = 0.15):
        """
        Initialize the sample data generator.
        
        Args:
            num_records: Number of records to generate
            random_seed: Random seed for reproducibility
            default_rate: Default rate (proportion of loans that will default)
        """
        self.num_records = num_records
        self.random_seed = random_seed
        self.default_rate = default_rate
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Define value ranges for numerical features
        self.numerical_ranges = {
            "loan_amount": (1000, 50000),  # Loan amount in $
            "annual_income": (20000, 150000),  # Annual income in $
            "debt_to_income_ratio": (0.1, 0.6),  # Debt-to-income ratio
            "monthly_debt": (200, 5000),  # Monthly debt in $
            "credit_score": (300, 850),  # Credit score
            "num_credit_inquiries": (0, 10),  # Number of credit inquiries
            "num_late_payments_30d": (0, 5),  # Number of 30-day late payments
            "num_late_payments_60d": (0, 3),  # Number of 60-day late payments
            "num_late_payments_90d": (0, 2),  # Number of 90-day late payments
            "credit_line_age": (0, 240),  # Credit line age in months
            "loan_term": (12, 60),  # Loan term in months
            "interest_rate": (3.0, 15.0),  # Interest rate in %
        }
        
        # Define categorical feature values
        self.categorical_values = {
            "employment_status": ["Employed", "Self-employed", "Unemployed", "Retired"],
            "loan_purpose": ["Debt consolidation", "Home improvement", "Business", "Education", "Other"],
            "home_ownership": ["Own", "Mortgage", "Rent", "Other"],
            "application_type": ["Individual", "Joint"],
        }
        
    def _generate_id(self, prefix: str, idx: int) -> str:
        """Generate a unique ID with a prefix."""
        return f"{prefix}-{idx:06d}"
    
    def _generate_application_date(self) -> datetime:
        """Generate a random application date within the last year."""
        days_ago = random.randint(0, 365)
        return datetime.now() - timedelta(days=days_ago)
    
    def _generate_numerical_features(self) -> Dict[str, float]:
        """Generate random numerical features based on defined ranges."""
        features = {}
        for feature, (min_val, max_val) in self.numerical_ranges.items():
            if feature in ["num_credit_inquiries", "num_late_payments_30d", 
                          "num_late_payments_60d", "num_late_payments_90d",
                          "loan_term", "credit_line_age"]:
                # Integer features
                features[feature] = random.randint(min_val, max_val)
            else:
                # Float features
                features[feature] = round(random.uniform(min_val, max_val), 2)
        return features
    
    def _generate_categorical_features(self) -> Dict[str, str]:
        """Generate random categorical features."""
        features = {}
        for feature, values in self.categorical_values.items():
            features[feature] = random.choice(values)
        return features
    
    def _calculate_default_probability(self, features: Dict[str, Any]) -> float:
        """
        Calculate probability of default based on features.
        This is a simplified model for demonstration purposes.
        """
        # Base probability centered at the default rate
        prob = self.default_rate
        
        # Credit score effect (higher score -> lower default probability)
        credit_score_factor = 1.0 - (features["credit_score"] - 300) / 550
        prob += credit_score_factor * 0.3
        
        # Debt-to-income effect (higher DTI -> higher default probability)
        dti_factor = features["debt_to_income_ratio"] / 0.6
        prob += dti_factor * 0.2
        
        # Late payments effect (more late payments -> higher default probability)
        late_payment_factor = (features["num_late_payments_30d"] * 0.05 +
                              features["num_late_payments_60d"] * 0.1 +
                              features["num_late_payments_90d"] * 0.15)
        prob += late_payment_factor
        
        # Income effect (higher income -> lower default probability)
        income_factor = 1.0 - (features["annual_income"] - 20000) / 130000
        prob += income_factor * 0.1
        
        # Loan amount effect (higher loan amount -> higher default probability)
        loan_factor = (features["loan_amount"] - 1000) / 49000
        prob += loan_factor * 0.1
        
        # Clamp probability between 0 and 1
        return max(0.0, min(1.0, prob))
    
    def generate_data(self) -> pd.DataFrame:
        """
        Generate sample data for loan applications.
        
        Returns:
            DataFrame with generated data
        """
        data = []
        for i in range(self.num_records):
            # Generate numerical and categorical features
            record = {
                "application_id": self._generate_id("APP", i),
                "customer_id": self._generate_id("CUST", i),
                "application_date": self._generate_application_date()
            }
            
            # Add numerical features
            record.update(self._generate_numerical_features())
            
            # Add categorical features
            record.update(self._generate_categorical_features())
            
            # Calculate default probability
            default_prob = self._calculate_default_probability(record)
            
            # Determine default status based on probability
            record[TARGET_COLUMN] = 1 if random.random() < default_prob else 0
            
            data.append(record)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Ensure date columns are datetime type
        for date_col in DATE_FEATURES:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col])
        
        logger.info(f"Generated {len(df)} sample records")
        
        return df
    
    def save_to_csv(self, output_dir: str = None, filename: str = "sample_data.csv") -> str:
        """
        Generate sample data and save to CSV file.
        
        Args:
            output_dir: Directory to save the CSV file (defaults to data/raw)
            filename: Name of the CSV file
            
        Returns:
            Path to the saved CSV file
        """
        df = self.generate_data()
        
        if output_dir is None:
            output_dir = os.path.join(DATA_DIR, "raw")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Construct file path
        file_path = os.path.join(output_dir, filename)
        
        # Save to CSV
        df.to_csv(file_path, index=False)
        
        logger.info(f"Saved sample data to {file_path}")
        
        return file_path
    
    def save_to_database(self, table_name: str = "loan_applications") -> None:
        """
        Generate sample data and save to PostgreSQL database.
        
        Args:
            table_name: Name of the table to save the data to
        """
        from sqlalchemy import create_engine
        import os
        
        df = self.generate_data()
        
        # Get database URL from environment
        database_url = os.environ.get("DATABASE_URL")
        
        if database_url:
            # Create SQLAlchemy engine
            engine = create_engine(database_url)
            
            # Save to database
            df.to_sql(table_name, engine, if_exists="replace", index=False)
            
            logger.info(f"Saved {len(df)} records to database table '{table_name}'")
        else:
            logger.error("DATABASE_URL environment variable not set")
            
def generate_sample_data(num_records: int = None, output_file: str = None) -> str:
    """
    Convenience function to generate sample data.
    
    Args:
        num_records: Number of records to generate
        output_file: Path to output file
        
    Returns:
        Path to the saved CSV file
    """
    if num_records is None:
        num_records = MOCK_DATA_CONFIG["num_records"]
    
    generator = SampleDataGenerator(
        num_records=num_records,
        random_seed=MOCK_DATA_CONFIG["random_seed"],
        default_rate=MOCK_DATA_CONFIG["default_rate"]
    )
    
    if output_file:
        return generator.save_to_csv(filename=os.path.basename(output_file),
                                     output_dir=os.path.dirname(output_file))
    else:
        return generator.save_to_csv()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Generate sample data
    file_path = generate_sample_data()
    print(f"Sample data saved to: {file_path}")