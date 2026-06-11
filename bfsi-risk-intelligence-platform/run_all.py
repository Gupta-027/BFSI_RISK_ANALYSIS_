"""
Convenience runner for the BFSI Risk Intelligence Platform.

Runs the full data pipeline in one command:
  1. Generate (or fall back to synthetic) raw data for all four modules
  2. Train all four models (each training step auto-runs its preprocessing)

After this finishes, launch the UI with:
  streamlit run frontend/app.py
"""

import os
import sys

# Make the src package importable when running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data_fetching.fetch_credit_data import fetch_credit_data
from data_fetching.fetch_fraud_data import fetch_fraud_data
from data_fetching.fetch_churn_data import fetch_churn_data
from data_fetching.create_collection_data import create_collection_data

from models.train_credit_model import train_credit_risk_model
from models.train_fraud_model import train_fraud_model
from models.train_churn_model import train_churn_model
from models.train_collection_model import train_collection_model


def main():
    print("\n" + "#" * 60)
    print("# STEP 1 / 2 : Generating raw datasets")
    print("#" * 60)
    fetch_credit_data()
    fetch_fraud_data()
    fetch_churn_data()
    create_collection_data()

    print("\n" + "#" * 60)
    print("# STEP 2 / 2 : Training models (preprocessing runs automatically)")
    print("#" * 60)
    train_credit_risk_model()
    train_fraud_model()
    train_churn_model()
    train_collection_model()

    print("\n" + "=" * 60)
    print("✓ All datasets generated and all models trained.")
    print("  Next: streamlit run frontend/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
