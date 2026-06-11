# BFSI Risk Intelligence Platform

A clean, production-ready machine learning project for BFSI companies (Banks, NBFCs, Fintechs, Insurance) to assess and manage financial risk across multiple dimensions.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange)

---

## 📋 Overview

This platform provides end-to-end machine learning solutions for four critical BFSI risk use cases:

| Module | Purpose | Output |
|--------|---------|--------|
| **Credit Risk** | Assess creditworthiness | Low/Medium/High Risk + Lending Decision |
| **Fraud Detection** | Real-time fraud detection | Fraud Probability + Action (Allow/OTP/Block) |
| **Churn Prediction** | Identify at-risk customers | Churn Risk + Retention Strategy |
| **Collection Risk** | Prioritize recovery efforts | EMI Delay Probability + Collection Priority |

---

## 🎯 Business Problem

Financial institutions face multiple challenges:

1. **Credit Risk**: Poor lending decisions lead to defaults and NPAs
2. **Fraud**: Transaction fraud causes financial losses and customer trust issues
3. **Churn**: Losing customers affects revenue and market share
4. **Collections**: Inefficient collection strategies waste resources

This platform addresses all four problems with interpretable, deployable ML models.

---

## 🏗️ Project Structure

```
bfsi-risk-intelligence-platform/
│
├── README.md                           # Project documentation
├── INTERVIEW_EXPLANATION.md            # Interview preparation guide
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
│
├── data/
│   ├── raw/                           # Raw datasets
│   ├── processed/                     # Cleaned, preprocessed data
│   └── sample_inputs/                 # Batch upload templates (per module)
│       ├── credit_risk_sample.csv
│       ├── fraud_sample.csv
│       ├── churn_sample.csv
│       └── collection_risk_sample.csv
│
├── notebooks/
│   ├── 01_credit_risk_eda.ipynb      # Credit Risk EDA
│   ├── 02_fraud_detection_eda.ipynb  # Fraud Detection EDA
│   ├── 03_churn_prediction_eda.ipynb # Churn Prediction EDA
│   └── 04_collection_risk_eda.ipynb  # Collection Risk EDA
│
├── src/
│   ├── config.py                     # Configuration & paths
│   │
│   ├── data_fetching/                # Data loading scripts
│   │   ├── fetch_credit_data.py
│   │   ├── fetch_fraud_data.py
│   │   ├── fetch_churn_data.py
│   │   └── create_collection_data.py
│   │
│   ├── preprocessing/                # Data preprocessing
│   │   ├── credit_preprocessing.py
│   │   ├── fraud_preprocessing.py
│   │   ├── churn_preprocessing.py
│   │   └── collection_preprocessing.py
│   │
│   ├── models/                       # Model training
│   │   ├── train_credit_model.py
│   │   ├── train_fraud_model.py
│   │   ├── train_churn_model.py
│   │   └── train_collection_model.py
│   │
│   ├── prediction/                   # Model prediction
│   │   ├── predict_credit_risk.py
│   │   ├── predict_fraud.py
│   │   ├── predict_churn.py
│   │   ├── predict_collection_risk.py
│   │   └── batch_predictor.py        # Score a whole file at once
│   │
│   ├── validation/                   # Input file validation
│   │   ├── credit_schema.py          # Required columns per module
│   │   ├── fraud_schema.py
│   │   ├── churn_schema.py
│   │   ├── collection_schema.py
│   │   └── validate_input_file.py
│   │
│   ├── file_processing/              # Read uploaded files
│   │   ├── read_csv_excel.py
│   │   └── read_word_table.py        # Optional .docx table support
│   │
│   └── utils/                        # Utilities
│       ├── metrics.py               # Evaluation metrics
│       ├── model_io.py              # Model save/load
│       ├── risk_rules.py            # Risk category + action thresholds
│       └── common.py                # Helper functions
│
├── frontend/                          # Streamlit UI
│   ├── app.py                        # Main application
│   ├── styles.css                    # White + blue theme
│   ├── ui_helpers.py                 # UI helper functions
│   ├── components/                   # Reusable UI pieces
│   │   ├── header.py  hero.py  cards.py  risk_badges.py
│   │   ├── result_panel.py  footer.py
│   │   ├── file_uploader.py          # Upload widgets + sample download
│   │   └── data_table.py             # KPI cards + results table
│   └── pages/                        # One file per page
│       ├── home.py  ...  model_info_page.py
│       └── batch_prediction_page.py  # Batch Risk Scoring
│
└── artifacts/                         # Saved models
    ├── credit_risk_model.pkl
    ├── fraud_model.pkl
    ├── churn_model.pkl
    ├── collection_model.pkl
    ├── credit_preprocessor.pkl
    ├── fraud_preprocessor.pkl
    ├── churn_preprocessor.pkl
    └── collection_preprocessor.pkl
```

---

## 📊 ML Modules

### 1. Credit Risk Prediction

**Objective**: Classify customers as Low/Medium/High credit risk

**Features** (8):
- `age`, `job`, `credit_amount`, `duration`, `purpose`, `housing`, `saving_accounts`, `checking_account`

**Models Trained**:
- Logistic Regression
- Random Forest Classifier

**Output**:
- Risk Category (Low/Medium/High)
- Risk Probability
- Business Recommendation (Approve/Review/Reject)

---

### 2. Fraud Detection

**Objective**: Detect fraudulent transactions in real-time

**Features** (9):
- `transaction_amount`, `transaction_hour`, `old_balance`, `new_balance`, `merchant_risk_score`, `device_changed`, `location_changed`, `is_new_beneficiary`, `failed_attempts_last_24h`

**Special Handling**: Class imbalance (5% fraud) → Balanced class weights

**Models Trained**:
- Logistic Regression (balanced)
- Random Forest (balanced)

**Evaluation Metrics**:
- Precision, Recall, F1-Score (prioritize recall)
- ROC-AUC

**Output**:
- Fraud Probability
- Fraud Category (Safe/Suspicious/High Risk)
- Action (Allow/OTP/Block)

---

### 3. Customer Churn Prediction

**Objective**: Predict customer churn and recommend retention strategies

**Features** (10):
- `credit_score`, `age`, `tenure`, `balance`, `number_of_products`, `has_credit_card`, `is_active_member`, `estimated_salary`, `complaints`, `satisfaction_score`

**Models Trained**:
- Logistic Regression
- Random Forest Classifier

**Output**:
- Churn Probability
- Churn Risk Category (Low/Medium/High)
- Retention Suggestion (No Action/Send Offer/Priority Call)

---

### 4. Loan Collection Risk

**Objective**: Predict EMI delay and prioritize collection efforts

**Features** (10):
- `loan_amount`, `emi_amount`, `monthly_income`, `credit_score`, `past_due_days`, `missed_payments`, `loan_tenure_months`, `employment_type`, `existing_loans_count`, `repayment_ratio`

**Models Trained**:
- Logistic Regression
- Random Forest Classifier (multi-class)

**Output**:
- EMI Delay Probability
- Collection Priority (Low/Medium/High)
- Collection Action (Normal Reminder/Follow-up Call/Escalation)

---

## 🔄 Workflow

```
1. Data Fetching        → Create/load datasets in data/raw/
      ↓
2. Data Preprocessing   → Clean, encode, scale → data/processed/
      ↓
3. EDA                  → Explore distributions in notebooks/
      ↓
4. Model Training       → Train & evaluate → Save to artifacts/
      ↓
5. Prediction API       → Load model + predict
      ↓
6. Streamlit UI         → User interface for testing
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip

### Step 1: Clone & Navigate

```bash
cd bfsi-risk-intelligence-platform
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Data Loading

### Step 1: Fetch Datasets

```bash
# Credit Risk Data
python src/data_fetching/fetch_credit_data.py

# Fraud Detection Data
python src/data_fetching/fetch_fraud_data.py

# Churn Prediction Data
python src/data_fetching/fetch_churn_data.py

# Collection Risk Data
python src/data_fetching/create_collection_data.py
```

✓ Data is saved in `data/raw/` with automatic synthetic fallback

---

## 🔧 Data Preprocessing

```bash
# Credit Risk
python src/preprocessing/credit_preprocessing.py

# Fraud Detection
python src/preprocessing/fraud_preprocessing.py

# Churn Prediction
python src/preprocessing/churn_preprocessing.py

# Collection Risk
python src/preprocessing/collection_preprocessing.py
```

✓ Preprocessed data in `data/processed/`
✓ Preprocessors saved in `artifacts/`

---

## 🤖 Model Training

```bash
# Credit Risk
python src/models/train_credit_model.py

# Fraud Detection
python src/models/train_fraud_model.py

# Churn Prediction
python src/models/train_churn_model.py

# Collection Risk
python src/models/train_collection_model.py
```

> **Note:** Each training script automatically runs its preprocessing step
> first, so the processed data and saved preprocessor always stay in sync with
> the current raw data. Running the preprocessing scripts above separately is
> optional — handy if you only want to inspect the cleaned data.

✓ Best models selected based on F1-Score
✓ Models saved in `artifacts/`
✓ Evaluation metrics printed to console

---

## 🎨 Streamlit UI

```bash
streamlit run frontend/app.py
```

**Pages:**
1. **Home** - Platform overview
2. **Credit Risk Prediction** - Assess creditworthiness
3. **Fraud Detection** - Real-time fraud scoring
4. **Customer Churn Prediction** - Identify at-risk customers
5. **Loan Collection Risk** - Prioritize recovery efforts
6. **Batch Prediction** - Score a whole file of records at once
7. **Model Information** - Technical details

---

## 📁 Batch Prediction Workflow

The single-record pages are great for testing one customer at a time. In a real
bank/NBFC/fintech, teams need to score **thousands** of customers or
transactions in one go. The **Batch Prediction** page does exactly that.

**How it works:**

1. **User uploads CSV/Excel customer or transaction data** (Word `.docx` tables
   are also supported).
2. **App validates required columns** — it checks the file is not empty and
   that every column the chosen model needs is present, showing a clear error
   otherwise.
3. **Correct ML model is loaded** along with its preprocessor (selected from the
   module dropdown).
4. **Predictions are generated for all rows** in one vectorized pass.
5. **Risk category and business action are added** to each row, plus the risk
   probability and the module name.
6. **User filters high/medium/low risk records** by risk category, business
   action, and a minimum/maximum probability range.
7. **User downloads the final scored file** as CSV with one click.

**Added output columns:**

| Column | Meaning |
|--------|---------|
| `prediction_probability` | Risk probability (0–1) for the row |
| `risk_category` | Business risk label (e.g. *High Risk*, *Suspicious*) |
| `business_action` | Recommended action (e.g. *Reject / Detailed Review*) |
| `model_module` | Which module produced the score |

**Where the logic lives (clean separation):**

| Responsibility | Location |
|----------------|----------|
| Read CSV / Excel / Word | `src/file_processing/` |
| Validate required columns | `src/validation/` |
| Score every row | `src/prediction/batch_predictor.py` |
| Risk category & action rules | `src/utils/risk_rules.py` |
| Page UI only | `frontend/pages/batch_prediction_page.py` |

**Sample input files** for each module live in `data/sample_inputs/`. The page
has a *Download sample input file* button so users can grab a template, fill it
with their data, and upload it.

---

## 📊 Model Evaluation Metrics

### Credit Risk
- **Accuracy**: Overall correctness
- **Precision/Recall**: Per-class performance
- **F1-Score**: Balanced metric
- **Confusion Matrix**: Prediction breakdown

### Fraud Detection (Imbalanced Data)
- **Precision**: False alarm rate (important!)
- **Recall**: Fraud detection rate (critical!)
- **F1-Score**: Balanced between precision & recall
- **ROC-AUC**: Overall discriminative ability
- **Confusion Matrix**: True/False positives & negatives

### Churn Prediction
- **Accuracy**: Overall performance
- **Precision/Recall**: Identify churners correctly
- **F1-Score**: Balance between precision & recall
- **Confusion Matrix**: Prediction breakdown

### Collection Risk
- **Accuracy**: Per-priority classification
- **F1-Score (weighted)**: Handle imbalance
- **Confusion Matrix**: Priority classification accuracy

---

## 🎓 Interview Preparation

See [INTERVIEW_EXPLANATION.md](INTERVIEW_EXPLANATION.md) for:
- Why this project matters for BFSI
- Business problem breakdown
- Model selection rationale
- Metric selection explanations
- Technical implementation details
- Potential improvements

---

## 📝 Resume Bullet Points

✓ **Built an end-to-end BFSI Risk Intelligence Platform** using Python, scikit-learn, and Streamlit for financial risk analytics

✓ **Developed four independent ML modules** for credit risk scoring, fraud detection, customer churn prediction, and loan collection risk prioritization

✓ **Implemented preprocessing pipelines** including missing value handling, categorical encoding, outlier removal, and feature scaling for production readiness

✓ **Designed a Streamlit UI** enabling real-time predictions with probability scores and business-actionable recommendations for BFSI teams

✓ **Handled class imbalance** in fraud detection using balanced class weights and evaluated using precision, recall, and F1-score

✓ **Saved model artifacts** using joblib for production deployment with separate preprocessor persistence for consistent predictions

---

## 🔍 Key Features

| Feature | Implementation |
|---------|-----------------|
| **Simple Code** | Easy to understand, well-commented, interview-friendly |
| **Modular Design** | Each module independent, can be explained separately |
| **Production Ready** | Model persistence, preprocessing pipelines, error handling |
| **Clean UI** | Streamlit app with intuitive navigation and clear results |
| **Synthetic Data** | Fallback data generation when API access is unavailable |
| **Evaluation Focused** | Appropriate metrics for each use case (not just accuracy) |
| **Business Context** | Outputs include risk categories and actionable recommendations |
| **Documentation** | Comprehensive README, inline comments, EDA notebooks |

---

## 📈 Sample Predictions

### Credit Risk
Input: 35-year-old, admin job, ₹5000 credit, 24 months  
Output: `Low Risk (78% probability) → Approve`

### Fraud Detection
Input: ₹500 transaction at 3 AM, device changed, new beneficiary  
Output: `High Fraud Risk (92% probability) → Block Temporarily`

### Churn Prediction
Input: Active member, 5 years tenure, ₹100k balance, 0 complaints  
Output: `Low Churn Risk (12% probability) → No action`

### Collection Risk
Input: ₹500k loan, 45 days past due, 3 missed payments, ₹80k income  
Output: `High Priority (88% probability) → Collection team escalation`

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Data Processing** | Pandas, NumPy |
| **ML Models** | Scikit-learn (Logistic Regression, Random Forest) |
| **Model Persistence** | Joblib |
| **Frontend** | Streamlit |
| **Evaluation** | Scikit-learn metrics |
| **Notebooks** | Jupyter |

---

## 📚 Exploratory Data Analysis

Run Jupyter notebooks for detailed EDA:

```bash
jupyter notebook notebooks/01_credit_risk_eda.ipynb
jupyter notebook notebooks/02_fraud_detection_eda.ipynb
jupyter notebook notebooks/03_churn_prediction_eda.ipynb
jupyter notebook notebooks/04_collection_risk_eda.ipynb
```

Each notebook includes:
- Data loading and exploration
- Feature distributions
- Target variable analysis
- Missing values assessment
- Key insights for modeling

---

## 🎯 Use Cases

1. **Banks** - Lending decisions, fraud prevention, customer retention, NPA management
2. **NBFCs** - Credit assessment, collection optimization, risk management
3. **Fintechs** - Real-time fraud detection, customer scoring, churn prevention
4. **Insurance** - Risk assessment, customer lifetime value, claims fraud detection

---

## 🔐 Important Notes

- ✓ Synthetic data provided as fallback
- ✓ No real customer data in repository
- ✓ Models are for demonstration/learning
- ✓ Comprehensive preprocessing ensures data quality
- ✓ Balanced metrics used for imbalanced datasets
- ✓ All code is reproducible with fixed random seeds

---

## 📞 Support & Questions

For detailed technical explanations, see [INTERVIEW_EXPLANATION.md](INTERVIEW_EXPLANATION.md)

---

## 📄 License

MIT License - Feel free to use for learning and projects

---

**Built for BFSI Risk Intelligence | Clean Code | Production Ready | Interview Friendly** 🏦
