"""Required columns for a Fraud Detection batch input file."""

REQUIRED_COLUMNS = [
    "transaction_amount",
    "transaction_hour",
    "old_balance",
    "new_balance",
    "merchant_risk_score",
    "device_changed",
    "location_changed",
    "is_new_beneficiary",
    "failed_attempts_last_24h",
]
