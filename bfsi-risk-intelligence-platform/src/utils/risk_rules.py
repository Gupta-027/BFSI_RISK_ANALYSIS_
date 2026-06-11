"""
Business risk rules for the four BFSI modules.

A single source of truth that turns a model's risk *probability* (a float
between 0 and 1) into a business-friendly **risk category** and a recommended
**business action**.

This is used by the batch predictor so that the same thresholds drive the
result table, the KPI cards and the filters. Each module has three bands:
low / medium / high.

Each band is a tuple:  (upper_bound, level, category, action)
A probability falls into the first band whose ``upper_bound`` it is below.
"""

# Internal level ("low"/"medium"/"high") is kept alongside the display
# category so the UI can colour-code and count without re-parsing labels.
RISK_RULES = {
    "Credit Risk": [
        (0.35, "low", "Low Risk", "Approve"),
        (0.65, "medium", "Medium Risk", "Manual Review"),
        (1.01, "high", "High Risk", "Reject / Detailed Review"),
    ],
    "Fraud Detection": [
        (0.30, "low", "Safe", "Allow Transaction"),
        (0.70, "medium", "Suspicious", "OTP / Manual Review"),
        (1.01, "high", "High Fraud Risk", "Block Temporarily"),
    ],
    "Customer Churn": [
        (0.35, "low", "Low Churn Risk", "No Immediate Action"),
        (0.65, "medium", "Medium Churn Risk", "Send Offer"),
        (1.01, "high", "High Churn Risk", "Priority Retention Call"),
    ],
    "Loan Collection Risk": [
        (0.35, "low", "Low Priority", "Normal Reminder"),
        (0.65, "medium", "Medium Priority", "Follow-up Call"),
        (1.01, "high", "High Priority", "Collection Team Escalation"),
    ],
}


def classify(module_name, probability):
    """
    Map a risk probability to (level, category, action) for a module.

    Parameters
    ----------
    module_name : str
        One of the keys in RISK_RULES.
    probability : float
        Risk probability between 0 and 1.

    Returns
    -------
    tuple : (level, category, action)
        level is 'low' / 'medium' / 'high'.
    """
    bands = RISK_RULES[module_name]
    for upper_bound, level, category, action in bands:
        if probability < upper_bound:
            return level, category, action
    # Fallback (should never trigger because the last bound is > 1).
    return bands[-1][1], bands[-1][2], bands[-1][3]
