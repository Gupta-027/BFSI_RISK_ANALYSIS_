"""
Reusable risk badge logic.

Maps any risk label used across the four modules to one of three levels
(low / medium / high) and renders a coloured pill badge.

  low    -> green   (Low Risk, Safe, Low Priority)
  medium -> amber   (Medium Risk, Suspicious, Medium Priority)
  high   -> red     (High Risk, High Fraud Risk, High Priority)
"""

LOW_LABELS = {"low", "safe", "low priority"}
MEDIUM_LABELS = {"medium", "suspicious", "medium priority"}


def badge_level(category):
    """Return 'low', 'medium' or 'high' for a given category string."""
    c = str(category).strip().lower()
    if c in LOW_LABELS:
        return "low"
    if c in MEDIUM_LABELS:
        return "medium"
    return "high"


def badge_html(category, label=None):
    """Return an HTML span styled as a coloured risk badge."""
    level = badge_level(category)
    text = label or category
    return f'<span class="badge badge-{level}">{text}</span>'
