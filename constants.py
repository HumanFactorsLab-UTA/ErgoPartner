"""
Shared Constants - Centralized constants used across the project.

This module eliminates duplication of magic numbers, color mappings,
risk thresholds, and other shared configuration values.
"""

# ---------------------------------------------------------------------------
# REBA Risk Bands
# ---------------------------------------------------------------------------
# Score Interpretation (Hignett & McAtamney, 2000):
#     1       : Negligible risk - no action necessary
#     2-3     : Low risk - change may be needed
#     4-7     : Medium risk - further investigation, change soon
#     8-10    : High risk - investigate and implement change
#     11+     : Very high risk - implement change immediately
#
# Each tuple: (low_score, high_score, label, color_hex)
REBA_RISK_BANDS = [
    (1, 1, "Negligible", "#2ecc71"),
    (2, 3, "Low", "#f1c40f"),
    (4, 7, "Medium", "#e67e22"),
    (8, 10, "High", "#e74c3c"),
    (11, 15, "Very High", "#8e44ad"),
]

# REBA risk level lookup: score -> (risk_level, action_level)
REBA_RISK_LEVELS = {
    1: ("Negligible", "None necessary"),
    2: ("Low", "May be necessary"),
    3: ("Low", "May be necessary"),
    4: ("Medium", "Necessary"),
    5: ("Medium", "Necessary"),
    6: ("Medium", "Necessary"),
    7: ("Medium", "Necessary"),
    8: ("High", "Necessary soon"),
    9: ("High", "Necessary soon"),
    10: ("High", "Necessary soon"),
    11: ("Very High", "Immediate action"),
    12: ("Very High", "Immediate action"),
    13: ("Very High", "Immediate action"),
    14: ("Very High", "Immediate action"),
    15: ("Very High", "Immediate action"),
}


def reba_risk_level(score: int) -> str:
    """Return the REBA risk level string for a given score."""
    for low, high, label, _ in REBA_RISK_BANDS:
        if low <= score <= high:
            return label
    return "Very High"  # 11+


# ---------------------------------------------------------------------------
# RULA Risk Bands
# ---------------------------------------------------------------------------
RULA_RISK_BANDS = [
    (1, 2, "Acceptable", "#2ecc71"),
    (3, 4, "Investigate", "#f1c40f"),
    (5, 6, "Investigate Soon", "#e67e22"),
    (7, 7, "Investigate Immediately", "#e74c3c"),
]

# ---------------------------------------------------------------------------
# Risk Level Colors (shared across all charting)
# ---------------------------------------------------------------------------
RISK_COLORS = {
    "Negligible": "#2ecc71",
    "Low": "#f1c40f",
    "Medium": "#e67e22",
    "High": "#e74c3c",
    "Very High": "#8e44ad",
}

# ---------------------------------------------------------------------------
# Camera Role Colors (for per-camera visualization)
# ---------------------------------------------------------------------------
CAMERA_ROLE_COLORS = {
    "front": "#e74c3c",
    "side": "#3498db",
    "iso": "#2ecc71",
}

# ---------------------------------------------------------------------------
# Default Processing Parameters
# ---------------------------------------------------------------------------
DEFAULT_FPS = 30.0
DEFAULT_BUFFER_SECONDS = 3.0
DEFAULT_MAX_PEAK_FRAMES = 20
DEFAULT_REBA_PEAK_THRESHOLD = 7
DEFAULT_RULA_PEAK_THRESHOLD = 5
