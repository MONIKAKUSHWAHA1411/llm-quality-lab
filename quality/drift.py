import csv
from pathlib import Path
from statistics import mean

RESULTS_FILE = Path("results/phase1_scores.csv")


def check_relevance_drift(
    current_avg_relevance: float,
    window: int = 5,
    allowed_drop: float = 0.05
) -> bool:
    """
    Returns True if relevance drift is acceptable.
    Returns False if degradation exceeds allowed drop.
    """

    if not RESULTS_FILE.exists():
        return True  # no baseline yet

    historical = []

    with open(RESULTS_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            historical.append(float(row["avg_relevance"]))

    if len(historical) < window:
        return True  # insufficient data

    baseline = mean(historical[-window:])
    drift = baseline - current_avg_relevance

    return drift <= allowed_drop


def check_safety_drift(current_max_safety: float) -> bool:
    """
    Safety drift is strict.
    Any increase above zero is suspicious.
    """

    if not RESULTS_FILE.exists():
        return True

    historical = []

    with open(RESULTS_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            historical.append(float(row["max_safety"]))

    if not historical:
        return True

    baseline_max = max(historical)

    return current_max_safety <= baseline_max
