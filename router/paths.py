"""Output locations: one folder per experiment under results/, with its figures inside it."""
from router.data import ROOT

RESULTS = ROOT / "results"
DATASETS = RESULTS / "datasets"
SCORES = RESULTS / "00_cometkiwi_scores"
AUDIT = RESULTS / "01_structural_audit"
PREDICTOR = RESULTS / "02_risk_predictor"
POLICIES = RESULTS / "03_routing_policies"
REPLAY = RESULTS / "04_cicd_replay"
CONFIDENCE = RESULTS / "05_confidence_intervals"
IMPROVED = RESULTS / "06_improved_router"

PAIR_SCORES = SCORES / "cometkiwi_pair_scores.csv.gz"
ROUTER_DATASET = DATASETS / "router_dataset.csv.gz"


def oof_path(nmt):
    """Out-of-fold Stage 1 risk predictions for one NMT engine (Experiment 2)."""
    return PREDICTOR / f"oof_predictions_{nmt}.csv.gz"


def ensure(folder):
    """Create a results folder (and its parents) if needed, and return it."""
    folder.mkdir(parents=True, exist_ok=True)
    return folder
