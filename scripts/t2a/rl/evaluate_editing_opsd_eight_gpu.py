"""Eight-worker audio evaluation with world-invariant native validation noise."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))

from scripts.t2a.rl import evaluate_editing_opsd_branch_cross as evaluation
from scripts.t2a.rl import validate_editing_opsd_native as native_validation
from scripts.t2a.rl.opsd_canonical_validation import validate_native


if __name__=='__main__':
    native_validation.validate_native=validate_native
    evaluation.main()
