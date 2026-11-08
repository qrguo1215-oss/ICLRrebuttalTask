"""Use the audited sibling runtime, without executing the correction task."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SHARED = ROOT.parent / "Steering_Revision"
if not (SHARED / "engine.py").is_file():
    raise RuntimeError("Clone the complete ICLRrebuttalTask repository; Steering_Revision is required.")
sys.path.insert(0, str(SHARED))

