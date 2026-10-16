"""Run the local package without installation or global environment changes."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from senior_ai.__main__ import main
raise SystemExit(main())
