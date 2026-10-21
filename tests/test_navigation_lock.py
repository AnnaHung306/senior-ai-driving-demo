import sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.navigation_lock import TickLease
class LeaseTests(unittest.TestCase):
    def test_competing_handle_rejected_then_release_allows_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=TickLease(Path(tmp)/'owner.lock');b=TickLease(Path(tmp)/'owner.lock');a.acquire()
            try:
                with self.assertRaises(RuntimeError):b.acquire()
            finally:a.release()
            b.acquire();b.release();b.release()
