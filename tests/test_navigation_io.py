import hashlib,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.navigation_io import read,sha
class ReplayIOTests(unittest.TestCase):
    def test_reject_nonfinite_record_and_preserve_raw_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'record.json';p.write_bytes(b'{"speed":NaN}\r\n')
            with self.assertRaises(ValueError):read(p)
            self.assertEqual(sha(p),hashlib.sha256(p.read_bytes()).hexdigest())
