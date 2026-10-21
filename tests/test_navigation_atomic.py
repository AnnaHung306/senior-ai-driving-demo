import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.navigation_atomic import write_json

class NavigationAtomicTests(unittest.TestCase):
    def test_transient_replace_preserves_old_complete_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'progress.json';path.write_text('{"old":true}');replace=Path.replace;calls=[]
            def guarded(source,target):
                calls.append(1)
                if len(calls)<3:
                    self.assertEqual(json.loads(path.read_text()),{'old':True});raise PermissionError('temporary read lock')
                return replace(source,target)
            with patch.object(Path,'replace',guarded):receipt=write_json(path,{'new':True})
            self.assertEqual(receipt['attempts'],3);self.assertEqual(json.loads(path.read_text()),{'new':True});self.assertEqual(list(Path(temp).glob('*.tmp')),[])
    def test_permanent_lock_bounded_and_original_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'progress.json';path.write_text('{"old":true}')
            with patch.object(Path,'replace',side_effect=PermissionError('locked')):
                with self.assertRaises(PermissionError):write_json(path,{'new':True},timeout_s=.03)
            self.assertEqual(json.loads(path.read_text()),{'old':True});self.assertEqual(list(Path(temp).glob('*.tmp')),[])
    def test_nonfinite_cannot_replace_previous(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'progress.json';write_json(path,{'valid':1})
            with self.assertRaises(ValueError):write_json(path,{'bad':float('nan')})
            self.assertEqual(json.loads(path.read_text()),{'valid':1})

if __name__=='__main__':unittest.main()
