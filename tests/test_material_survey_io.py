"""Regression for the observed Windows atomic-replace sharing violation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from survey_sunny_materials import write

TEST_ROOT = Path(__file__).resolve().parents[1] / 'artifacts/runs/test-temp'
TEST_ROOT.mkdir(parents=True, exist_ok=True)


class SurveyAtomicWriteTests(unittest.TestCase):
    def test_temporary_windows_lock_retries_without_partial_json(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as directory:
            target = Path(directory) / 'progress.json'
            target.write_text('{"completed": 7}', encoding='utf-8')
            actual_replace = Path.replace
            attempts = []

            def locked_then_available(source, destination):
                attempts.append(1)
                self.assertEqual(json.loads(target.read_text(encoding='utf-8')), {'completed': 7})
                if len(attempts) <= 2:
                    raise PermissionError('simulated Windows sharing violation')
                return actual_replace(source, destination)

            with patch.object(Path, 'replace', locked_then_available), patch('survey_sunny_materials.time.sleep'):
                write(target, {'completed': 8})
            self.assertEqual(len(attempts), 3)
            self.assertEqual(json.loads(target.read_text(encoding='utf-8')), {'completed': 8})

    def test_persistent_lock_is_bounded_and_retains_last_good_progress(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as directory:
            target = Path(directory) / 'progress.json'
            target.write_text('{"completed": 7}', encoding='utf-8')
            with patch.object(Path, 'replace', side_effect=PermissionError('locked')) as replace, patch('survey_sunny_materials.time.sleep'):
                with self.assertRaises(PermissionError):
                    write(target, {'completed': 8})
            self.assertEqual(replace.call_count, 8)
            self.assertEqual(json.loads(target.read_text(encoding='utf-8')), {'completed': 7})


if __name__ == '__main__':
    unittest.main()
