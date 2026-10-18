"""Native exact-process-instance checks for the one-shot continuation."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from continue_phase3_prior import alive
from hold_awake_for_training import creation_filetime


@unittest.skipUnless(os.name=='nt','Windows native process identity')
class ContinuationIdentityTests(unittest.TestCase):
    def test_current_process_only_matches_its_exact_creation(self):
        created=creation_filetime(os.getpid())
        self.assertTrue(alive(os.getpid(),created))
        self.assertFalse(alive(os.getpid(),created+1))

    def test_exited_child_with_retained_process_handle_is_not_alive(self):
        child=subprocess.Popen([sys.executable,'-c','pass'])
        try:
            created=creation_filetime(child.pid)
            child.wait(timeout=30)
            self.assertFalse(alive(child.pid,created))
        finally:
            if child.poll() is None:child.terminate();child.wait(timeout=30)

    def test_absent_process_is_not_alive(self):
        self.assertFalse(alive(4294967295,1))


if __name__=='__main__':unittest.main()
