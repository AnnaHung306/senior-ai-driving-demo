"""Actual GT sensor setup followed by intentional failure before ownership return."""
import sys,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from senior_ai.phase10_observer import EvaluationObserver
from senior_ai.navigation_runtime import drive
from senior_ai.navigation_io import read

class SetupFault(EvaluationObserver):
    def setup(self,*args):
        super().setup(*args)
        raise RuntimeError('intentional_evaluation_sensor_setup_failure_before_return')

def main():
    route=read(ROOT/'configs/phase89_routes_v1.json')['routes'][0]
    report=drive(route,ROOT/'artifacts/runs/phase10-drive-setup-fault-v1-001',threading.Event(),audit_sink=SetupFault())
    assert report['status']=='failed' and report['checks']['cleanup'] and report['checks']['cleanup_stopped']
    assert 'intentional_evaluation' in report['errors'][0]
    print({'status':'expected_setup_fault_stopped_and_owned_sensors_cleaned','checks':report['checks']})

if __name__=='__main__':main()
