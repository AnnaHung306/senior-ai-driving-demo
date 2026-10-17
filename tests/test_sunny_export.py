"""Result exports must not mix model versions or use stale timing receipts."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from export_sunny_results import validate_benchmark,final_receipt,export


class SunnyExportTests(unittest.TestCase):
    def setUp(self):
        self.hashes={'detection':'det-new','segmentation':'seg-new','light_state':'phase-new'}
        self.inputs={'input_manifest_sha256':'manifest','input_RGB_sha256':{str(i):'RGB'+str(i) for i in range(8)}}
        self.integrated={**self.inputs,'detector_sha256':'det-new','segmentation_sha256':'seg-new','phase_sha256':'phase-new',
                         'frame':123,'cameras':8,'warmup_bundles':3,'measured_bundles':2,
                         'eight_view_RGB_decode_to_CPU_perception_ms':{'samples':[100.,120.]}}

    def test_integrated_benchmark_binds_every_model(self):
        self.assertEqual(validate_benchmark(self.integrated,self.hashes,123,self.inputs),'integrated')
        for key in ['detector_sha256','segmentation_sha256','phase_sha256']:
            value=copy.deepcopy(self.integrated);value[key]='old-model'
            with self.assertRaisesRegex(ValueError,'tuple mismatch'):validate_benchmark(value,self.hashes,123,self.inputs)

    def test_integrated_benchmark_rejects_other_bundle_and_partial_measurements(self):
        with self.assertRaises(ValueError):validate_benchmark(self.integrated,self.hashes,124,self.inputs)
        for key,value in [('cameras',7),('measured_bundles',3),('warmup_bundles',0)]:
            changed=copy.deepcopy(self.integrated);changed[key]=value
            with self.assertRaises(ValueError):validate_benchmark(changed,self.hashes,123,self.inputs)
        changed=copy.deepcopy(self.integrated);changed['eight_view_RGB_decode_to_CPU_perception_ms']['samples']=[float('nan'),120.]
        with self.assertRaises(ValueError):validate_benchmark(changed,self.hashes,123,self.inputs)

    def test_same_frame_id_cannot_hide_different_RGB_bytes(self):
        changed=copy.deepcopy(self.integrated);changed['input_RGB_sha256']['0']='different-RGB'
        with self.assertRaisesRegex(ValueError,'RGB input mismatch'):validate_benchmark(changed,self.hashes,123,self.inputs)
        changed=copy.deepcopy(self.integrated);changed['input_manifest_sha256']='another-manifest'
        with self.assertRaisesRegex(ValueError,'RGB input mismatch'):validate_benchmark(changed,self.hashes,123,self.inputs)

    def test_individual_task_cannot_reuse_other_models_hash(self):
        value={**self.inputs,'task':'detection','checkpoint_sha256':'seg-new','samples':50,'warmup':10}
        with self.assertRaisesRegex(ValueError,'task/checkpoint'):validate_benchmark(value,self.hashes,123,self.inputs)
        with self.assertRaises(ValueError):validate_benchmark({},self.hashes,123,self.inputs)
        value.update(checkpoint_sha256='det-new',network_and_result_to_cpu_ms={'p50':1.,'p95':2.},
                     warm_filesystem_RGB_manifest_to_result_ms={'p50':2.,'p95':3.})
        self.assertEqual(validate_benchmark(value,self.hashes,123,self.inputs),'detection')

    def test_wrong_integrated_model_fails_before_creating_export_directory(self):
        parent=Path(__file__).resolve().parents[1]/'artifacts/runs/test-temp';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            root=Path(directory);runs={};hashes={}
            for name in ['detection','segmentation','light_state']:
                run=root/name;run.mkdir();runs[name]=run
                checkpoint=name.encode();(run/'best.pt').write_bytes(checkpoint)
                (run/'validation.json').write_bytes(b'{}');(run/'validation_frames.jsonl').write_bytes(b'{}\n')
                hashes[name]=hashlib.sha256(checkpoint).hexdigest()
                state={'status':'completed','best_sha256':hashes[name],'checkpoint_reload_identical':True,
                       'validation_sha256':hashlib.sha256(b'{}').hexdigest(),'validation_frames_sha256':hashlib.sha256(b'{}\n').hexdigest()}
                (run/'state.json').write_text(json.dumps(state),encoding='utf-8')
            demo=root/'demo';demo.mkdir()
            (demo/'perception.json').write_text(json.dumps({**self.inputs,'source':'learned_prediction','model_inputs':'RGB_only',
                'cameras':{str(i):{} for i in range(8)},'frame':123,'detector_sha256':hashes['detection'],
                'segmentation_sha256':hashes['segmentation'],'phase_sha256':hashes['light_state']}),encoding='utf-8')
            value=copy.deepcopy(self.integrated);value.update(detector_sha256=hashes['detection'],segmentation_sha256=hashes['segmentation'])
            benchmark=root/'benchmark.json';benchmark.write_text(json.dumps(value),encoding='utf-8')
            output=root/'export'
            args=SimpleNamespace(output=output,detector_run=runs['detection'],segmentation_run=runs['segmentation'],
                                 light_run=runs['light_state'],demo=demo,benchmark=[benchmark],root=root)
            with self.assertRaisesRegex(ValueError,'tuple mismatch'):export(args)
            self.assertFalse(output.exists())

    def test_stale_final_or_unbound_pass_cannot_be_exported_as_passed(self):
        parent=Path(__file__).resolve().parents[1]/'artifacts/runs/test-temp';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            run=Path(directory);state={'best_sha256':'new'}
            self.assertEqual(final_receipt(run,state),{'status':'not_evaluated'})
            (run/'final_evaluation.json').write_text(json.dumps({'status':'passed','checkpoint_sha256':'old'}),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'another checkpoint'):final_receipt(run,state)
            (run/'final_evaluation.json').write_text(json.dumps({'status':'passed','checkpoint_sha256':'new'}),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'marker'):final_receipt(run,state)
            (run/'lockbox_started.json').write_text(json.dumps({'best_sha256':'new'}),encoding='utf-8')
            self.assertEqual(final_receipt(run,state)['status'],'passed')


if __name__=='__main__':unittest.main()
