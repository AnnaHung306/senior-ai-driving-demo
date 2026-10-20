import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from senior_ai.bev_geometry import Grid
from senior_ai.world_display import relative_path, read, sha, boundary_segments
from senior_ai.world_viewer_service import ViewerBundle, FeedInbox, publish_existing_frame


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.root = Path(self.temp.name)
        (self.root/'frames').mkdir();(self.root/'frames/a.json').write_text('{}')
        digest = sha(self.root/'frames/a.json')
        self.manifest = {'schema_version': 1, 'source_mode': 'recorded_inference_replay', 'control_issued': False,
                         'GT_geometry_exported': False, 'formal_model_quality_passed': False, 'run_contract_sha256': 'a'*64,
                         'frames': [{'key': 's/1', 'scene_id': 's', 'frame': 1, 'timestamp_s': 1., 'file': 'frames/a.json', 'sha256': digest},
                                    {'key': 's/2', 'scene_id': 's', 'frame': 2, 'timestamp_s': 1.2, 'file': 'frames/a.json', 'sha256': digest}],
                         'scenes': [{'id': 's', 'frames': ['s/1', 's/2']}], 'file_inventory': {'frames/a.json': digest}}
        self.save();self.bundle = ViewerBundle(self.root);self.feed = self.root/'feed'
    def tearDown(self):self.temp.cleanup()
    def save(self): (self.root/'bundle.json').write_text(json.dumps(self.manifest), encoding='utf-8')
    def pointer(self, **changes):
        value = publish_existing_frame(self.bundle, self.feed, 's/1');value.update(changes)
        (self.feed/'latest.json').write_text(json.dumps(value));return value
    def test_paths_reject_traversal_windows_and_noncanonical(self):
        for p in ['../labels/a', '/labels/a', 'C:/x', 'frames\\a', 'frames//a', 'frames/./a', 'frames/%2e%2e/x', '']:
            with self.subTest(p=p), self.assertRaises(ValueError):relative_path(p)
        self.assertEqual(str(relative_path('frames/a.json')), 'frames/a.json')
    def test_nonfinite_json_rejected(self):
        for v in ['NaN', 'Infinity', '-Infinity']:
            p = self.root/'bad.json';p.write_text('{"value":'+v+'}')
            with self.assertRaises(ValueError):read(p)
    def test_bundle_rejects_changed_file_at_load_and_request(self):
        (self.root/'frames/a.json').write_text('{"changed":true}')
        with self.assertRaises(ValueError):ViewerBundle(self.root)
        with self.assertRaises(ValueError):self.bundle.file('frames/a.json')
    def test_unlisted_workspace_not_served(self):
        with self.assertRaises(FileNotFoundError):self.bundle.file('labels/a.json')
        with self.assertRaises(ValueError):self.bundle.file('../PHASE5.md')
    def test_provenance_and_quality_claims_rejected(self):
        for key in ['control_issued', 'GT_geometry_exported', 'formal_model_quality_passed']:
            original = copy.deepcopy(self.manifest);self.manifest[key] = True;self.save()
            with self.assertRaises(ValueError):ViewerBundle(self.root)
            self.manifest = original
    def test_duplicate_and_reversed_frames_rejected(self):
        self.manifest['frames'][1] = copy.deepcopy(self.manifest['frames'][0]);self.save()
        with self.assertRaises(ValueError):ViewerBundle(self.root)
        self.manifest['frames'][1].update(key='s/2', frame=2, timestamp_s=.9);self.save()
        with self.assertRaises(ValueError):ViewerBundle(self.root)
    def test_feed_waiting_ready_stale(self):
        inbox = FeedInbox(self.feed, self.bundle);self.assertEqual(inbox.latest()['status'], 'waiting')
        self.pointer();self.assertEqual(inbox.latest()['status'], 'ready')
        self.pointer(published_utc=(datetime.now(timezone.utc)-timedelta(seconds=4)).isoformat())
        self.assertEqual(inbox.latest()['status'], 'stale')
    def test_feed_wrong_source_gt_and_future_rejected(self):
        cases = [{'source_contract_sha256': 'b'*64}, {'key': 'unknown/4'}, {'actor_id': 7},
                 {'published_utc': datetime.now().isoformat()},
                 {'published_utc': (datetime.now(timezone.utc)+timedelta(seconds=30)).isoformat()}]
        for changes in cases:
            self.pointer(**changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):FeedInbox(self.feed, self.bundle).latest()
    def test_feed_time_reversal_rejected_and_can_recover_forward(self):
        inbox = FeedInbox(self.feed, self.bundle);publish_existing_frame(self.bundle, self.feed, 's/2');inbox.latest()
        self.pointer()
        with self.assertRaises(ValueError):inbox.latest()
        publish_existing_frame(self.bundle, self.feed, 's/2');self.assertEqual(inbox.latest()['status'], 'ready')
    def test_unverified_frame_cannot_be_published(self):
        with self.assertRaises(ValueError):publish_existing_frame(self.bundle, self.feed, 's/999')
    def test_windows_transient_read_lock_retries_atomic_replace(self):
        replace = Path.replace;calls = []
        def locked_once(source, target):
            calls.append(target)
            if len(calls)==1:raise PermissionError('Transient reader lock')
            return replace(source,target)
        with patch.object(Path,'replace',locked_once):publish_existing_frame(self.bundle,self.feed,'s/1')
        self.assertEqual(len(calls),2);self.assertEqual(FeedInbox(self.feed,self.bundle).latest()['status'],'ready')
        self.assertEqual(list(self.feed.glob('*.pending.json')),[])
    def test_interfaces_only_actual_road_sidewalk_adjacency(self):
        grid = Grid(0, 2, 0, 2, 1)
        self.assertEqual(boundary_segments(np.array([[1,255],[255,2]]), grid), [])
        self.assertEqual(boundary_segments(np.array([[1,2],[255,255]]), grid), [[[2.,1.],[1.,1.]]])
        self.assertEqual(boundary_segments(np.array([[1,255],[2,255]]), grid), [[[1.,0.],[1.,1.]]])


if __name__ == '__main__':unittest.main()
