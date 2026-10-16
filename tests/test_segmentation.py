"""Metric edge cases, label contract, integrity and checkpoint determinism."""
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from senior_ai.segmentation_metrics import confusion, metrics, prior_baselines
from senior_ai.segmentation_model import TinyUNet, map_target
from senior_ai.segmentation_training import fingerprints, load_checkpoint, restore_rng, save_checkpoint, seed_all, sha, weighted_loss


class SegmentationTests(unittest.TestCase):
    def test_confusion_ignore_and_hand_computed_metrics(self):
        target = np.array([0, 1, 1, 2, 255])
        pred = np.array([0, 1, 2, 2, 99])
        matrix = confusion(target, pred, 4)
        result = metrics(matrix, ['other', 'a', 'b', 'absent'])
        self.assertEqual(matrix.tolist(), [[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
        self.assertEqual(result['pixel_accuracy'], 0.75)
        self.assertEqual(result['focus_macro_iou'], 0.5)
        self.assertIsNone(result['classes'][3]['iou'])
        self.assertEqual(result['focus_supported_classes'], 2)
        with self.assertRaises(ValueError):
            confusion(np.array([1]), np.array([8]))

    def test_absent_class_false_positive_is_not_perfect(self):
        result = metrics(confusion(np.array([0]), np.array([1]), 2), ['other', 'rare'])
        self.assertEqual(result['classes'][1]['iou'], 0)
        self.assertIsNone(result['classes'][1]['recall'])
        self.assertIsNone(result['focus_macro_iou'])

    def test_baselines_use_training_prior_only(self):
        result = prior_baselines([1, 9], [9, 1], ['other', 'focus'])
        self.assertEqual(result['constant_train_majority']['pixel_accuracy'], 0.1)
        self.assertAlmostEqual(result['train_prior_expected_counts']['pixel_accuracy'], 0.18)

    def test_mapping_keeps_single_pixel_small_objects_and_unknown(self):
        config = json.loads((ROOT / 'configs/segmentation_v1.json').read_text())
        raw = np.array([[0, 1, 2, 12, 14, 18, 19, 24, 28]], np.uint8)
        self.assertEqual(map_target(raw, config).tolist(), [[255, 1, 2, 3, 4, 5, 6, 7, 0]])
        with self.assertRaises(ValueError):
            map_target(np.array([29], np.uint8), config)

    def test_deterministic_loss_matches_weighted_mean_and_ignores_unknown(self):
        logits = torch.tensor([[[[1., 2.]], [[3., 4.]]]], requires_grad=True)
        target = torch.tensor([[[0, 255]]])
        weights = torch.tensor([2., 3.])
        expected = torch.nn.functional.cross_entropy(logits, target, weight=weights, ignore_index=255)
        actual = weighted_loss(logits, target, weights)
        self.assertTrue(torch.allclose(actual, expected))
        actual.backward()
        self.assertEqual(float(logits.grad[..., 1].abs().sum()), 0)
        with self.assertRaises(ValueError):
            weighted_loss(logits, torch.full_like(target, 255), weights)

    def test_model_odd_size_and_gradient(self):
        torch.set_num_threads(2)
        model = TinyUNet(width=4)
        x = torch.rand(2, 3, 33, 65)
        output = model(x)
        self.assertEqual(tuple(output.shape), (2, 8, 33, 65))
        torch.nn.functional.cross_entropy(output, torch.ones(2, 33, 65, dtype=torch.long)).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_integrity_rejects_corruption_and_cross_split_duplicates(self):
        temp = ROOT / 'artifacts/runs/test-temp'
        temp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp) as folder:
            root = Path(folder)
            (root / 'plan.json').write_text(json.dumps({'scenes': [
                {'id': 'a', 'split': 'train'}, {'id': 'b', 'split': 'test'}]}))
            paths = ['inputs/rgb/a/x.png', 'inputs/rgb/b/x.png']
            inventory = []
            for i, name in enumerate(paths):
                path = root / name
                path.parent.mkdir(parents=True)
                path.write_bytes(bytes([i]))
                inventory.append({'path': name, 'bytes': 1, 'sha256': sha(path)})
            def write_inventory():
                (root / 'image_inventory.jsonl').write_text(''.join(json.dumps(i) + '\n' for i in inventory))
                (root / 'audit.json').write_text(json.dumps({'status': 'passed', 'checked_pngs': 2,
                    'inventory_sha256': sha(root / 'image_inventory.jsonl')}))
            write_inventory()
            (root / 'report.json').write_text(json.dumps({'status': 'passed', 'total_images': 2}))
            self.assertEqual(fingerprints(root, ROOT)['verified_images'], 2)
            (root / paths[1]).write_bytes(b'Z')
            with self.assertRaisesRegex(ValueError, 'integrity'):
                fingerprints(root, ROOT)
            (root / paths[1]).write_bytes(b'\x00')
            inventory[1]['sha256'] = inventory[0]['sha256']
            write_inventory()
            with self.assertRaisesRegex(ValueError, 'Cross-split'):
                fingerprints(root, ROOT)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA checkpoint RNG resume requires GPU')
    def test_epoch_boundary_checkpoint_repeats_next_update_exactly(self):
        from senior_ai.segmentation_training import configure_cuda, train_batch
        configure_cuda()
        seed_all(42)
        model = TinyUNet(width=4).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        scaler = torch.amp.GradScaler('cuda')
        weights = torch.ones(8, device='cuda')
        x = torch.rand(2, 3, 32, 64, device='cuda')
        y = torch.randint(0, 8, (2, 32, 64), device='cuda')
        train_batch(model, x, y, optimizer, scaler, weights, 5)
        scheduler.step()
        temp = ROOT / 'artifacts/runs/test-temp'
        temp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp) as folder:
            path = Path(folder) / 'checkpoint.pt'
            save_checkpoint(path, model, optimizer, scheduler, scaler, 1, .3, {'seed': 42}, {'data': 'test'})
            expected_random = (random.random(), float(np.random.random()), torch.rand(1).item(), torch.rand(1, device='cuda').item())
            train_batch(model, x, y, optimizer, scaler, weights, 5)
            expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
            state = load_checkpoint(path)
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            scaler.load_state_dict(state['scaler'])
            restore_rng(state['rng'])
            actual_random = (random.random(), float(np.random.random()), torch.rand(1).item(), torch.rand(1, device='cuda').item())
            self.assertEqual(expected_random, actual_random)
            train_batch(model, x, y, optimizer, scaler, weights, 5)
            self.assertTrue(all(torch.equal(expected[k], v) for k, v in model.state_dict().items()))

    @unittest.skipUnless(torch.cuda.is_available(), 'RGB-only production inference uses CUDA')
    def test_prediction_runs_with_only_rgb_and_no_ground_truth(self):
        from PIL import Image
        sys.path.insert(0, str(ROOT / 'scripts'))
        from predict_segmentation import predict
        config = json.loads((ROOT / 'configs/segmentation_v1.json').read_text())
        config.update(width=4, resolution_wh=[64, 32])
        temp = ROOT / 'artifacts/runs/test-temp'
        temp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp) as folder:
            root = Path(folder)
            (root / 'inputs/frames').mkdir(parents=True)
            (root / 'inputs/rgb').mkdir()
            Image.fromarray(np.full((32, 64, 3), 100, np.uint8)).save(root / 'inputs/rgb/a.png')
            frame = {'schema_version': 1, 'source': 'carla_rgb_only', 'scene_id': 'a', 'frame': 7,
                'timestamp_s': .35, 'calibration': 'calibration/a.json',
                'cameras': {str(i): {'image': 'inputs/rgb/a.png', 'frame': 7, 'timestamp_s': .35} for i in range(8)}}
            (root / 'inputs/frames/a.json').write_text(json.dumps(frame))
            torch.save({'config': config, 'model': TinyUNet(4).state_dict(), 'epoch': 0}, root / 'model.pt')
            result = predict(root / 'model.pt', root, 'inputs/frames/a.json', '0', root / 'prediction')
            self.assertEqual(result['source'], 'learned_prediction')
            self.assertFalse((root / 'labels').exists())
            confidence = np.load(root / 'prediction/uncalibrated_softmax_confidence.npy', allow_pickle=False)
            self.assertEqual(confidence.shape, (32, 64))
            self.assertTrue(np.isfinite(confidence).all())
