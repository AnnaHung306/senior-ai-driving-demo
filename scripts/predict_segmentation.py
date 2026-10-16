"""RGB-only learned prediction. Does not open dataset labels/audit/plan files."""
import argparse
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from PIL import Image
import torch
from senior_ai.core import write_json
from senior_ai.dataset import read_sensor_frame
from senior_ai.segmentation_data import load_inference_rgb
from senior_ai.segmentation_model import TinyUNet, colorize
from senior_ai.segmentation_training import configure_cuda, load_checkpoint, sha


def predict(checkpoint, root, manifest, camera, output):
    configure_cuda()
    state = load_checkpoint(checkpoint)
    config = state['config']
    model = TinyUNet(config['width']).cuda().eval()
    model.load_state_dict(state['model'])
    frame = read_sensor_frame(root, manifest)
    rgb = load_inference_rgb(root, manifest, camera)
    if rgb.shape[:2][::-1] != tuple(config['resolution_wh']):
        raise ValueError('Input shape differs from model contract')
    x = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0).cuda().float() / 255
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        logits = model(x).float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError('Invalid model output')
    ids = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    confidence = logits.softmax(1).amax(1)[0].cpu().numpy()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    output.mkdir(parents=True, exist_ok=False)
    Image.fromarray(ids).save(output / 'class_ids.png')
    Image.fromarray(colorize(ids)).save(output / 'prediction.png')
    Image.fromarray(rgb).save(output / 'rgb.png')
    np.save(output / 'uncalibrated_softmax_confidence.npy', confidence, allow_pickle=False)
    receipt = {'schema_version': 1, 'source': 'learned_prediction', 'model_version': config['version'],
        'checkpoint_sha256': sha(checkpoint), 'checkpoint_epoch': state['epoch'], 'frame': frame['frame'],
        'timestamp_s': frame['timestamp_s'], 'camera': camera, 'calibration_reference': frame['calibration'],
        'input_manifest': str(manifest), 'input_rgb_sha256': sha(root / frame['cameras'][camera]['image']),
        'class_names': config['class_names'], 'class_id_encoding': 'model IDs 0..7; not raw CARLA IDs',
        'elapsed_forward_and_cpu_output_s_no_warmup': elapsed,
        'confidence': 'uncalibrated max softmax; not safety probability',
        'outputs': ['class_ids.png', 'prediction.png', 'rgb.png', 'uncalibrated_softmax_confidence.npy']}
    write_json(output / 'prediction.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['checkpoint', 'root', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--camera', required=True)
    a = parser.parse_args()
    print(predict(a.checkpoint.resolve(), a.root.resolve(), a.manifest, a.camera, a.output.resolve()))
