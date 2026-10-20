"""Independent GT evaluation of completed RGB WorldFrames; no feedback to models."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[1];sys.path.insert(0, str(ROOT/'src'))
from senior_ai.bev_geometry import cameras_from_calibration
from senior_ai.cuboid import CLASSES, corners, EDGES, box_iou, wrap
from senior_ai.semantic_occupancy import VoxelGrid, LABELS
from senior_ai.core import write_json
from phase5_gt import read, sha, target_box, audited_inventory
from select_phase4_pilot import gt_points
from phase5_render import iso_point, COLORS


def ratio(a, b):return float(a/b) if b else None


def match_objects(predictions, truth, criterion='distance'):
    candidates = []
    for i, p in enumerate(predictions):
        for j, t in enumerate(truth):
            if p['class_id'] != t['class_id']:continue
            if criterion == 'distance':
                score = float(np.linalg.norm(np.array(p['center_ego_m'])[:2]-np.array(t['center_ego_m'])[:2]))
                if score <= 2:candidates.append((score, i, j))
            else:
                score = box_iou(p, t)[1]
                if score >= .25:candidates.append((-score, i, j))
    used_p = set();used_t = set();pairs = []
    for _, i, j in sorted(candidates):
        if i not in used_p and j not in used_t:
            used_p.add(i);used_t.add(j);pairs.append((i, j))
    return pairs


def regression_error(prediction, target):
    d = np.asarray(prediction['center_ego_m'])-target['center_ego_m'];a = float(abs(wrap(prediction['yaw_rad']-target['yaw_rad'])))
    bev, iou3d = box_iou(prediction, target)
    return {'center_3D_m': float(np.linalg.norm(d)), 'center_XY_m': float(np.linalg.norm(d[:2])), 'center_Z_m': float(abs(d[2])),
            'size_LWH_absolute_m': abs(np.array(prediction['size_lwh_m'])-target['size_lwh_m']).tolist(),
            'yaw_absolute_deg': float(np.rad2deg(a)), 'yaw_pi_symmetric_deg': float(np.rad2deg(min(a, np.pi-a))), 'BEV_IoU': bev, 'IoU3D': iou3d}


def merge_audit(objects, visible):
    def iou(a, b):
        overlap = max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))
        return overlap/max(1e-9, (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap)
    counts = Counter()
    for obj in objects:
        identities = set();matched = 0;counts['duplicate_observations_removed'] += len(obj['sources'])-1
        for source in obj['sources']:
            options = [(iou(source['bbox_xyxy'], t['visible_bbox_xyxy']), t['actor_id']) for t in visible[source['camera']] if t['category_id'] == obj['class_id']]
            if options and max(options)[0] >= .5:
                identities.add(max(options)[1]);matched += 1
        counts['GT_supported_merged_clusters'] += len(identities) == 1 and matched > 1
        counts['mixed_GT_actor_merges'] += len(identities) > 1
    return counts


def metrics_objects(predictions, truth):
    pairs = match_objects(predictions, truth);iou_pairs = match_objects(predictions, truth, 'IoU')
    result = {'TP': len(pairs), 'FP': len(predictions)-len(pairs), 'FN': len(truth)-len(pairs),
              'TP_IoU3D_025': len(iou_pairs), 'FP_IoU3D_025': len(predictions)-len(iou_pairs), 'FN_IoU3D_025': len(truth)-len(iou_pairs)}
    errors = [regression_error(predictions[i], truth[j]) for i, j in pairs]
    return result, errors, pairs


def occupancy_counts(predicted, truth):
    domain = truth != 255
    p = np.where(predicted == 255, 11, predicted).astype(np.int64)
    if np.any((p < 0) | (p > 11)):raise ValueError('Unsupported occupancy label')
    matrix = np.bincount((truth[domain].astype(np.int64)*12+p[domain]), minlength=11*12).reshape(11, 12)
    return matrix


def occupancy_scores(matrix):
    result = {}
    for category in range(11):
        tp = int(matrix[category, category]);fp = int(matrix[:, category].sum()-tp);fn = int(matrix[category].sum()-tp)
        result[str(category)] = {'name': LABELS.get(category, 'other_static_surface'), 'TP': tp, 'FP': fp, 'FN': fn,
                                 'GT_support': int(matrix[category].sum()), 'IoU': ratio(tp, tp+fp+fn)}
    occupied = int(matrix[1:].sum());domain = int(matrix.sum())
    return {'by_class': result, 'GT_known_voxels': domain, 'predicted_known_on_GT_domain': int(matrix[:, :11].sum()),
            'coverage_on_GT_domain': ratio(matrix[:, :11].sum(), domain), 'GT_occupied_predicted_free': int(matrix[1:, 0].sum()),
            'false_free_rate_on_GT_occupied': ratio(matrix[1:, 0].sum(), occupied), 'GT_free_predicted_occupied': int(matrix[0, 1:11].sum()),
            'GT_occupied_predicted_unknown': int(matrix[1:, 11].sum()), 'unknown_counted_as_FN': True}


def gt_occupancy(root, manifest, grid, truth, label, ego):
    values = np.full(grid.shape, 255, np.uint8);surfaces = [];conflicts = np.zeros(grid.shape, bool)
    inv_ego = np.linalg.inv(ego)
    mapping = {1: 1, 24: 1, 2: 2, 25: 9, 14: 3, 12: 4, 19: 5, 18: 6, 15: 7, 16: 8}
    for name, xyz, semantic, _, _ in gt_points(root, manifest, 8):
        origin = (inv_ego@np.asarray(label['capture_world_from_camera'][name]['depth']))[:3, 3]
        length = np.linalg.norm(xyz-origin, axis=-1)
        valid = np.isfinite(xyz).all(-1) & (length > .3) & (length < 35)
        endpoints = xyz[valid];lengths = length[valid]
        # Real depth rays independently define free space. Stop before surface cell.
        for start in range(0, len(endpoints), 512):
            end = endpoints[start:start+512];dist = lengths[start:start+512];steps = np.arange(0., 35., .25)
            ray = origin+(end[:, None, :]-origin)*np.minimum(steps[None, :]/dist[:, None], 1.)[..., None]
            indices, inside = grid.index(ray);inside &= steps[None, :] < dist[:, None]-grid.resolution_m/2
            selected = indices[inside];values[tuple(selected.T)] = 0
        surface_xyz = xyz[valid].copy();tags = semantic[valid]
        # Declared flat-ground reference: only near-plane ground surfaces snap to z=0.
        ground = np.isin(tags, [1, 2, 24, 25]) & (abs(surface_xyz[:, 2]) <= .2)
        surface_xyz[ground, 2] = 0
        indices, inside = grid.index(surface_xyz)
        categories = np.full(len(tags), 10, np.uint8)
        for raw, category in mapping.items():categories[tags == raw] = category
        surfaces.append((indices[inside], categories[inside]))
    # Apply endpoints after every camera's free rays so free never overwrites surfaces.
    for indices, categories in surfaces:
        values[tuple(indices.T)] = categories
    xyz = grid.centers();actor_labels = np.zeros(grid.shape, np.uint8)
    for actor in truth:
        d = xyz-np.array(actor['center_ego_m']);rotation = np.array(actor['rotation_reference_from_box'])
        inside = (abs(d@rotation) <= np.array(actor['size_lwh_m'])/2).all(-1)
        category = actor['class_id']+2
        conflicts |= inside & (actor_labels != 0) & (actor_labels != category)
        actor_labels[inside] = category
    values[actor_labels != 0] = actor_labels[actor_labels != 0]
    values[conflicts] = 255
    plane = grid.ground.points();values[(abs(plane[..., 0]) < 2.5) & (abs(plane[..., 1]) < 1.1)] = 255
    return values, {'depth_stride': 8, 'flat_ground_snap_tolerance_m': .2,
                    'GT_actor_volume': 'full captured 3D rotation, only positive-visible union actors within25m',
                    'actor_class_conflict_voxels_excluded': int(conflicts.sum()), 'fine_surface_GT_complete': False}


def render_comparison(predicted, truth, objects, gt, grid, caption):
    image = Image.new('RGB', (1280, 620), (18, 23, 32));draw = ImageDraw.Draw(image)
    for offset, values, name in [(0, truth, 'GT reference: real rays + visible actor volumes'), (320, predicted, 'RGB prediction: ground rays + learned cuboids')]:
        colors = np.full(grid.ground.shape+(3,), [24, 28, 36], np.uint8)
        colors[np.any(values == 0, axis=2)] = [48, 53, 63]
        for category, color in [(1, (70, 100, 140)), (2, (130, 110, 90)), (9, (80, 100, 65)), (10, (125, 125, 125))]:
            colors[np.any(values == category, axis=2)] = color
        for category, color in COLORS.items():colors[np.any(values == category+2, axis=2)] = color
        image.paste(Image.fromarray(colors).resize((300, 500), Image.Resampling.NEAREST), (offset, 45))
        draw.text((offset+3, 8), name, fill='white')
    for group, color in [(gt, (90, 240, 130)), (objects, (255, 160, 75))]:
        for obj in group:
            uv = iso_point(corners(obj))*.75+np.array([665, 90])
            for i, j in EDGES:draw.line([tuple(uv[i]), tuple(uv[j])], fill=color, width=2)
    draw.text((650, 8), 'GT upright projection=green; predicted=orange', fill='white')
    draw.text((5, 568), caption, fill='white');draw.text((5, 590), 'Evaluation-only GT. Unknown is dark; free is gray. Occupancy is an approximation, not reconstructed mesh.', fill=(180, 190, 205))
    return image


def main(args):
    root = args.root.resolve();run = args.run.resolve();out = args.output.resolve()
    if out.exists():raise FileExistsError('New independent evaluation output required')
    state = read(run/'state.json');contract = read(run/'contract.json');config = read(run/'config.json');grid = VoxelGrid(**config['occupancy'])
    if state['status'] != 'completed_world_inference_pending_independent_evaluation':raise ValueError('Completed WorldFrames required')
    golden, inventory = audited_inventory(root, ROOT)
    for name, digest in contract['sources'].items():
        if sha(run/'source'/name) != digest:raise ValueError('Frozen inference source changed')
    out.mkdir(parents=True);verified = {};records = [];totals = {name: Counter() for name in ['learned', 'prior']}
    errors = {name: [] for name in totals};class_totals = {n: {c: Counter() for c in CLASSES} for n in totals};class_errors = {n: defaultdict(list) for n in totals}
    scene_totals = {n: defaultdict(Counter) for n in totals};matrix = np.zeros((11, 12), np.int64);scene_matrices = defaultdict(lambda: np.zeros((11, 12), np.int64))
    last_matches = {};tracking = Counter();last_scene = None;scene_index = 0

    def verify(relative):
        if relative not in verified:
            digest = sha(root/relative)
            if inventory.get(relative) != digest:raise ValueError('GT data changed/unbound: '+relative)
            verified[relative] = digest

    for row in state['records']:
        folder = Path(row['folder']);world = read(run/folder/'world.json');manifest = world['parent_manifest'];verify(manifest);frame = read(root/manifest)
        verify(frame['calibration']);cameras = cameras_from_calibration(read(root/frame['calibration']))
        scene = world['scene_id'];number = world['frame'];label_path = f'labels/frames/{scene}/{number:08}.json';verify(label_path);label = read(root/label_path)
        ego = np.asarray(label['capture_world_from_camera']['front']['rgb'])@cameras['front'].invT;inv_ego = np.linalg.inv(ego)
        visible_ids = {};visible_by_camera = {}
        for name in cameras:
            relative = f'labels/visible/{scene}/{name}/{number:08}.json';verify(relative);visible_by_camera[name] = read(root/relative)['positive_objects']
            for obj in visible_by_camera[name]:
                if obj['category_id'] in CLASSES:visible_ids[obj['actor_id']] = obj['category_id']
            for modality in ['depth', 'semantic_segmentation']:verify(label[modality][name]['image'])
        truth = []
        for actor in label['objects_3d']:
            if actor['actor_id'] not in visible_ids:continue
            box = target_box(actor, inv_ego)
            if np.linalg.norm(box['center_ego_m'][:2]) <= 25:
                truth.append({**box, 'class_id': visible_ids[actor['actor_id']], 'class_name': CLASSES[visible_ids[actor['actor_id']]]})
        record = {'scene_id': scene, 'frame': number, 'folder': str(folder), 'GT_actors': truth}
        for name, objects in [('learned', world['objects']), ('prior', world['prior_baseline_objects'])]:
            predicted = [o for o in objects if np.linalg.norm(o['center_ego_m'][:2]) <= 25]
            counts, errs, pairs = metrics_objects(predicted, truth);totals[name].update(counts);errors[name] += errs;scene_totals[name][scene].update(counts)
            audit = merge_audit(predicted, visible_by_camera);totals[name].update(audit);counts.update(audit)
            record[name] = {'counts': counts, 'matched_errors': errs,
                            'pairs': [{'track_id': predicted[i].get('track_id'), 'GT_actor_id': truth[j]['actor_id']} for i, j in pairs]}
            for category in CLASSES:
                cc, ce, _ = metrics_objects([o for o in predicted if o['class_id'] == category], [o for o in truth if o['class_id'] == category])
                class_totals[name][category].update(cc);class_errors[name][category] += ce
        if scene != last_scene:last_matches = {};scene_index = 0;last_scene = scene
        scene_index += 1
        for pair in record['learned']['pairs']:
            actor = pair['GT_actor_id'];track = pair['track_id']
            if actor in last_matches:
                old_track, old_index = last_matches[actor]
                if old_index == scene_index-1:
                    tracking['consecutive_matched_actor_transitions'] += 1
                    tracking['ID_switches'] += track != old_track
                else:
                    tracking['matched_actor_reacquisitions_after_gap'] += 1
                    tracking['changed_ID_after_gap'] += track != old_track
            last_matches[actor] = track, scene_index
        with np.load(run/folder/'occupancy.npz') as package:predicted_grid = package['semantic']
        if sha(run/folder/'occupancy.npz') != world['occupancy']['sha256']:raise ValueError('Occupancy output changed')
        target_grid, scope = gt_occupancy(root, manifest, grid, truth, label, ego)
        counts = occupancy_counts(predicted_grid, target_grid);matrix += counts;scene_matrices[scene] += counts
        record['occupancy_confusion_11x12'] = counts.tolist();record['occupancy_GT_scope'] = scope;records.append(record)
        dest = out/folder;dest.mkdir(parents=True);np.savez_compressed(dest/'GT_reference.npz', semantic=target_grid)
        render_comparison(predicted_grid, target_grid, world['objects'], truth, grid, f'{scene} frame{number} | 3D regression / detections / occupancy errors are evaluated separately').save(dest/'evaluation.png')
        print(f'evaluated {scene}/{number}', flush=True)

    def aggregate(counts, errs):
        result = dict(counts)
        result.update({'precision_at_2m': ratio(counts['TP'], counts['TP']+counts['FP']), 'recall_at_2m': ratio(counts['TP'], counts['TP']+counts['FN']),
                       'precision_IoU3D_025': ratio(counts['TP_IoU3D_025'], counts['TP_IoU3D_025']+counts['FP_IoU3D_025']),
                       'recall_IoU3D_025': ratio(counts['TP_IoU3D_025'], counts['TP_IoU3D_025']+counts['FN_IoU3D_025']),
                       'errors_conditional_on_2m_matches': {key: np.asarray([r[key] for r in errs]).mean(0).tolist() for key in errs[0]} if errs else None})
        return result
    report = {'status': 'completed_baseline_evaluation_pending_visual_review', 'scope': '32 known short train/validation bundles, not independent heldout or formal quality pass',
              'objects': {name: aggregate(totals[name], errors[name]) for name in totals},
              'by_class': {n: {str(c): aggregate(class_totals[n][c], class_errors[n][c]) for c in CLASSES} for n in totals},
              'by_scene': {n: {s: aggregate(v, []) for s, v in scene_totals[n].items()} for n in totals},
              'tracking': {'ID_switches': 0, 'changed_ID_after_gap': 0, **dict(tracking), 'GT_identity_used_in_inference': False,
                           'scope': 'Only consecutive correctly class/position matched transitions; misses and FP remain in object P/R'},
              'occupancy': occupancy_scores(matrix), 'occupancy_confusion_11x12': matrix.tolist(),
              'occupancy_by_scene': {s: occupancy_scores(m) for s, m in scene_matrices.items()},
              'timing_ms': {key: {'p50': float(np.quantile([read(run/Path(r['folder'])/'world.json')['timing_ms'][key] for r in state['records']], .5)),
                                  'p95': float(np.quantile([read(run/Path(r['folder'])/'world.json')['timing_ms'][key] for r in state['records']], .95))}
                            for key in ['decode_crops_3D_fusion_tracking', 'occupancy', 'upstream_Phase4_compute']},
              'metric_limits': ['IoU3D compares upright predicted and yaw-projected GT boxes; GT occupancy actor volume retains full captured3D rotation',
                                'GT occupancy is sparse real surface/ray evidence plus positive-visible actor cuboid volume, not complete geometry',
                                'Ground within0.2m is snapped to z0; flat-ground IoU is not a measured curb-height score',
                                'Prior and learned each use their own fixed25m center gate; all positive GT actors in that range count even if head cannot project them',
                                'Validation shapes reuse the same simulator assets; low dimension MAE is not unseen-shape generalization'],
              'run_contract_sha256': sha(run/'contract.json'), 'evaluator_sha256': sha(Path(__file__)),
              'GT_helper_sha256': sha(ROOT/'scripts/phase5_gt.py'), 'depth_helper_sha256': sha(ROOT/'scripts/select_phase4_pilot.py'),
              'approved_parent_data': golden, 'verified_GT_files': len(verified), 'records': records}
    write_json(out/'GT_input_inventory.json', verified);write_json(out/'report.json', report)
    for name in ['scripts/evaluate_phase5_world.py', 'scripts/phase5_gt.py', 'scripts/select_phase4_pilot.py', 'scripts/phase5_render.py']:
        dest = out/'source'/name;dest.parent.mkdir(parents=True, exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    print({k: report[k] for k in ['status', 'objects', 'tracking']}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--root', type=Path, required=True);p.add_argument('--run', type=Path, required=True);p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
