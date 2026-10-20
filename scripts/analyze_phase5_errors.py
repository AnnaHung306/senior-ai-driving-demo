"""Supplementary common-match / inherited-ignore audit; never changes raw scores."""
import argparse
from collections import Counter
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1];sys.path.insert(0, str(ROOT/'src'))
from phase5_gt import read, sha
from senior_ai.core import write_json
from senior_ai.detection_metrics import iou
from senior_ai.visibility_policy import ignore_background_proposals
from evaluate_phase5_world import metrics_objects, match_objects


def ignore_objects(objects, truth, visible):
    # A known positive wins over ignore evidence, in2D or3D. Every source must
    # support ignore before a fused proposal may be omitted from this audit.
    protected = {i for i, _ in match_objects(objects, truth)} | {i for i, _ in match_objects(objects, truth, 'IoU')}
    retained = [];ignored = []
    for i, obj in enumerate(objects):
        source_status = []
        for source in obj['sources']:
            label = visible[source['camera']];box = source['bbox_xyxy']
            ignored_2d = ignore_background_proposals([box], [0], [r['bbox_xyxy'] for r in label['ignore_regions']])[0] == -1
            positive_overlap = max(iou(box, [r['visible_bbox_xyxy'] for r in label['positive_objects']]), default=0.)
            source_status.append(bool(ignored_2d and positive_overlap < .5))
        if i not in protected and source_status and all(source_status):ignored.append(i)
        else:retained.append(obj)
    return retained, ignored


def main(args):
    report = read(args.evaluation/'report.json');data = read(args.crops/'contract.json');result = {};checked = {}
    for split in ['train', 'validation']:
        rows = [r for r in report['records'] if r['scene_id'] in data['split_summaries'][split]['scenes']]
        shared = {n: [] for n in ['learned', 'prior']};counts = {n: Counter() for n in shared};ignored = Counter()
        for row in rows:
            maps = {n: {p['GT_actor_id']: error for p, error in zip(row[n]['pairs'], row[n]['matched_errors'])} for n in shared}
            for actor in maps['learned'].keys() & maps['prior'].keys():
                for name in shared:shared[name].append(maps[name][actor])
            world = read(args.run/row['folder']/'world.json');visible = {}
            for camera in ['front', 'front_right', 'right', 'rear_right', 'rear', 'rear_left', 'left', 'front_left']:
                relative = f"labels/visible/{row['scene_id']}/{camera}/{row['frame']:08}.json"
                path = args.root/relative;digest = sha(path)
                if read(args.evaluation/'GT_input_inventory.json')[relative] != digest:raise ValueError('GT visible label changed')
                checked[relative] = digest;visible[camera] = read(path)
            for name, key in [('learned', 'objects'), ('prior', 'prior_baseline_objects')]:
                objects = [o for o in world[key] if np.linalg.norm(o['center_ego_m'][:2]) <= 25]
                retained, excluded = ignore_objects(objects, row['GT_actors'], visible);ignored[name] += len(excluded)
                count, _, _ = metrics_objects(retained, row['GT_actors']);counts[name].update(count)
        result[split] = {'shared_GT_frame_pairs': len(shared['learned']),
                         'common_match_errors': {n: {k: np.asarray([x[k] for x in errors]).mean(0).tolist() for k in errors[0]} if errors else None for n, errors in shared.items()},
                         'inherited_2D_ignore_aware_supplement': {n: {**dict(c), 'ignored_fused_proposals': ignored[n],
                            'precision_at_2m': c['TP']/(c['TP']+c['FP']) if c['TP']+c['FP'] else None,
                            'recall_at_2m': c['TP']/(c['TP']+c['FN']) if c['TP']+c['FN'] else None} for n, c in counts.items()}}
    write_json(args.output, {'status': 'completed_supplementary_audit_not_quality_pass', 'raw_evaluation_not_modified': True,
               'evaluation_report_sha256': sha(args.evaluation/'report.json'), 'tool_sha256': sha(Path(__file__)), 'by_split': result,
               'policy': 'Supplement only: existing2D ignore IoA>=0.5 or center rule, all sources ignored, known2D or3D positives override. Strict raw3D report retained.',
               'common_match_scope': 'Shared actor-frame pairs class/XY matched within2m by BOTH methods; conditional selection, not full recall or accuracy',
               'verified_visible_GT_files': checked})
    print(json_output(result))


def json_output(value):
    import json
    return json.dumps(value)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['root', 'run', 'evaluation', 'crops', 'output']:p.add_argument('--'+name, type=Path, required=True)
    main(p.parse_args())
