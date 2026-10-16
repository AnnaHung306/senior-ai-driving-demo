"""Turn per-frame evaluation counts into a deterministic FP/FN triage database."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from senior_ai.core import write_json


def catalog(run):
    config = json.loads((run / 'config.json').read_text())
    output = {'source': 'ground_truth_for_error_analysis_only', 'checkpoint': 'best.pt',
              'ranking': 'top 3 FP pixels and FN pixels per split/class, stable frame order on ties', 'cases': [],
              'semantic_support_strata': {},
              'strata_note': 'GT pixels of a class per image: absent, 1..100, 101..1000, >1000. '
                             'This is not instance size, occlusion annotation, small-object AP or an acceptance threshold.'}
    for split in ['validation', 'test']:
        frames = [json.loads(line) for line in (run / f'{split}_errors/frames.jsonl').read_text().splitlines()]
        output['semantic_support_strata'][split] = {}
        for cls, name in enumerate(config['class_names'][1:], start=1):
            cases = []
            strata = {key: {'images': 0, 'support_pixels': 0, 'predicted_pixels': 0, 'true_positive_pixels': 0}
                      for key in ['absent', '1..100', '101..1000', '>1000']}
            for f in frames:
                c = f['classes'][cls]
                true_positive = round((c['recall'] or 0) * c['support_pixels'])
                support = c['support_pixels']
                key = 'absent' if support == 0 else '1..100' if support <= 100 else '101..1000' if support <= 1000 else '>1000'
                strata[key]['images'] += 1
                strata[key]['support_pixels'] += support
                strata[key]['predicted_pixels'] += c['predicted_pixels']
                strata[key]['true_positive_pixels'] += true_positive
                cases.append({k: f[k] for k in ['scene_id', 'group_id', 'frame', 'timestamp_s', 'camera', 'weather', 'input_manifest', 'rgb', 'semantic']} |
                    {'class': name, 'split': split, 'false_positive_pixels': c['predicted_pixels'] - true_positive,
                     'false_negative_pixels': c['support_pixels'] - true_positive, 'metrics': c})
            for counts in strata.values():
                tp, support, predicted = counts['true_positive_pixels'], counts['support_pixels'], counts['predicted_pixels']
                counts['iou'] = tp / (support + predicted - tp) if support + predicted - tp else None
                counts['recall'] = tp / support if support else None
                counts['precision'] = tp / predicted if predicted else None
            output['semantic_support_strata'][split][name] = strata
            for error in ['false_positive_pixels', 'false_negative_pixels']:
                for case in sorted(cases, key=lambda c: -c[error])[:3]:
                    if case[error] > 0:
                        output['cases'].append({**case, 'ranked_by': error})
    write_json(run / 'error_catalog.json', output)
    return len(output['cases'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    print(catalog(parser.parse_args().run.resolve()))
