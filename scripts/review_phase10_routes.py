"""Render all attempted routes with readable equal-scale map axes."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    output = ROOT / 'artifacts/runs/phase10-route-visual-v2-001'
    output.mkdir(exist_ok=False)
    report = read(ROOT / 'artifacts/runs/phase10-evaluation-final-v1-002/report.json')
    plt.style.use('dark_background')
    panels = []
    for index, run in enumerate(report['runs']):
        folder = ROOT / 'artifacts/runs' / run['run_id']
        saved = read(folder / 'report.json')
        records = [read(folder / row['file']) for row in saved['records']]
        if not records:
            continue
        path = np.array(saved['route']['points_xy_m'])
        actual = np.array([r['ego_pose_navigation_prior'][:2] for r in records])
        both = np.vstack([path, actual])
        center = (both.max(axis=0) + both.min(axis=0)) / 2
        radius = max(float(np.ptp(both, axis=0).max()) * .55, 10.)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
        axes[0].plot(path[:, 0], path[:, 1], label='Map route prior')
        axes[0].plot(actual[:, 0], actual[:, 1], label='Ego localization prior')
        axes[0].set_xlim(center[0]-radius, center[0]+radius)
        axes[0].set_ylim(center[1]-radius, center[1]+radius)
        axes[0].set_aspect('equal')
        axes[0].set_xlabel('World x m'); axes[0].set_ylabel('World y m')
        axes[0].legend(fontsize=7)
        times = [r['simulation_s'] for r in records]
        axes[1].plot(times, [r['speed_mps'] for r in records], label='Actual speed')
        axes[1].plot(times, [r['plan']['target_speed_mps'] for r in records], label='Target speed')
        axes[1].set_xlabel('Simulation seconds; final cleanup excluded')
        axes[1].set_ylabel('m/s'); axes[1].legend(fontsize=7)
        fig.suptitle(run['run_id'] + ' | ' + run['status'], fontsize=10)
        name = f'route-{index:02d}.png'
        fig.savefig(output / name, dpi=120); plt.close(fig)
        panels.append(name)
    for start in range(0, len(panels), 4):
        page = Image.new('RGB', (1440, 4*510), '#111b26')
        draw = ImageDraw.Draw(page)
        for j, name in enumerate(panels[start:start+4]):
            with Image.open(output/name) as image:
                page.paste(image.convert('RGB'), (0, j*510+25))
            draw.text((8, j*510+5), name, fill='white')
        page.save(output/f'contact-{start//4}.jpg', quality=90)
    (output/'catalog.json').write_text(json.dumps({'status':'pending_actual_review',
        'panels':panels,'all_attempts':len(report['runs']),
        'zero_frame_setup_fault_has_no_curve':True,
        'reason_for_v2':'Original equal-aspect narrow straight routes collapsed map panel; expand equal metric limits.'},indent=2)+'\n',encoding='utf8')
    print({'panels':len(panels),'output':str(output)})


if __name__ == '__main__':
    main()
