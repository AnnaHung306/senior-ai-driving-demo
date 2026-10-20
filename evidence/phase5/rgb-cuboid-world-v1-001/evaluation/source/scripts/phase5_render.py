"""Code-rendered prediction views; no labels or simulator access."""
import numpy as np
from PIL import Image, ImageDraw
from senior_ai.cuboid import corners, EDGES

COLORS = {1: (110, 175, 255), 2: (255, 125, 200), 3: (100, 235, 160), 4: (255, 180, 75), 5: (180, 150, 255), 6: (120, 230, 230)}


def camera_overlay(rgb, camera, objects):
    image = Image.fromarray(rgb).copy();draw = ImageDraw.Draw(image)
    for obj in objects:
        if not any(s['camera'] == camera.name for s in obj['sources']):
            continue
        uv, depth = camera.project(corners(obj));uv -= .5
        for i, j in EDGES:
            if min(depth[i], depth[j]) > .25 and np.isfinite(uv[[i, j]]).all():
                draw.line([tuple(uv[i]), tuple(uv[j])], fill=COLORS[obj['class_id']], width=2)
        center, d = camera.project([obj['center_ego_m']])
        if d[0] > .25 and np.isfinite(center).all():
            draw.text(tuple(center[0]), obj['track_id'].split(':')[-1]+' '+obj['class_name'], fill='white')
    return image


def iso_point(points):
    p = np.asarray(points)
    return np.stack([340+(p[..., 1]-p[..., 0]*.48)*13, 360-(p[..., 0]*.5+p[..., 2]*1.6)*13], -1)


def panel(rgb, cameras, objects, occupancy, grid, caption):
    image = Image.new('RGB', (1280, 950), (16, 20, 28));draw = ImageDraw.Draw(image)
    for i, (name, camera) in enumerate(cameras.items()):
        view = camera_overlay(rgb[name], camera, objects).resize((320, 180))
        x, y = (i % 4)*320, (i//4)*180;image.paste(view, (x, y));draw.text((x+4, y+4), name, fill='red')
    known_ground = np.any((occupancy >= 1) & (occupancy <= 2), axis=2)
    road = np.any(occupancy == 1, axis=2);sidewalk = np.any(occupancy == 2, axis=2)
    pixels = np.full(grid.ground.shape+(3,), [25, 29, 38], np.uint8)
    pixels[np.any(occupancy == 0, axis=2)] = [48, 52, 62];pixels[road] = [65, 95, 130];pixels[sidewalk] = [125, 105, 90]
    for category in range(1, 7):
        pixels[np.any(occupancy == category+2, axis=2)] = COLORS[category]
    bev = Image.fromarray(pixels).resize((400, 500), Image.Resampling.NEAREST)
    image.paste(bev, (0, 395));draw.text((4, 372), 'PREDICTED SEMANTIC VOXELS, top down', fill='white')
    iso = Image.new('RGB', (850, 520), (19, 24, 34));d = ImageDraw.Draw(iso)
    ground = grid.ground.points()[::3, ::3];mask = known_ground[::3, ::3]
    for p in iso_point(ground[mask]):
        d.ellipse((p[0]-1, p[1]-1, p[0]+1, p[1]+1), fill=(58, 86, 116))
    for x in [0, 10, 20, 30]:
        a, b = iso_point([[x, -15, 0], [x, 15, 0]])
        d.line([tuple(a), tuple(b)], fill=(38, 48, 62));d.text(tuple(a), f'{x}m', fill=(140, 150, 165))
    for obj in objects:
        uv = iso_point(corners(obj));color = COLORS[obj['class_id']]
        for i, j in EDGES:d.line([tuple(uv[i]), tuple(uv[j])], fill=color, width=2)
        p = iso_point(obj['center_ego_m']);d.text(tuple(p), obj['track_id'].split(':')[-1]+' '+obj['class_name'], fill=color)
    image.paste(iso, (420, 395));draw.text((425, 372), 'LEARNED RGB CUBOIDS + observed ground; upright/flat-ground assumptions', fill='white')
    draw.text((5, 909), caption, fill='white');draw.text((5, 928), 'Unknown is preserved. Cuboid volumes are coarse hypotheses; no GT pose/depth in inference. No vehicle control.', fill=(170, 185, 205))
    return image
