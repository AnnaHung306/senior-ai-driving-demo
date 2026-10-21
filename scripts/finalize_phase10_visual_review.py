"""Create labelled route contact sheets; record human/assistant review separately."""
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'artifacts/runs/phase10-visual-QA-v1-001'
    images = sorted(folder.glob('route-*.png'))
    for start in range(0, len(images), 4):
        out = folder / f'route-contact-{start // 4}.jpg'
        if out.exists():
            raise FileExistsError(out)
        page = Image.new('RGB', (1440, 4 * 510), '#111b26')
        draw = ImageDraw.Draw(page)
        for index, path in enumerate(images[start:start + 4]):
            with Image.open(path) as image:
                image = image.convert('RGB')
                image.thumbnail((1440, 480))
                page.paste(image, (0, index * 510 + 25))
            draw.text((8, index * 510 + 5), path.name, fill='white')
        page.save(out, quality=90)
    print({'route_panels': len(images), 'contact_pages': (len(images) + 3) // 4})


if __name__ == '__main__':
    main()
