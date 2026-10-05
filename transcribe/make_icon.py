"""Generates assets/transcribe.ico and assets/logo.png (waveform on a blue rounded tile)."""
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).parent / "assets"
OUT.mkdir(exist_ok=True)


def make_logo(size=512):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    # vertical gradient, blue-500 -> blue-700, masked to a rounded square
    grad = Image.new("RGBA", (size, size))
    gd = ImageDraw.Draw(grad)
    top, bottom = (66, 139, 255), (30, 88, 214)
    for y in range(size):
        t = y / (size - 1)
        c = tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3)) + (255,)
        gd.line([(0, y), (size, y)], fill=c)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius=int(size * 0.22), fill=255)
    img.paste(grad, (0, 0), mask)

    # symmetric waveform bars
    d = ImageDraw.Draw(img)
    heights = [0.20, 0.38, 0.60, 0.78, 0.60, 0.38, 0.20]
    n = len(heights)
    bw, gap = size * 0.070, size * 0.040
    total = n * bw + (n - 1) * gap
    x0, cy = (size - total) / 2, size / 2
    for i, h in enumerate(heights):
        x = x0 + i * (bw + gap)
        hh = h * size * 0.52
        d.rounded_rectangle([x, cy - hh / 2, x + bw, cy + hh / 2], radius=bw / 2, fill=(255, 255, 255, 255))
    return img


logo = make_logo()
logo.save(OUT / "logo.png")
logo.save(OUT / "transcribe.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print("wrote", OUT / "logo.png", "and", OUT / "transcribe.ico")
