"""Render any text in kraftfahrzeugkennzeichenschirftart with a slight 3D stamp effect.

Usage: python3 render_3d_text.py "M-AB 1234" [out.png]
"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

TTF = "/home/carlo/prj/kraftfahrzeugkennzeichenschriftart/kraftfahrzeugkennzeichenschirftart.ttf"
SIZE = 220          # raster size of the flat glyphs (px cap height-ish)
SCALE = 2           # supersample factor for smooth bevels

text = sys.argv[1] if len(sys.argv) > 1 else "M-AB 1234"
out = sys.argv[2] if len(sys.argv) > 2 else "/home/carlo/prj/kraftfahrzeugkennzeichenschriftart/plate_3d.png"

font = ImageFont.truetype(TTF, SIZE)
tmp = Image.new("L", (10, 10), 255)
bb = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
tw, th = bb[2] - bb[0], bb[3] - bb[1]
pad = SIZE // 2
img = Image.new("L", (tw + 2 * pad, th + 2 * pad), 255)
ImageDraw.Draw(img).text((pad - bb[0], pad - bb[1]), text, font=font, fill=0)
img = img.resize((img.width * SCALE, img.height * SCALE), Image.LANCZOS)

a = np.asarray(img).astype(np.float32) / 255.0
h, w = a.shape
dark = a < 0.5
dark_pil = Image.fromarray((dark * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(5))
dark = np.asarray(dark_pil) > 127

height_pil = Image.fromarray((dark * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(SCALE * 1.2))
height = np.asarray(height_pil).astype(np.float32) / 255.0
height = height * height * (3 - 2 * height)

strength = 6.0
zy, zx = np.gradient(height)
nx, ny, nz = -zx * strength, -zy * strength, np.ones_like(zx)
inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
nx, ny, nz = nx * inv, ny * inv, nz * inv

L = np.array([-0.45, -0.55, 0.70], np.float32); L /= np.linalg.norm(L)
diff = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
V = np.array([0.05, 0.08, 1.0], np.float32); V /= np.linalg.norm(V)
Hx, Hy, Hz = L + V; Hx, Hy, Hz = Hx / np.linalg.norm([Hx, Hy, Hz]), Hy / np.linalg.norm([Hx, Hy, Hz]), Hz / np.linalg.norm([Hx, Hy, Hz])
spec = np.clip(nx * Hx + ny * Hy + nz * Hz, 0, 1) ** 28

blur = np.asarray(Image.fromarray((height * 255).astype(np.uint8)).filter(
    ImageFilter.GaussianBlur(SCALE * 2.2))).astype(np.float32) / 255.0
crevice = np.clip((blur - height) * 2.2, 0, 1) * (dark == 0)

rng = np.random.default_rng(7)
grain = rng.normal(0, 1, (h, w)).astype(np.float32)
grain = (grain - grain.min()) / np.ptp(grain)
metal = 0.965 + (grain - 0.5) * 0.012
yy, xx = np.mgrid[0:h, 0:w].astype(np.float32); yy /= h; xx /= w
sheen = 0.02 * np.exp(-((yy - 0.12) ** 2) / 0.02)
base = np.clip(metal + sheen, 0, 1)[..., None] * np.array([0.985, 0.986, 0.982], np.float32)

shade = 0.68 + 0.32 * diff[..., None]
plate = base * shade + (spec[..., None] * 0.22) * (1 - dark[..., None] * 0.9)
plate -= (crevice * 0.45)[..., None]

face_light = 0.055 + 0.035 * nz + 0.035 * diff
face = np.zeros_like(plate)
face[..., 0], face[..., 1], face[..., 2] = face_light * 1.02, face_light * 1.02, face_light * 1.05
face += spec[..., None] * 0.22
rim_light = np.clip(-(zx + zy) * 10, 0, 1) * dark
rim_dark = np.clip((zx + zy) * 10, 0, 1) * dark
face += rim_light[..., None] * 0.38
face -= rim_dark[..., None] * 0.05

result = np.where(dark[..., None], face, plate)
result = np.clip(result, 0, 1)
Image.fromarray((result * 255).astype(np.uint8)).save(out)
print("saved", out)
