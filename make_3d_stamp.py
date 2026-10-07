"""Embossed license-plate render of the font alphabet, tuned to real plate photos.

Reference measurements (bleistifte.png, cap height Hc ~ 93 px):
  - lighting mostly diffuse, soft key from upper left
  - lit bevel: thin bright rim (+6..14 %), ~1.5 % Hc wide
  - shadow side: soft dark ramp ~7-9 % Hc, down to 0.4-0.6 x foil
  - enclosed counters ~17 % darker than open plate
  - ink: satin charcoal ~0.10-0.13, slightly darker at its rounded edge
  - foil: 0.78-0.86, smooth, large soft illumination gradient
Output: plain white plate without frame; flat foil renders exactly white.
"""
import numpy as np
import cv2

SRC = "/home/carlo/.cursor/projects/home-carlo-prj-kraftfahrzeugkennzeichenschriftart/assets/image-5b624b40-4da9-407b-9f32-825a854ea95e.png"
OUT = "/home/carlo/prj/kraftfahrzeugkennzeichenschriftart/kennzeichen_3d_stamp"

S = 8            # upscale factor (source cap height ~140 px -> Hc ~ 1120 px)
PM, BM = 34, 0   # white margin around content (source px)
B = 6.0 * S      # stamped bevel width: wider = softer slope (~4 % Hc)
SOFT = 0.55 * S  # final whole-image blur sigma
HG = 2.2 * S     # letter relief height: shallower = lighter bevel shading
F, HF = 5.0 * S, 0.25  # drawn-metal fillet: longer, lower = gentler wide ramp
rng = np.random.default_rng(7)

src = cv2.imread(SRC, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255
h0, w0 = src.shape[:2]
pad = PM + BM
H, W = (h0 + 2 * pad) * S, (w0 + 2 * pad) * S
f32 = np.float32


def up(x, s=0.6):
    x = cv2.copyMakeBorder(x, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    x = cv2.resize(x, (W, H), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(x, (0, 0), S * s)


def blur(x, s):
    return cv2.GaussianBlur(x, (0, 0), s)


def noise(s, amp=1.0):
    n = blur(rng.standard_normal((H, W), dtype=f32), s)
    return n / (n.std() + 1e-8) * amp


def clean(mask, min_area):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


def sdf(mask):
    m = mask.astype(np.uint8)
    din = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    sd = din - dout
    return sd - 0.5 * np.sign(sd)  # boundary at 0, positive inside


# ---------------- masks ----------------
r, g, b = src[..., 0], src[..., 1], src[..., 2]
lum = 0.299 * r + 0.587 * g + 0.114 * b
blue1 = ((b - r > 0.25) & (b > 0.5)).astype(f32)
blue_col = np.median(src[blue1 > 0], axis=0)
blue_zone = cv2.dilate(blue1, np.ones((7, 7), np.uint8))
dark1 = np.clip((0.62 - lum) / 0.35, 0, 1) * (1 - blue_zone)

glyph = clean(up(dark1, 1.2) > 0.5, 40 * S * S)
band = clean(up(blue1) > 0.5, 10 * S * S)
sd_g = blur(sdf(glyph), 0.4 * S)
a_ink = np.clip(sd_g + noise(1.5, 0.15) + 0.5, 0, 1)

stars = np.zeros((H, W), np.uint8)              # EU star ring, redrawn (too small to upscale)
STAR_C, STAR_R, STAR_SZ = (58 + pad, 87.1 + pad), 23.4, 2.6
for i in range(12):
    t = np.radians(30 * i)
    cx, cy = STAR_C[0] + STAR_R * np.sin(t), STAR_C[1] - STAR_R * np.cos(t)
    pts = [(cx + STAR_SZ * (1 if j % 2 == 0 else 0.42) * np.sin(np.pi * j / 5),
            cy - STAR_SZ * (1 if j % 2 == 0 else 0.42) * np.cos(np.pi * j / 5)) for j in range(10)]
    cv2.fillPoly(stars, [np.round(np.array(pts) * S * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
a_blue = np.clip(sdf(band) + 0.5, 0, 1) * (1 - stars.astype(f32) / 255)
del dark1, stars

# ---------------- height field (in px) ----------------
def stamp(d, w):  # S-curve stamped profile: 1 on top, 0 on plate
    t = np.clip(d / w, 0, 1)
    return 0.5 * (1 + np.cos(np.pi * t))

d_let = np.maximum(-sd_g - 0.25 * B, 0)         # ink stops short of the shoulder: white rim
h = HG * ((1 - HF) * stamp(d_let, B) + HF * np.exp(-d_let / F))
h = blur(h, 0.6 * S)  # round the stamped shoulder: blurs the bevel highlight/shadow edge
del d_let

# ---------------- normals ----------------
gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3) / 8
gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3) / 8
inv = 1 / np.sqrt(gx * gx + gy * gy + 1)
nx, ny, nz = -gx * inv, -gy * inv, inv
del gx, gy, inv

elev = np.radians(30)
ld = np.array([-0.5, -0.866]); ld /= np.linalg.norm(ld)
L = np.array([ld[0] * np.cos(elev), ld[1] * np.cos(elev), np.sin(elev)], f32)
ndl = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)

occ = np.zeros((H, W), f32)                     # ray-marched cast shadows toward the light
tan_e = np.tan(elev)
for k in range(1, int(HG * 1.35 / tan_e) + 3):
    M = np.array([[1, 0, ld[0] * k], [0, 1, ld[1] * k]], f32)
    hs = cv2.warpAffine(h, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_REPLICATE)
    np.maximum(occ, hs - h - k * tan_e, out=occ)
shadow = blur(np.clip(occ / (0.12 * HG), 0, 1), 1.5 * S)
del occ, hs

occl = np.clip((blur(h, 1.5 * S) - h) / HG * 0.6 + (blur(h, 6 * S) - h) / HG * 0.45, 0, 1)
ao = 1 - 0.30 * occl  # softer contact darkening: lightens the bevel shadow side

E = 0.30 * ndl * (1 - 0.85 * shadow) + 0.72 * ao * (0.5 + 0.5 * nz ** 2)  # more ambient, less direct = softer bevel
E /= 0.30 * L[2] + 0.72                         # flat open plate renders exactly white
Hh = L + np.array([0, 0, 1], f32); Hh /= np.linalg.norm(Hh)
ndh = np.clip(nx * Hh[0] + ny * Hh[1] + nz * Hh[2], 0, 1) * (1 - 0.9 * shadow)
Ry, Rz = 2 * nz * ny, 2 * nz * nz - 1
env = np.clip(0.55 - 0.9 * Ry - 0.25 * (1 - Rz), 0, 1.3) - 0.55
fres = 0.04 + 0.96 * (1 - nz) ** 5
del ndl, nx, ny, Ry, Rz

# ---------------- materials ----------------
mott = noise(32, 1.0)
grain = noise(0.7, 1.0)

foil = (1 - 0.03 * occl)[..., None] * np.ones(3, f32)  # lighter occlusion tint on the bevel
foil_spec = 0.04 * ndh ** 30  # halved: less sparkle on the bevel slope
foil_env = 0.04 + 0.30 * fres  # dimmer rim reflection

ink_alb = 0.11 + 0.004 * grain + 0.005 * mott
ink_spec = 0.06 * ndh ** 14
ink_env = 0.10 + 0.90 * fres

blue = blue_col[None, None, :] * (1 + 0.006 * grain + 0.008 * mott)[..., None]

ink = a_ink[..., None]
bl = (a_blue * (1 - a_ink))[..., None]
alb = foil * (1 - ink - bl) + ink_alb[..., None] * ink + blue * bl
spec = foil_spec * (1 - ink[..., 0]) + ink_spec * ink[..., 0]
envk = foil_env * (1 - ink[..., 0]) + ink_env * ink[..., 0]
del foil, blue

out = alb * E[..., None] + (spec + envk * env * 0.35)[..., None]  # quieter bevel rim light
del alb, spec, envk, E
out = np.clip(blur(out, SOFT), 0, 1)

img = (out[..., ::-1] * 255 + 0.5).astype(np.uint8)
cv2.imwrite(OUT + ".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
cv2.imwrite(OUT + ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
cv2.imwrite(OUT + "_preview.jpg", cv2.resize(img, (2400, int(2400 * H / W)), interpolation=cv2.INTER_AREA),
            [cv2.IMWRITE_JPEG_QUALITY, 92])
print("saved", OUT, img.shape)
