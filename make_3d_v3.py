"""v3: max-resolution embossed license plate render.

Glyph contours are rebuilt from the source at S x scale (smooth sub-pixel edges),
then rendered as stamped sheet metal: black hot-foil on the raised letter tops,
white retroreflective foil on plate and bevels, raised black DIN border,
ray-marched shadows, ambient occlusion and environment reflections.
"""
import numpy as np
import cv2

SRC = "/home/carlo/.cursor/projects/home-carlo-prj-kraftfahrzeugkennzeichenschriftart/assets/image-5b624b40-4da9-407b-9f32-825a854ea95e.png"
OUT = "/home/carlo/prj/kraftfahrzeugkennzeichenschriftart/kennzeichen_3d_photorealistic_v3"

S = 8            # upscale factor
PM, BM = 34, 40  # plate margin around content, background margin (source px)
B = 3.0 * S      # stamped bevel width
HG = 2.6 * S     # letter relief height
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

# ---------------- plate geometry ----------------
yy, xx = np.mgrid[0:H, 0:W].astype(f32)
hx, hy, rad = (w0 + 2 * PM) * S / 2, (h0 + 2 * PM) * S / 2, 14 * S
qx = np.abs(xx - W / 2) - (hx - rad)
qy = np.abs(yy - H / 2) - (hy - rad)
sd_p = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - rad
del qx, qy
a_plate = np.clip(0.5 - sd_p, 0, 1)

INSET, BW = 9 * S, 4 * S                        # border bead centre inset, black line width
db = np.abs(sd_p + INSET) - BW / 2
a_border = np.clip(0.5 - db, 0, 1)

# ---------------- height field (in px) ----------------
def stamp(d):  # S-curve stamped profile: 1 on top, 0 on plate
    t = np.clip(d / B, 0, 1)
    return 0.5 * (1 + np.cos(np.pi * t))

h_let = stamp(np.maximum(-sd_g, 0))
h_bead = 0.85 * stamp(np.maximum(db, 0))
edge = np.clip(-sd_p / (3 * S), 0, 1)
h = HG * (np.maximum(h_let, h_bead) - 0.6 * (1 - edge) ** 2)
h += noise(3, 0.02) + noise(90, 0.8)            # orange peel + slight sheet waviness
h = blur(h, 1.2)
del h_let, h_bead, edge

# ---------------- normals ----------------
gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3) / 8
gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3) / 8
inv = 1 / np.sqrt(gx * gx + gy * gy + 1)
nx, ny, nz = -gx * inv, -gy * inv, inv
del gx, gy, inv

elev = np.radians(38)
ld = np.array([-0.55, -0.835]); ld /= np.linalg.norm(ld)
L = np.array([ld[0] * np.cos(elev), ld[1] * np.cos(elev), np.sin(elev)], f32)
ndl = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)

# ray-marched cast shadows toward the light
occ = np.zeros((H, W), f32)
tan_e = np.tan(elev)
for k in range(1, int(HG * 1.3 / tan_e) + 3):
    M = np.array([[1, 0, ld[0] * k], [0, 1, ld[1] * k]], f32)
    hs = cv2.warpAffine(h, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_REPLICATE)
    np.maximum(occ, hs - h - k * tan_e, out=occ)
shadow = blur(np.clip(occ / (0.12 * HG), 0, 1), 2.5 * S / 8 * 3)
del occ, hs

occl = np.clip((blur(h, 6 * S) - h) / HG * 0.9 + (blur(h, 1.5 * S) - h) / HG * 0.8, 0, 1)
ao = 1 - 0.55 * occl

E = 0.95 * ndl * (1 - 0.8 * shadow) + 0.45 * ao * (0.55 + 0.45 * nz)
Hh = L + np.array([0, 0, 1], f32); Hh /= np.linalg.norm(Hh)
ndh = np.clip(nx * Hh[0] + ny * Hh[1] + nz * Hh[2], 0, 1) * (1 - 0.9 * shadow)
Ry, Rz = 2 * nz * ny, 2 * nz * nz - 1
env = np.clip(0.55 - 0.9 * Ry - 0.25 * (1 - Rz), 0, 1.3) - 0.55
fres = 0.04 + 0.96 * (1 - nz) ** 5
del ndl, nx, ny, Ry, Rz

# ---------------- materials ----------------
k = 2 * np.pi / (3 * S * 2)                     # retroreflective hex bead lattice (~3 mm)
hexp = (np.cos(k * xx) + np.cos(k * (0.5 * xx + 0.866 * yy)) + np.cos(k * (-0.5 * xx + 0.866 * yy)) - 0.75) / 2.25
mott = noise(60 * S / 8 * 4, 1.0)
grain = noise(0.7, 1.0)
dust = blur((rng.random((H, W), dtype=f32) > 0.99985).astype(f32), 1.2) * 6

foil_alb = (0.925 + 0.004 * hexp + 0.008 * mott + 0.004 * grain - 0.10 * dust) * (1 - 0.14 * occl)
foil = foil_alb[..., None] * np.array([1.0, 0.992, 0.965], f32)
foil_spec = 0.16 * ndh ** 40 * (1 + 0.3 * hexp)
foil_env = 0.28 + 0.72 * fres

ink_alb = 0.032 + 0.006 * grain + 0.004 * mott
ink_spec = 0.11 * ndh ** 18
ink_env = 0.10 + 0.90 * fres

blue = blue_col[None, None, :] * (1 + 0.006 * grain + 0.008 * mott)[..., None]

ink = np.maximum(a_ink, a_border)[..., None]
bl = (a_blue * (1 - a_ink))[..., None]
alb = foil * (1 - ink - bl) + ink_alb[..., None] * ink + blue * bl
spec = foil_spec * (1 - ink[..., 0]) + ink_spec * ink[..., 0]
envk = foil_env * (1 - ink[..., 0]) + ink_env * ink[..., 0]
del foil, blue, foil_alb, hexp

plate_rgb = alb * E[..., None] + (spec + envk * env * 0.55)[..., None]
plate_rgb *= np.array([1.0, 0.995, 0.985], f32)  # warm key light
del alb, spec, envk, E

# ---------------- background surface + plate drop shadow ----------------
bg_l = 0.17 + 0.012 * mott + 0.01 * grain
bg = bg_l[..., None] * np.array([0.98, 0.99, 1.03], f32)
M = np.array([[1, 0, -ld[0] * 7 * S], [0, 1, -ld[1] * 7 * S]], f32)
psh = cv2.warpAffine(a_plate, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
psh = blur(psh, 9 * S)
bg *= (1 - 0.75 * psh)[..., None]

out = plate_rgb * a_plate[..., None] + bg * (1 - a_plate[..., None])
del plate_rgb, bg

vig = 1 - 0.10 * (((xx / W - 0.48) ** 2 + (yy / H - 0.45) ** 2) / 0.5)
out = out * vig[..., None] + noise(0.6, 0.004)[..., None]
out = np.clip(out, 0, 1)

img = (out[..., ::-1] * 255 + 0.5).astype(np.uint8)
cv2.imwrite(OUT + ".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
cv2.imwrite(OUT + ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
cv2.imwrite(OUT + "_preview.jpg", cv2.resize(img, (2400, int(2400 * H / W)), interpolation=cv2.INTER_AREA),
            [cv2.IMWRITE_JPEG_QUALITY, 92])
print("saved", OUT, img.shape)
