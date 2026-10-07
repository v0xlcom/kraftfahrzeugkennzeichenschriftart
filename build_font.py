"""Build kraftfahrzeugkennzeichenschirftart TTF from original.png (clean master of fig1)."""
import numpy as np
from PIL import Image, ImageFilter
import potrace
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.pens.cu2quPen import Cu2QuPen

SRC = "/home/carlo/prj/kraftfahrzeugkennzeichenschriftart/dreidimensionale_kraftfahrzeugkennzeichenschirftart.png"
FAMILY = "kraftfahrzeugkennzeichenschirftart"  # as requested (note: correct DE is ...schriftart)
EM = 1000
CAP = 700          # cap height in units
LSB = 90           # side bearing
UPSCALE = 3

img = Image.open(SRC).convert("RGB")
W, H = img.size
a = np.asarray(img).astype(np.float32)
R, G, B = a[..., 0], a[..., 1], a[..., 2]
lum = 0.299 * R + 0.587 * G + 0.114 * B
blue = (B > 110) & (B - R > 25) & (G > 70)
ink = (lum < 150) & (~blue)
# NOTE: no whole-stripe wipe here: rows 1-3 have glyphs (J/T/0) under the
# blue box's x-range; the blue-pixel mask above already excludes the EU field.
ink_pil = Image.fromarray((ink * 255).astype(np.uint8))
ink_pil = ink_pil.filter(ImageFilter.MedianFilter(3))
ink = np.asarray(ink_pil) > 127

# --- row split via horizontal projection ---
rowsum = ink.sum(axis=1)
mask = rowsum > (W * 0.02)
bands, inb, st = [], False, 0
for y in range(H):
    if mask[y] and not inb:
        inb, st = True, y
    elif not mask[y] and inb:
        inb = False
        if y - st > 20:
            bands.append((st, y))
if inb and H - st > 20:
    bands.append((st, H))
print("row bands:", bands)
assert len(bands) == 4, f"expected 4 rows, got {len(bands)}"

ROW_NAMES = [
    list("ABCDEFGHI"),
    list("JKLMNOPQRS"),
    list("TUVWXYZ") + ["Ä", "Ö", "Ü"],
    list("0123456789"),
]

def split_cols(band):
    y0, y1 = band
    colsum = ink[y0:y1, :].sum(axis=0)
    cm = colsum > ((y1 - y0) * 0.03)
    boxes, inb, st = [], False, 0
    for x in range(W):
        if cm[x] and not inb:
            inb, st = True, x
        elif not cm[x] and inb:
            inb = False
            if x - st > 12:
                boxes.append((st, x))
    if inb and W - st > 12:
        boxes.append((st, W))
    # merge boxes split by thin gaps (e.g. inner counters don't split cols since projection;
    # but '0' slash-gap is horizontal so fine). Merge gaps < 8px.
    merged = []
    for b in boxes:
        if merged and b[0] - merged[-1][1] < 8:
            merged[-1] = (merged[-1][0], b[1])
        else:
            merged.append(b)
    return merged

all_boxes = [split_cols(b) for b in bands]
for i, bx in enumerate(all_boxes):
    print(f"row{i}: {len(bx)} boxes")
    assert len(bx) == len(ROW_NAMES[i]), f"row{i}: want {len(ROW_NAMES[i])} got {len(bx)}"

# --- trace one glyph crop to cubic contours (potrace coords, y down) ---
def trace_glyph_crop(y0, y1, x0, x1):
    pad = 6
    y0 = max(0, y0 - pad); y1 = min(H, y1 + pad)
    x0 = max(0, x0 - pad); x1 = min(W, x1 + pad)
    crop = Image.fromarray((ink[y0:y1, x0:x1] * 255).astype(np.uint8))
    crop = crop.resize((crop.width * UPSCALE, crop.height * UPSCALE), Image.LANCZOS)
    arr = np.asarray(crop)
    binarr = (arr > 140).astype(np.uint8)  # True=white bg
    # potrace Bitmap: 0=black(background?) -- Bitmap(data) treats nonzero as black? use invert check:
    # potracer: Bitmap(data, blacklevel); data>blacklevel => black? We want ink(black glyph)=1.
    data = (binarr == 0).astype(np.uint8)  # glyph pixels = 1? test both below
    # NOTE: potrace lib treats 1 as foreground? verify empirically: trace and check bbox non-empty
    bmp = potrace.Bitmap((data * 255).astype(np.uint8))
    path = bmp.trace(turdsize=4, alphamax=1.0, opttolerance=0.3)
    if len(path) == 0:  # try inverted
        bmp = potrace.Bitmap(((1 - data) * 255).astype(np.uint8))
        path = bmp.trace(turdsize=4, alphamax=1.0, opttolerance=0.3)
    return path, crop.size

# quick polarity self-test on first glyph
tp, _ = trace_glyph_crop(*bands[0], *all_boxes[0][0])
print("self-test curves:", len(tp))
assert len(tp) > 0, "potrace polarity wrong / empty trace"

glyph_paths = {}   # name -> (path, crop_w, crop_h, row_h)
for ri, band in enumerate(bands):
    y0, y1 = band
    for name, (x0, x1) in zip(ROW_NAMES[ri], all_boxes[ri]):
        p, (cw, ch) = trace_glyph_crop(y0, y1, x0, x1)
        glyph_paths[name] = (p, cw, ch, y1 - y0)
        print(f"{name}: {len(p)} contours, crop {cw}x{ch}")

# --- assemble TTF ---
UNITS_PER_CROP_PX = None
order = [".notdef", "space"]
names = []
cmap = {}
for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ0123456789":
    names.append(ch)
order += names
# lowercase umlauts share outlines
order += ["ä", "ö", "ü"]

UL = {"Ä": "Adieresis", "Ö": "Odieresis", "Ü": "Udieresis",
      "ä": "adieresis", "ö": "odieresis", "ü": "udieresis"}
def psname(c):
    if c in UL:
        return UL[c]
    if len(c) == 1 and (c.isalpha() or c.isdigit()):
        return c if c.isupper() or c.isdigit() else c
    return c

gorder = [".notdef", "space"] + [psname(c) for c in names] + ["adieresis", "odieresis", "udieresis"]

fb = FontBuilder(EM, isTTF=True)
fb.setupGlyphOrder(gorder)
metrics = {}
glyfs = {}
pen_of = {}

def _pt(p):
    if hasattr(p, "x"):
        return (p.x, p.y)
    try:
        return (p[0], p[1])
    except TypeError:
        return (p.x, p.y)


def draw_path(cu_pen, path, sx, sy, yflip_base):
    # path: potrace Path, coords in crop px (upscaled). sx: scale to units.
    for curve in path:
        segs = list(curve)
        if not segs:
            continue
        # start point = end of last segment (closed contour)
        def ep(s):
            return _pt(s.end_point)
        start = ep(segs[-1])
        cu_pen.moveTo((start[0] * sx + LSB, yflip_base - start[1] * sx))
        for s in segs:
            if s.is_corner:
                cx, cy = _pt(s.c)
                ex, ey = _pt(s.end_point)
                cu_pen.lineTo((cx * sx + LSB, yflip_base - cy * sx))
                cu_pen.lineTo((ex * sx + LSB, yflip_base - ey * sx))
            else:
                c1x, c1y = _pt(s.c1)
                c2x, c2y = _pt(s.c2)
                ex, ey = _pt(s.end_point)
                cu_pen.curveTo((c1x * sx + LSB, yflip_base - c1y * sx),
                               (c2x * sx + LSB, yflip_base - c2y * sx),
                               (ex * sx + LSB, yflip_base - ey * sx))
        cu_pen.closePath()

BASELINE = 150  # descender 0..150 empty; cap top = 150+700=850
for c in names:
    pname = psname(c)
    p, cw, ch, rowh = glyph_paths[c]
    s = CAP / (rowh * UPSCALE)
    adv = int(round(cw * s + 2 * LSB))
    metrics[pname] = (adv, LSB)
    ttpen = TTGlyphPen(None)
    cu = Cu2QuPen(ttpen, max_err=1.0)
    ybase = BASELINE + ch * s
    draw_path(cu, p, s, None, ybase)
    glyfs[pname] = ttpen.glyph()

# .notdef + space
from fontTools.ttLib.tables._g_l_y_f import Glyph
nd = TTGlyphPen(None).glyph()
glyfs[".notdef"] = nd
metrics[".notdef"] = (600, 0)
glyfs["space"] = TTGlyphPen(None).glyph()
metrics["space"] = (500, 0)
for n in ["adieresis", "odieresis", "udieresis"]:
    src = {"adieresis": "Adieresis", "odieresis": "Odieresis", "udieresis": "Udieresis"}[n]
    glyfs[n] = glyfs[src]
    metrics[n] = metrics[src]

fb.setupCharacterMap({ord("A") + i: chr(ord("A") + i) for i in range(26)} |
                     {ord("0") + i: str(i) for i in range(10)} |
                     {0xC4: "Adieresis", 0xD6: "Odieresis", 0xDC: "Udieresis",
                      0xE4: "adieresis", 0xF6: "odieresis", 0xFC: "udieresis",
                      0x20: "space"})
fb.setupGlyf(glyfs)
fb.setupHorizontalMetrics(metrics)
fb.setupHorizontalHeader(ascent=900, descent=250)
fb.setupNameTable({"familyName": FAMILY, "styleName": "Regular",
                   "uniqueFontIdentifier": f"{FAMILY}-Regular",
                   "fullName": f"{FAMILY} Regular",
                   "psName": f"{FAMILY}-Regular", "version": "Version 1.0"})
fb.setupOS2(sTypoAscender=900, sTypoDescender=-250, usWinAscent=900, usWinDescent=250)
fb.setupPost()
fb.setupMaxp()
out = f"/home/carlo/prj/kraftfahrzeugkennzeichenschriftart/{FAMILY}.ttf"
fb.save(out)
print("saved", out)
