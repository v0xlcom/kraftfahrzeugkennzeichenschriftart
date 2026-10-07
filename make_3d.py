"""Photorealistic pressed-metal license plate effect, preserves glyphs exactly."""
import numpy as np
from PIL import Image, ImageFilter

SRC = "/home/carlo/.cursor/projects/home-carlo-prj-kraftfahrzeugkennzeichenschriftart/assets/image-09bc520a-beee-46d3-9517-9e480b2d54d2.png"
DST = "/home/carlo/prj/kraftfahrzeugkennzeichenschriftart/kennzeichen_3d_photorealistic.png"

SCALE = 3  # upscale for smooth bevels
img = Image.open(SRC).convert("RGB")
W, H = img.size
img_big = img.resize((W*SCALE, H*SCALE), Image.LANCZOS)
a = np.asarray(img_big).astype(np.float32) / 255.0
h, w, _ = a.shape

# --- masks: black glyphs vs blue EU field ---
lum = 0.299*a[...,0] + 0.587*a[...,1] + 0.114*a[...,2]
# blue field: strong blue, low red
blue_mask = (a[...,2] > 0.45) & (a[...,2] - a[...,0] > 0.15) & (a[...,1] > 0.35)
dark_mask = (lum < 0.55) & (~blue_mask)
# clean tiny specks
dark_pil = Image.fromarray((dark_mask*255).astype(np.uint8)).filter(ImageFilter.MedianFilter(3*SCALE//2*2+1))
dark_mask = np.asarray(dark_pil) > 127

# --- height field: glyphs raised (embossed/pressed) ---
height_pil = Image.fromarray((dark_mask*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(SCALE*1.2))
height = np.asarray(height_pil).astype(np.float32)/255.0
# rounded bevel: smoothstep-ish
height = height*height*(3-2*height)

# normals from height
strength = 6.0
zy, zx = np.gradient(height)
nx = -zx*strength; ny = -zy*strength; nz = np.ones_like(nx)
inv = 1.0/np.sqrt(nx*nx+ny*ny+nz*nz)
nx*=inv; ny*=inv; nz*=inv

# light from top-left, slightly frontal
L = np.array([-0.45, -0.55, 0.70], np.float32); L/=np.linalg.norm(L)
diff = np.clip(nx*L[0]+ny*L[1]+nz*L[2], 0, 1)
# specular (Blinn-Phong)
V = np.array([0.05, 0.08, 1.0], np.float32); V/=np.linalg.norm(V)
Hx, Hy, Hz = L+V; Hn=np.sqrt(Hx*Hx+Hy*Hy+Hz*Hz); Hx/=Hn; Hy/=Hn; Hz/=Hn
spec = np.clip(nx*Hx+ny*Hy+nz*Hz, 0, 1)**28

# cavity/AO: darken at glyph feet (edges of height)
edge = np.abs(height-Image.fromarray((height*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(SCALE*2.5).convert("L") and 0)) if False else None
blur_pil = Image.fromarray((height*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(SCALE*2.2))
blur = np.asarray(blur_pil).astype(np.float32)/255.0
crevice = np.clip((blur-height)*2.2, 0, 1)*(dark_mask==0)  # ring just outside glyphs
inner = np.clip((height-blur)*1.5, 0, 1)

# --- brushed white-metal plate base ---
rng = np.random.default_rng(7)
grain_raw = rng.normal(0, 1, (h, w)).astype(np.float32)
grain = np.asarray(Image.fromarray(((grain_raw-grain_raw.min())/np.ptp(grain_raw)*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))).astype(np.float32)
grain = (grain-127.5)/127.5
# vertical brushed streaks
streak_raw = rng.normal(0, 1, (1, w)).astype(np.float32).repeat(h, axis=0)
streak = np.asarray(Image.fromarray(((streak_raw-streak_raw.min())/np.ptp(streak_raw)*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.0))).astype(np.float32)
streak = (streak-127.5)/127.5
# simple: combine
metal = 0.965 + grain*0.006 + streak*0.004
# soft studio reflection: diagonal gradient + top sheen
yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
yy/=h; xx/=w
sheen = 0.02*np.exp(-((yy-0.12)**2)/0.02) + 0.012*(1-xx)*yy + 0.008*xx*(1-yy)
base = np.clip(metal + sheen, 0, 1)[...,None] * np.array([0.985, 0.986, 0.982], np.float32)

# apply relief shading to plate
shade = 0.68 + 0.32*diff[...,None]   # diffuse relief
plate = base * shade + (spec[...,None]*0.22) * (1-dark_mask[...,None]*0.9)
plate -= (crevice*0.45)[...,None]   # contact shadow at glyph feet
plate += (np.clip(ny,-1,1)*0.0)[...,None]

# --- glyph faces: near-black with subtle top-light gradient + edge bevel tint ---
face_light = 0.055 + 0.035*nz + 0.035*diff  # deep glossy black like real plates
face = np.zeros_like(plate)
face[...,0]=face_light*1.02; face[...,1]=face_light*1.02; face[...,2]=face_light*1.05
face += spec[...,None]*0.22  # glossy top edge on black
# bevel rim: lit rim on top-left of glyph, dark rim bottom-right
rim_light = np.clip(-(zx+zy)*10, 0, 1)*dark_mask
rim_dark = np.clip((zx+zy)*10, 0, 1)*dark_mask
face += rim_light[...,None]*0.38
face -= rim_dark[...,None]*0.05

out = np.where(dark_mask[...,None], face, plate)

# keep EU blue field printed but with plate shading + gloss
if blue_mask.any():
    blue_orig = a.copy()
    # deepen blue slightly for realism
    gloss = (0.85+0.3*diff+spec*0.4)[...,None]
    blue_shaded = blue_orig*gloss
    # thin embossed border effect around blue field via its edges
    bm = Image.fromarray((blue_mask*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(SCALE*0.8))
    bmh = np.asarray(bm).astype(np.float32)/255.0
    bzy, bzx = np.gradient(bmh)
    bedge = np.clip(np.abs(bzx)+np.abs(bzy),0,1)
    light_e = np.clip(-(bzx+bzy)*6,0,1)[...,None]
    dark_e = np.clip((bzx+bzy)*6,0,1)[...,None]
    blue_shaded = blue_shaded*(1-dark_e*0.35)+light_e*0.25
    out = np.where(blue_mask[...,None], blue_shaded, out)

# plate border: rounded dark edge + inner highlight for stamped metal rim
Y, X = np.mgrid[0:h, 0:w]
m = int(8*SCALE)
border_dist = np.minimum(np.minimum(X, w-1-X), np.minimum(Y, h-1-Y))
rim = np.clip(1-border_dist/m, 0, 1)
out = out*(1-rim[...,None]*0.55) + rim[...,None]*np.array([0.12,0.12,0.13])
inner_hi = np.clip(1-np.abs(border_dist-m-3*SCALE)/(3*SCALE),0,1)
out += inner_hi[...,None]*0.10

# gentle vignette + final grade
vig = 1 - 0.10*(((xx-0.5)**2+(yy-0.5)**2)/0.5)
out = np.clip(out*vig[...,None], 0, 1)
out = np.clip((out-0.5)*1.04+0.5+0.01, 0, 1)  # slight contrast

Image.fromarray((out*255).astype(np.uint8)).save(DST)
print("saved", DST)
