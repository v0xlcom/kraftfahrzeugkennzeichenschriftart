"""v2: adopt real-photo pressing (M HH 2092) + max resolution. Glyphs preserved exactly."""
import numpy as np
from PIL import Image, ImageFilter

SRC = "/home/carlo/.cursor/projects/home-carlo-prj-kraftfahrzeugkennzeichenschriftart/assets/image-09bc520a-beee-46d3-9517-9e480b2d54d2.png"
DST = "/home/carlo/prj/kraftfahrzeugkennzeichenschriftart/kennzeichen_3d_photorealistic_v2.png"

SCALE = 4  # max resolution: 1024x683 -> 4096x2732
img = Image.open(SRC).convert("RGB")
W, H = img.size
img_big = img.resize((W*SCALE, H*SCALE), Image.LANCZOS)
a = np.asarray(img_big).astype(np.float32) / 255.0
h, w, _ = a.shape

lum = 0.299*a[...,0] + 0.587*a[...,1] + 0.114*a[...,2]
blue_mask = (a[...,2] > 0.45) & (a[...,2]-a[...,0] > 0.15) & (a[...,1] > 0.35)
dark_mask = (lum < 0.55) & (~blue_mask)
dark_mask = np.asarray(Image.fromarray((dark_mask*255).astype(np.uint8)).filter(ImageFilter.MedianFilter(9))) > 127

# --- height: crisp stamp edge like real photo (narrow bevel) ---
height = np.asarray(Image.fromarray((dark_mask*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(SCALE*0.9))).astype(np.float32)/255.0
height = height*height*(3-2*height)
strength = 7.0
zy, zx = np.gradient(height)
nx=-zx*strength; ny=-zy*strength; nz=np.ones_like(nx)
inv=1.0/np.sqrt(nx*nx+ny*ny+nz*nz); nx*=inv; ny*=inv; nz*=inv
L=np.array([-0.38,-0.50,0.78],np.float32); L/=np.linalg.norm(L)
diff=np.clip(nx*L[0]+ny*L[1]+nz*L[2],0,1)
V=np.array([0.05,0.08,1.0],np.float32); V/=np.linalg.norm(V)
Hx,Hy,Hz=L+V; Hn=np.sqrt(Hx*Hx+Hy*Hy+Hz*Hz); Hx/=Hn;Hy/=Hn;Hz/=Hn
spec=np.clip(nx*Hx+ny*Hy+nz*Hz,0,1)**32

# --- cast drop shadow SE (as in photo: soft, offset down-right) ---
dx, dy = int(w*0.0045), int(h*0.009)  # ~18px / ~24px at 4x
m8 = (dark_mask*255).astype(np.uint8)
shifted = np.zeros_like(m8)
shifted[dy:, dx:] = m8[:-dy or h, :-dx or w] if dy and dx else m8
cast = np.asarray(Image.fromarray(shifted).filter(ImageFilter.GaussianBlur(SCALE*1.6))).astype(np.float32)/255.0
cast = np.clip(cast - height*0.92, 0, 1)  # only outside glyphs

# contact AO ring
blur = np.asarray(Image.fromarray((height*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(SCALE*2.0))).astype(np.float32)/255.0
crevice = np.clip((blur-height)*2.0,0,1)*(~dark_mask)

# --- plate base: warm dirty off-white like real foil + speckle ---
rng = np.random.default_rng(11)
g = rng.normal(0,1,(h,w)).astype(np.float32)
g = (np.asarray(Image.fromarray(((g-g.min())/np.ptp(g)*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7))).astype(np.float32)-127.5)/127.5
# low-freq mottling (dirt clouds)
small = rng.normal(0,1,(h//16,w//16)).astype(np.float32)
mottle = np.asarray(Image.fromarray(((small-small.min())/np.ptp(small)*255).astype(np.uint8)).resize((w,h),Image.BILINEAR).filter(ImageFilter.GaussianBlur(8))).astype(np.float32)
mottle=(mottle-127.5)/127.5
# sparse dust specks
speck = (rng.random((h,w))>0.9985).astype(np.float32)
speck = np.asarray(Image.fromarray((speck*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.5))).astype(np.float32)/255.0*3.0
yy,xx=np.mgrid[0:h,0:w].astype(np.float32); yy/=h; xx/=w
sheen=0.018*np.exp(-((yy-0.10)**2)/0.025)+0.010*(1-xx)*yy
base_lum=np.clip(0.945+g*0.005+mottle*0.014+sheen-speck*0.05,0,1)
base=np.stack([base_lum*0.995,base_lum*0.988,base_lum*0.962],axis=-1)  # warm tint #EFEDE2-ish

shade=0.70+0.30*diff[...,None]
plate=base*shade+(spec[...,None]*0.10)*(1-dark_mask[...,None])
plate-= (cast*0.50)[...,None]      # main cast shadow SE
plate-= (crevice*0.30)[...,None]   # contact darkening
# crisp top-edge sky highlight (thin, like photo)
top_hi=np.clip(-ny*2.2,0,1)*np.clip(zx*0+ (1-nz)*3,0,1)
top_hi=np.clip(-ny*1.8,0,1)*(height>0.05)*(height<0.95)
plate+= (top_hi*0.35)[...,None]*(1-dark_mask[...,None])

# --- glyph faces: satin-matte worn black, darker bottom ---
face_g = (rng.normal(0,1,(h,w)).astype(np.float32))
face_g = (np.asarray(Image.fromarray(((face_g-face_g.min())/np.ptp(face_g)*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))).astype(np.float32)-127.5)/127.5
vert = 1.0 - yy*0.35  # slightly lighter top, darker base (cylinder shading)
face_lum = np.clip(0.075*vert + 0.020*nz + 0.015*diff + face_g*0.008, 0.02, 0.30)
face=np.stack([face_lum*1.03,face_lum*1.03,face_lum*1.07],axis=-1)
face+=spec[...,None]*0.10
rim_l=np.clip(-(zx+zy)*11,0,1)*dark_mask
face+=rim_l[...,None]*0.30  # upper bevel catchlight on black
out=np.where(dark_mask[...,None],face,plate)

# blue EU band
if blue_mask.any():
    gloss=(0.82+0.28*diff+spec*0.25)[...,None]
    bs=np.clip(a*gloss+(cast*0.2)[...,None]*0+ (top_hi*0.1)[...,None],0,1)
    out=np.where(blue_mask[...,None],bs,out)

# pressed outer rim
Y,X=np.mgrid[0:h,0:w]
m=int(9*SCALE)
bd=np.minimum(np.minimum(X,w-1-X),np.minimum(Y,h-1-Y))
rim=np.clip(1-bd/m,0,1)
out=out*(1-rim[...,None]*0.60)+rim[...,None]*np.array([0.10,0.10,0.11])
out+=np.clip(1-np.abs(bd-m-3*SCALE)/(3*SCALE),0,1)[...,None]*0.09

vig=1-0.09*(((xx-0.5)**2+(yy-0.5)**2)/0.5)
out=np.clip(np.clip(out*vig[...,None],0,1)-0.5, -0.5,0.5)*1.03+0.5+0.008
out=np.clip(out,0,1)
Image.fromarray((out*255).astype(np.uint8)).save(DST)
print("saved", DST, out.shape)
