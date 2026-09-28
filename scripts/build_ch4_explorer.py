"""Build the interactive Ch4 explorer assets for negidamd.github.io.

Background : CAT12 MNI152NLin2009cAsym T1 template (1 mm, Template_T1), masked with CAT12 brainmask
             (CSF-inclusive, so ventricles and cisterns stay visible), cropped to the brain bounding box.
Ch4        : Julich-Brain / SPM Anatomy Toolbox probabilistic map of Ch 4 (basal forebrain),
             MNI152NLin2009cAsym 1 mm (EBRAINS, Zaborszky et al. 2008, NeuroImage 42:1127).
Ch1-3      : Julich-Brain v3 maximum probability map, label "Ch 123 (Basal Forebrain)" (grayvalue 79).
Template_T1 and both atlases share the 193x229x193 2009c grid.
CSF removal: the brain-masked (CSF-inclusive) template was segmented with FSL FAST (fast -t 1 -n 3);
Ch4 voxels whose CSF partial-volume estimate is > 0.5 or that lie outside the brain mask are set to 0.
NOTE: FAST must NOT be run on Template_T1_masked - CAT12 already zeroed CSF there, so FAST's darkest
class becomes dark gray matter (this removed Ch4 core voxels in an earlier build).
Cached PVE: scripts/fast_template/Template_T1_brain_withCSF_pve_csf.nii.gz

Outputs (assets/ch4-explorer/):
  coronal_t1.jpg     5x5 sprite of T1 coronal slices (2.5x upsampled)
  coronal_lbl.png    5x5 sprite of label slices at 1 mm  (R = T1 mask, G = Ch4 prob*255/0.70, B = Ch1-3)
  axial_t1.jpg / axial_lbl.png, sagittal_t1.jpg / sagittal_lbl.png  (planes through the Ch4 peak)
  meta.json          geometry + MNI coordinates for every slice
  ch4_card.png/.pdf  static card image (coronal, YlOrRd)
"""
import json, os
import numpy as np, nibabel as nib
from scipy import ndimage
from PIL import Image
from nibabel.processing import resample_from_to
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAT = "/Users/ahmednegida/matlab_toolboxes/spm12_toolbox_backup/cat12/templates_MNI152NLin2009cAsym/"
A = "/Users/ahmednegida/Library/Mobile Documents/com~apple~CloudDocs/3-resources/brain-atlases/"
MPM = A + "hbp-d000001_jubrain-cytoatlas_pub-MPM-collections-26/MPM/"
OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "ch4-explorer")
UP = 2.5

full = nib.load(CAT + "Template_T1.nii.gz")
c_img = nib.load(A + "BF_Atlas_Resources/masks/JuBrain_wholeCh4_prob_MNI152NLin2009cAsym_1mm.nii.gz")
assert np.allclose(full.affine, c_img.affine)
BM = resample_from_to(nib.load(CAT + "brainmask.nii.gz"), full, order=1).get_fdata() > 0.5
nz = np.argwhere(BM); lo_, hi_ = nz.min(0) - 2, nz.max(0) + 3
crop = tuple(slice(max(0, a), b) for a, b in zip(lo_, hi_))
T = np.where(BM, full.get_fdata(), 0)[crop].astype(np.float32)
BMc = BM[crop]
aff = full.affine.copy(); aff[:3, 3] = full.affine[:3, :3] @ np.array([c.start for c in crop]) + full.affine[:3, 3]
t_img = nib.Nifti1Image(T, aff)
sx, sy, sz = T.shape
CH4 = c_img.get_fdata()[crop].astype(np.float32)
mpm = sum(nib.load(MPM + f).get_fdata()[crop] for f in os.listdir(MPM)
          if f.endswith(".nii.gz") and "MPMAtlas" in f and f.count("_") > 3 and not f.startswith("._"))
# l and r files hold disjoint voxels, so summing them keeps the label values
CH123 = (np.isin(mpm, [79])).astype(np.float32)
assert CH4.shape == T.shape
PVE_CSF = nib.load(os.path.join(os.path.dirname(__file__), "fast_template", "Template_T1_brain_withCSF_pve_csf.nii.gz")).get_fdata()[crop]
csf_mask = (PVE_CSF > 0.5) | ~BMc
n_before = int((CH4 > 0).sum())
CH4[csf_mask] = 0
print("CSF removal: Ch4 voxels", n_before, "->", int((CH4 > 0).sum()))

lo, hi = np.percentile(T[T > 0], [0.5, 99.7])
T8 = np.clip((T - lo) / (hi - lo), 0, 1) ** 0.9
PMAX = float(CH4.max())
print("crop", [(c.start, c.stop) for c in crop], "Ch4 max", PMAX, "Ch123 vox", int(CH123.sum()))

def world(i, j, k):
    return (t_img.affine @ np.array([i, j, k, 1.0]))[:3]

def plane(vol, axis, idx):
    """Return a 2D display array: rows = superior->inferior (or anterior->posterior for axial),
    cols = subject left->right? No: neurological view, image left = subject left (x increasing to the right)."""
    if axis == "cor":
        s = vol[:, idx, :]          # (x, z)
        return s.T[::-1, :]         # rows z top->bottom, cols x left->right
    if axis == "ax":
        s = vol[:, :, idx]          # (x, y)
        return s.T[::-1, :]         # rows y anterior at top
    s = vol[idx, :, :]              # (y, z)
    return s.T[::-1, :]             # rows z, cols y posterior->anterior

def t1_png(a):
    big = ndimage.zoom(a, UP, order=3)
    return Image.fromarray((np.clip(big, 0, 1) * 255).astype(np.uint8), "L")

def lbl_png(axis, idx):
    r = (plane(T8, axis, idx) > 0.02).astype(np.float32)
    g = plane(CH4, axis, idx) / PMAX
    b = plane(CH123, axis, idx)
    return Image.fromarray((np.stack([r, g, b], -1) * 255).round().astype(np.uint8), "RGB")

# ---------- coronal stack through the Ch4 extent ----------
ys = np.where(CH4.max(axis=(0, 2)) > 0.05)[0]
jlist = list(range(ys.min() - 2, ys.max() + 3))
step = max(1, int(np.ceil(len(jlist) / 25)))
jlist = jlist[::step][:25]
n = len(jlist); cols = 5; rows = int(np.ceil(n / cols))
t0 = t1_png(plane(T8, "cor", jlist[0])); W, H = t0.size
l0 = lbl_png("cor", jlist[0]); w, h = l0.size
spr_t = Image.new("L", (W * cols, H * rows)); spr_l = Image.new("RGB", (w * cols, h * rows))
slices = []
for s, j in enumerate(jlist):
    r_, c_ = divmod(s, cols)
    spr_t.paste(t1_png(plane(T8, "cor", j)), (c_ * W, r_ * H))
    spr_l.paste(lbl_png("cor", j), (c_ * w, r_ * h))
    slices.append({"y": int(round(world(0, j, 0)[1])), "j": int(j),
                   "ch4_max": round(float(CH4[:, j, :].max()), 3)})
spr_t.save(os.path.join(OUT, "coronal_t1.jpg"), quality=86, optimize=True, progressive=True)
spr_l.save(os.path.join(OUT, "coronal_lbl.png"), optimize=True)

# ---------- axial + sagittal through the (left) Ch4 peak ----------
pk = np.unravel_index(np.argmax(CH4), CH4.shape)
views = {}
for axis, idx, name in [("ax", pk[2], "axial"), ("sag", pk[0], "sagittal")]:
    t1_png(plane(T8, axis, idx)).save(os.path.join(OUT, f"{name}_t1.jpg"), quality=88, optimize=True)
    im = lbl_png(axis, idx); im.save(os.path.join(OUT, f"{name}_lbl.png"), optimize=True)
    views[name] = {"w": im.size[0], "h": im.size[1], "index": int(idx)}
wpk = world(*pk)
meta = {
    "shape": [int(sx), int(sy), int(sz)], "affine": np.round(t_img.affine, 3).tolist(),
    "up": UP, "pmax": round(PMAX, 4),
    "coronal": {"w": w, "h": h, "W": W, "H": H, "cols": cols, "rows": rows, "slices": slices},
    "axial": {**views["axial"], "z": int(round(wpk[2]))},
    "sagittal": {**views["sagittal"], "x": int(round(wpk[0]))},
    "peak_mni": [int(round(v)) for v in wpk],
    "sources": {
        "template": "MNI152NLin2009cAsym T1, 1 mm (CAT12 template)",
        "ch4": "Julich-Brain probabilistic cytoarchitectonic map of Ch 4 (Zaborszky et al., 2008)",
        "ch123": "Julich-Brain v3 maximum probability map, Ch 1-3",
    },
}
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
print("coronal slices", n, "y", slices[0]["y"], "->", slices[-1]["y"], "peak", meta["peak_mni"],
      "sprite", spr_t.size, spr_l.size)

# ---------- static card image (coronal through peak, YlOrRd) ----------
j = pk[1]
bg = plane(T8, "cor", j); ov = plane(CH4, "cor", j); ch = plane(CH123, "cor", j)
zz, xx = np.nonzero(bg > 0.02)
fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
fig.patch.set_facecolor("#0b1b33"); ax.set_facecolor("#0b1b33")
ax.imshow(bg, cmap="gray", interpolation="bicubic", vmin=0, vmax=1)
ax.imshow(np.ma.masked_less(ov, 0.05), cmap="YlOrRd", vmin=0.05, vmax=PMAX, interpolation="bilinear", alpha=0.95)
if ch.any():
    ax.contour(ndimage.gaussian_filter(ch, 0.6), levels=[0.5], colors=["#1B7837"], linewidths=1.2)
cz, cx = np.mean(np.nonzero(ov > 0.05), axis=1)
ax.set_xlim(cx - 60, cx + 60); ax.set_ylim(cz + 40, cz - 40); ax.axis("off")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"ch4_card.{ext}"), dpi=290, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
