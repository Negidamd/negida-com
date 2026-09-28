"""Ch4 illustration for the PD-MCI Ch4 (Parkinsonism Relat Disord 2025) tile.

Not a figure from the article: MNI152NLin2009cAsym T1 (CAT12 Template_T1, CSF-inclusive brainmask)
with the Julich-Brain v3 maximum probability map label "Ch 4 (Basal Forebrain)" (grayvalue 13, both
hemispheres) filled in solid red. Ch4 voxels with FSL FAST CSF PVE > 0.5 are removed (same rule as the explorer).
"""
import os
import numpy as np, nibabel as nib
from nibabel.processing import resample_from_to
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

HERE = os.path.dirname(__file__)
CAT = "/Users/ahmednegida/matlab_toolboxes/spm12_toolbox_backup/cat12/templates_MNI152NLin2009cAsym/"
MPM = ("/Users/ahmednegida/Library/Mobile Documents/com~apple~CloudDocs/3-resources/brain-atlases/"
       "hbp-d000001_jubrain-cytoatlas_pub-MPM-collections-26/MPM/")
OUT = os.path.join(HERE, "..", "assets", "featured")

full = nib.load(CAT + "Template_T1.nii.gz"); T = full.get_fdata()
BM = resample_from_to(nib.load(CAT + "brainmask.nii.gz"), full, order=1).get_fdata() > 0.5
T = np.where(BM, T, 0)
lab = np.zeros(T.shape, bool)
for f in os.listdir(MPM):
    if f.endswith(".nii.gz") and "MPMAtlas" in f and not f.startswith("._"):
        img = nib.load(MPM + f); assert np.allclose(img.affine, full.affine)
        lab |= img.get_fdata() == 13
csf = nib.load(os.path.join(HERE, "fast_template", "Template_T1_brain_withCSF_pve_csf.nii.gz")).get_fdata() > 0.5
n0 = int(lab.sum()); lab &= ~csf & BM
print("Ch4 MPM voxels", n0, "->", int(lab.sum()))

inv = np.linalg.inv(full.affine)
def vox(x, y, z): return np.round(inv @ [x, y, z, 1])[:3].astype(int)
# planes through the largest Ch4 cross-sections (left hemisphere for the sagittal view)
xs = lab.sum(axis=(1, 2)); xs[vox(0, 0, 0)[0]:] = 0
i0 = int(np.argmax(xs)); j0 = int(np.argmax(lab.sum(axis=(0, 2)))); k0 = int(np.argmax(lab.sum(axis=(0, 1))))
print("planes MNI x, y, z =", [int(round(v)) for v in (full.affine @ [i0, j0, k0, 1])[:3]])
hi = np.percentile(T[BM], 99.5)
red = ListedColormap(["#E3171B"])
views = [("sag", T[i0, :, :].T[::-1], lab[i0, :, :].T[::-1]),
         ("cor", T[:, j0, :].T[::-1], lab[:, j0, :].T[::-1]),
         ("ax", T[:, :, k0].T[::-1], lab[:, :, k0].T[::-1])]
fig, axs = plt.subplots(1, 3, figsize=(5.8, 2.2), constrained_layout=True,
                        gridspec_kw={"width_ratios": [229, 193, 193]})
for ax, (name, bg, m) in zip(axs, views):
    rows, cols = np.nonzero(bg > 0)
    ax.imshow(np.ma.masked_where(bg <= 0, bg), cmap="gray", vmin=0, vmax=hi, interpolation="bicubic")
    ax.imshow(np.ma.masked_where(~m, m), cmap=red, interpolation="nearest", alpha=1)
    ax.set_xlim(cols.min() - 2, cols.max() + 2); ax.set_ylim(rows.max() + 2, rows.min() - 2); ax.axis("off")
    if name != "sag":
        ax.text(0.02, 0.97, "L", transform=ax.transAxes, va="top", fontsize=7, fontweight="bold")
        ax.text(0.98, 0.97, "R", transform=ax.transAxes, va="top", ha="right", fontsize=7, fontweight="bold")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"ch4_red_mni.{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
