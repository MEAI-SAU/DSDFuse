from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT.parent / "visio_assets"
OUT = ROOT / "updated_figures"
OUT.mkdir(parents=True, exist_ok=True)

COL = {
    "bg": "#f7f7f5",
    "line": "#121212",
    "struct": "#d9ebf8",
    "func": "#e5f3db",
    "forward": "#d9ebf8",
    "noise": "#7f7f7f",
    "diff": "#fff1cc",
    "bank": "#efe2f7",
    "fusion": "#fde8d5",
    "decoder": "#bccae7",
    "refine": "#ece6f7",
    "white": "#ffffff",
    "yellow": "#ffd21e",
    "yellow2": "#ffea55",
    "yellow3": "#f5c400",
}


def rbox(ax, x, y, w, h, text="", fc=None, ec=None, lw=1.6, rs=0.018,
         fs=10, weight="normal", family="DejaVu Serif", ls="-", z=2):
    patch = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={rs}",
        linewidth=lw,
        edgecolor=ec or COL["line"],
        facecolor=fc or COL["white"],
        linestyle=ls,
        zorder=z,
    )
    ax.add_patch(patch)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight=weight, family=family, zorder=z + 1, linespacing=1.1)
    return patch


def rect(ax, x, y, w, h, text="", fc=None, ec=None, lw=1.2, fs=10, weight="normal"):
    patch = patches.Rectangle((x, y), w, h, linewidth=lw, edgecolor=ec or COL["line"],
                              facecolor=fc or COL["white"], zorder=2)
    ax.add_patch(patch)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight=weight, family="DejaVu Serif", zorder=3)
    return patch


def arrow(ax, p1, p2, lw=1.6, ls="-", head=11, rad=0.0, color=None, z=4):
    ax.annotate(
        "",
        xy=p2, xytext=p1,
        arrowprops=dict(
            arrowstyle="-|>",
            lw=lw,
            linestyle=ls,
            color=color or COL["line"],
            mutation_scale=head,
            shrinkA=0,
            shrinkB=0,
            connectionstyle=f"arc3,rad={rad}",
        ),
        zorder=z,
    )


def latent(ax, x, y, color, label, scale=1.0):
    w, h = 0.026 * scale, 0.060 * scale
    dx, dy = 0.008 * scale, 0.008 * scale
    for i in range(2):
        patch = patches.Rectangle(
            (x + i * dx, y + i * dy), w, h,
            linewidth=1.0, edgecolor=COL["line"], facecolor=color, zorder=2 + i
        )
        ax.add_patch(patch)
    ax.text(x + w / 2 + dx, y - 0.018 * scale, label, ha="center", va="top",
            fontsize=10, family="DejaVu Serif", zorder=4)


def icon_bank(ax, x, y):
    rbox(ax, x, y, 0.090, 0.070, "Selected\nprocess bank", fc=COL["white"], fs=8.7)
    bars = [(0.014, 0.040, COL["yellow2"]), (0.030, 0.031, COL["yellow"]),
            (0.046, 0.022, COL["yellow3"])]
    for dx, h, c in bars:
        rect(ax, x + dx, y + 0.020, 0.014, h, fc=c, lw=0.8)


def image(ax, path, x, y, zoom=0.13):
    if not path.exists():
        rect(ax, x - 0.03, y - 0.03, 0.06, 0.06, "img", fs=7)
        return
    img = mpimg.imread(path)
    box = AnnotationBbox(
        OffsetImage(img, zoom=zoom), (x, y), frameon=True, pad=0.0,
        bboxprops=dict(edgecolor=COL["line"], linewidth=1.0)
    )
    ax.add_artist(box)


def small_stage_row(ax, x, y, labels, fills, w=0.350, h=0.050):
    step = w / len(labels)
    for i, (lab, fc) in enumerate(zip(labels, fills)):
        rbox(ax, x + i * step, y, step - 0.008, h, lab, fc=fc, fs=8.5,
             weight="bold" if "CSI" in lab or "Mamba" in lab else "normal", rs=0.010)
        if i < len(labels) - 1:
            arrow(ax, (x + i * step + step - 0.008, y + h / 2), (x + (i + 1) * step, y + h / 2), lw=1.0, head=8)


def draw_framework():
    fig = plt.figure(figsize=(14.8, 8.8), dpi=260, facecolor=COL["bg"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    outer = rbox(ax, 0.03, 0.11, 0.94, 0.82, fc=COL["bg"], lw=1.8, rs=0.03)
    ax.plot([0.03, 0.97], [0.66, 0.66], color=COL["line"], lw=1.2)
    ax.text(0.50, 0.965, "Overall Framework of DSDFuse", ha="center", va="top",
            fontsize=22, fontweight="bold", family="DejaVu Sans")
    ax.text(0.50, 0.675, "Diffusion Forward Process", ha="center", va="bottom",
            fontsize=14, family="DejaVu Serif")
    ax.text(0.50, 0.125, "Diffusion Reverse Process and Inference", ha="center", va="bottom",
            fontsize=14, family="DejaVu Serif")

    # Inputs and encoder
    rect(ax, 0.05, 0.81, 0.040, 0.048, r"$I_s$", fs=14)
    rect(ax, 0.05, 0.73, 0.040, 0.048, r"$I_f$", fs=14)
    enc = patches.Polygon([[0.12, 0.705], [0.12, 0.88], [0.22, 0.835], [0.22, 0.75]],
                          closed=True, facecolor=COL["decoder"], edgecolor=COL["line"], linewidth=1.2, zorder=2)
    ax.add_patch(enc)
    ax.text(0.165, 0.792, "Encoder", ha="center", va="center", fontsize=14, family="DejaVu Serif")
    arrow(ax, (0.09, 0.834), (0.12, 0.834), lw=1.3)
    arrow(ax, (0.09, 0.754), (0.12, 0.754), lw=1.3)

    latent(ax, 0.28, 0.755, COL["forward"], r"$z_s^0$")
    latent(ax, 0.31, 0.755, "#c8e4c0", r"$z_f^0$")
    arrow(ax, (0.22, 0.792), (0.28, 0.792), lw=1.3)
    ax.text(0.40, 0.812, r"$q(z_t|z_0)$", fontsize=13, family="DejaVu Serif")
    ax.text(0.47, 0.795, r"$\cdots$", fontsize=17, family="DejaVu Serif")
    arrow(ax, (0.35, 0.792), (0.40, 0.792), lw=1.3)
    arrow(ax, (0.51, 0.792), (0.56, 0.792), lw=1.3)
    latent(ax, 0.56, 0.755, COL["noise"], r"$z_s^t$")
    latent(ax, 0.59, 0.755, "#9b9b9b", r"$z_f^t$")
    ax.text(0.68, 0.812, r"$q(z_T|z_t)$", fontsize=13, family="DejaVu Serif")
    ax.text(0.75, 0.795, r"$\cdots$", fontsize=17, family="DejaVu Serif")
    arrow(ax, (0.63, 0.792), (0.68, 0.792), lw=1.3)
    arrow(ax, (0.79, 0.792), (0.84, 0.792), lw=1.3)
    latent(ax, 0.84, 0.755, "#a8d47a", r"$z_s^T$")
    latent(ax, 0.87, 0.755, "#8bc85b", r"$z_f^T$")

    # Bank source in forward band
    icon_bank(ax, 0.84, 0.845)
    ax.plot([0.86, 0.86, 0.34, 0.34], [0.845, 0.925, 0.925, 0.815], color=COL["line"], lw=1.0, ls="--")
    arrow(ax, (0.86, 0.845), (0.86, 0.844), lw=1.0, ls="--", head=8)

    # Reverse process inputs
    latent(ax, 0.05, 0.50, "#a8d47a", r"$z_s^T$", scale=1.05)
    latent(ax, 0.05, 0.30, "#8bc85b", r"$z_f^T$", scale=1.05)

    # DSB upper/lower
    rbox(ax, 0.11, 0.46, 0.56, 0.14, "Dual-Stream Denoising Backbone (DSB)", fc=COL["struct"],
         fs=11.2, weight="bold", rs=0.018)
    rbox(ax, 0.11, 0.26, 0.56, 0.14, "Dual-Stream Denoising Backbone (DSB)", fc=COL["func"],
         fs=11.2, weight="bold", rs=0.018)
    ax.text(0.14, 0.565, "structure stream", fontsize=8, family="DejaVu Serif")
    ax.text(0.14, 0.365, "function stream", fontsize=8, family="DejaVu Serif")
    labels = ["Stage 0\nLocal", "Stage 1\nLocal", "Stage 2\nMamba", "Stage 3\nMamba", "CSI + RB"]
    fills = [COL["white"], COL["white"], "#d1e5f5", "#bdd8ef", COL["white"]]
    small_stage_row(ax, 0.15, 0.50, labels, fills)
    small_stage_row(ax, 0.15, 0.30, labels, fills)
    arrow(ax, (0.09, 0.536), (0.11, 0.535), lw=1.2)
    arrow(ax, (0.09, 0.336), (0.11, 0.335), lw=1.2)
    ax.annotate("", xy=(0.42, 0.345), xytext=(0.42, 0.545),
                arrowprops=dict(arrowstyle="<->", lw=1.1, color=COL["line"], mutation_scale=9))
    ax.text(0.43, 0.44, "CSI", fontsize=8.5, fontweight="bold", family="DejaVu Sans")

    # Timestep and scheduler
    rbox(ax, 0.21, 0.41, 0.11, 0.038, r"Timestep $t$", fc=COL["diff"], fs=9.2, weight="bold", rs=0.010)
    arrow(ax, (0.265, 0.448), (0.265, 0.46), lw=1.0, ls="--", head=7)
    arrow(ax, (0.265, 0.41), (0.265, 0.40), lw=1.0, ls="--", head=7)
    rbox(ax, 0.54, 0.46, 0.09, 0.052, "Scheduler\nupdate", fc=COL["diff"], fs=8.2, weight="bold", rs=0.010)
    rbox(ax, 0.54, 0.26, 0.09, 0.052, "Scheduler\nupdate", fc=COL["diff"], fs=8.2, weight="bold", rs=0.010)
    arrow(ax, (0.67, 0.532), (0.54, 0.486), lw=1.0, head=8)
    arrow(ax, (0.67, 0.332), (0.54, 0.286), lw=1.0, head=8)
    ax.plot([0.585, 0.585, 0.095, 0.095, 0.11], [0.46, 0.43, 0.43, 0.51, 0.51], color=COL["line"], lw=0.9)
    ax.plot([0.585, 0.585, 0.095, 0.095, 0.11], [0.26, 0.23, 0.23, 0.31, 0.31], color=COL["line"], lw=0.9)
    ax.text(0.515, 0.395, r"$z_t \rightarrow z_{t-1}$", fontsize=9.7, family="DejaVu Serif")

    # Selected bank
    rbox(ax, 0.39, 0.40, 0.11, 0.046, r"$B_k$" + "\nStages 1,2,3", fc=COL["bank"], fs=8.7,
         weight="bold", rs=0.010)
    for x in [0.24, 0.31, 0.39]:
        arrow(ax, (x, 0.50), (0.425, 0.446), lw=0.9, ls="--", head=6)
        arrow(ax, (x, 0.35), (0.425, 0.418), lw=0.9, ls="--", head=6)

    # Final latents
    latent(ax, 0.70, 0.49, COL["struct"], r"$\hat z_s^0$", scale=1.02)
    latent(ax, 0.70, 0.29, COL["func"], r"$\hat z_f^0$", scale=1.02)
    arrow(ax, (0.67, 0.532), (0.70, 0.53), lw=1.2)
    arrow(ax, (0.67, 0.332), (0.70, 0.33), lw=1.2)

    # Fusion, decoder, sgrr
    rbox(ax, 0.77, 0.37, 0.10, 0.14, "MS-RGSF\nHead", fc=COL["fusion"], fs=11, weight="bold", rs=0.016)
    arrow(ax, (0.735, 0.53), (0.77, 0.47), lw=1.2)
    arrow(ax, (0.735, 0.33), (0.77, 0.41), lw=1.2)
    arrow(ax, (0.50, 0.423), (0.77, 0.44), lw=1.0, ls="--", head=8)
    latent(ax, 0.90, 0.415, "#a6cce8", r"$z_{fuse}$")
    arrow(ax, (0.87, 0.44), (0.90, 0.44), lw=1.2)

    encpoly = patches.Polygon([[0.84, 0.18], [0.84, 0.28], [0.94, 0.25], [0.94, 0.21]],
                              closed=True, facecolor=COL["decoder"], edgecolor=COL["line"], linewidth=1.1, zorder=2)
    ax.add_patch(encpoly)
    ax.text(0.885, 0.23, "Decoder", ha="center", va="center", fontsize=12.5, family="DejaVu Serif", zorder=3)
    rbox(ax, 0.95, 0.20, 0.05, 0.055, "SGRR", fc=COL["refine"], fs=8.8, weight="bold", rs=0.010)
    arrow(ax, (0.92, 0.405), (0.865, 0.28), lw=1.0, rad=-0.12)
    arrow(ax, (0.94, 0.23), (0.95, 0.23), lw=1.2)
    image(ax, ASSETS / "PET_MRI_Output_Fusion.png", 0.975, 0.135, zoom=0.11)
    arrow(ax, (0.998, 0.23), (0.998, 0.16), lw=1.1)
    ax.text(0.975, 0.075, "Fused image", ha="center", fontsize=9.5, fontweight="bold", family="DejaVu Sans")

    # Legend
    leg = rbox(ax, 0.03, 0.02, 0.94, 0.055, fc=COL["white"], lw=1.0, rs=0.008)
    lx, ly = 0.05, 0.048
    latent(ax, lx, ly - 0.018, COL["struct"], "", scale=0.70)
    ax.text(lx + 0.050, ly, "latent feature", va="center", fontsize=7.8, family="DejaVu Serif")
    rbox(ax, lx + 0.17, ly - 0.016, 0.055, 0.028, "DSB", fc=COL["struct"], fs=7.2, weight="bold", rs=0.005)
    ax.text(lx + 0.235, ly, "denoising backbone", va="center", fontsize=7.8, family="DejaVu Serif")
    rbox(ax, lx + 0.40, ly - 0.016, 0.070, 0.028, "MS-RGSF", fc=COL["fusion"], fs=6.8, weight="bold", rs=0.005)
    ax.text(lx + 0.480, ly, "fusion head", va="center", fontsize=7.8, family="DejaVu Serif")
    rbox(ax, lx + 0.58, ly - 0.016, 0.040, 0.028, r"$B_k$", fc=COL["bank"], fs=7.8, weight="bold", rs=0.005)
    ax.text(lx + 0.630, ly, "selected process bank", va="center", fontsize=7.8, family="DejaVu Serif")
    arrow(ax, (lx + 0.77, ly), (lx + 0.81, ly), lw=1.0, head=6)
    ax.text(lx + 0.82, ly, "feature flow", va="center", fontsize=7.8, family="DejaVu Serif")
    ax.plot([lx + 0.90, lx + 0.94], [ly, ly], color=COL["line"], lw=1.0, ls="--")
    ax.text(lx + 0.95, ly, "conditioning / bank injection", va="center", fontsize=7.8, family="DejaVu Serif")

    out_png = OUT / "fig1_dsdfuse_top_journal_framework_clean.png"
    out_svg = OUT / "fig1_dsdfuse_top_journal_framework_clean.svg"
    out_pdf = OUT / "fig1_dsdfuse_top_journal_framework_clean.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(out_png)
    print(out_svg)
    print(out_pdf)


if __name__ == "__main__":
    draw_framework()
