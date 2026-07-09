from pathlib import Path
from math import atan2, cos, sin, pi

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "updated_figures"
ASSETS = ROOT.parent / "visio_assets"
OUT.mkdir(parents=True, exist_ok=True)

FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/Library/Fonts/Arial.ttf"),
]
FONT_BOLD_CANDIDATES = [
    Path("C:/Windows/Fonts/arialbd.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/Library/Fonts/Arial Bold.ttf"),
]


def find_font(candidates):
    for path in candidates:
        if path.exists():
            return path
    return None


FONT = find_font(FONT_CANDIDATES)
FONT_BOLD = find_font(FONT_BOLD_CANDIDATES) or FONT


def ft(size, bold=False):
    font_path = FONT_BOLD if bold else FONT
    if font_path is None:
        return ImageFont.load_default()
    return ImageFont.truetype(str(font_path), size=size)


def text_size(draw, text, font):
    box = draw.multiline_textbbox((0, 0), text, font=font, spacing=4, align="center")
    return box[2] - box[0], box[3] - box[1]


def wrap_text(draw, text, font, max_w):
    lines = []
    for part in str(text).split("\n"):
        words = part.split(" ")
        line = ""
        for word in words:
            test = word if not line else line + " " + word
            if draw.textbbox((0, 0), test, font=font)[2] <= max_w or not line:
                line = test
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return "\n".join(lines)


def centered_text(draw, box, text, font, fill=(0, 0, 0), max_w=None):
    x0, y0, x1, y1 = box
    if max_w is None:
        max_w = x1 - x0 - 14
    text = wrap_text(draw, text, font, max_w)
    w, h = text_size(draw, text, font)
    draw.multiline_text(
        (x0 + (x1 - x0 - w) / 2, y0 + (y1 - y0 - h) / 2),
        text,
        font=font,
        fill=fill,
        spacing=4,
        align="center",
    )


def round_box(draw, box, text="", fill=(245, 245, 245), outline=(0, 0, 0), width=3,
              radius=18, font=None, bold=False, dash=False):
    if dash:
        dashed_round_rect(draw, box, radius, outline, width)
    else:
        draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
    if text:
        centered_text(draw, box, text, font or ft(24, bold), max_w=box[2] - box[0] - 18)


def dashed_round_rect(draw, box, radius, color, width=3, dash=16, gap=10):
    # A light-weight dashed rectangle; corners are approximated with a solid rounded outline.
    draw.rounded_rectangle(box, radius=radius, outline=color, width=width)
    x0, y0, x1, y1 = box
    for x in range(int(x0 + radius), int(x1 - radius), dash + gap):
        draw.line((x, y0, min(x + dash, x1 - radius), y0), fill=(245, 245, 245), width=width + 2)
        draw.line((x, y1, min(x + dash, x1 - radius), y1), fill=(245, 245, 245), width=width + 2)
    for y in range(int(y0 + radius), int(y1 - radius), dash + gap):
        draw.line((x0, y, x0, min(y + dash, y1 - radius)), fill=(245, 245, 245), width=width + 2)
        draw.line((x1, y, x1, min(y + dash, y1 - radius)), fill=(245, 245, 245), width=width + 2)


def arrow(draw, p1, p2, fill=(0, 0, 0), width=4, head=16, both=False, dash=None):
    if dash:
        draw.line((p1, p2), fill=fill, width=width)
    else:
        draw.line((p1, p2), fill=fill, width=width)
    def head_at(a, b):
        ang = atan2(b[1] - a[1], b[0] - a[0])
        pts = [
            b,
            (b[0] - head * cos(ang - pi / 7), b[1] - head * sin(ang - pi / 7)),
            (b[0] - head * cos(ang + pi / 7), b[1] - head * sin(ang + pi / 7)),
        ]
        draw.polygon(pts, fill=fill)
    head_at(p1, p2)
    if both:
        head_at(p2, p1)


def op_circle(draw, center, text, r=28, font=None):
    x, y = center
    draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255), outline=(0, 0, 0), width=3)
    centered_text(draw, (x - r, y - r, x + r, y + r), text, font or ft(24, True))


def cube(draw, x, y, w=46, h=66, fill=(184, 215, 238), label=None, outline=(0, 0, 0)):
    dx, dy = 16, -14
    draw.rectangle((x, y, x + w, y + h), fill=fill, outline=outline, width=3)
    draw.polygon([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
                 fill=tuple(max(0, c - 20) for c in fill), outline=outline)
    draw.polygon([(x, y), (x + dx, y + dy), (x + w + dx, y + dy), (x + w, y)],
                 fill=tuple(min(255, c + 18) for c in fill), outline=outline)
    if label:
        centered_text(draw, (x - 12, y + h + 8, x + w + dx + 12, y + h + 45), label, ft(20, True))


def cylinder(draw, x, y, w=76, h=92, fill=(225, 213, 240), outline=(0, 0, 0), text=""):
    eh = 22
    draw.rectangle((x, y + eh / 2, x + w, y + h - eh / 2), fill=fill, outline=None)
    draw.ellipse((x, y, x + w, y + eh), fill=fill, outline=outline, width=3)
    draw.ellipse((x, y + h - eh, x + w, y + h), fill=fill, outline=outline, width=3)
    draw.line((x, y + eh / 2, x, y + h - eh / 2), fill=outline, width=3)
    draw.line((x + w, y + eh / 2, x + w, y + h - eh / 2), fill=outline, width=3)
    if text:
        centered_text(draw, (x - 12, y + h + 4, x + w + 12, y + h + 42), text, ft(18, True))


def paste_asset(canvas, name, box):
    path = ASSETS / name
    if not path.exists():
        return
    img = Image.open(path).convert("RGB")
    x0, y0, x1, y1 = box
    img.thumbnail((x1 - x0, y1 - y0), Image.LANCZOS)
    x = int(x0 + (x1 - x0 - img.width) / 2)
    y = int(y0 + (y1 - y0 - img.height) / 2)
    canvas.paste(img, (x, y))


def legend(draw, box, items=("C", "x", "+")):
    round_box(draw, box, fill=(255, 255, 255), radius=12, width=3)
    x0, y0, x1, y1 = box
    x = x0 + 60
    y = y0 + 45
    if "C" in items:
        op_circle(draw, (x, y), "C", r=26, font=ft(22, True))
        draw.text((x + 48, y - 12), "Channel concatenation", font=ft(22), fill=(0, 0, 0))
        y += 58
    if "x" in items:
        op_circle(draw, (x, y), "x", r=23, font=ft(22, True))
        draw.text((x + 48, y - 12), "Element-wise multiplication", font=ft(20), fill=(0, 0, 0))
        y += 58
    if "+" in items:
        op_circle(draw, (x, y), "+", r=23, font=ft(22, True))
        draw.text((x + 48, y - 12), "Element-wise addition", font=ft(20), fill=(0, 0, 0))
    arrow(draw, (x1 - 250, y0 + 58), (x1 - 170, y0 + 58), width=3)
    draw.text((x1 - 140, y0 + 44), "Feature flow", font=ft(22), fill=(0, 0, 0))


def draw_overall():
    W, H = 2250, 940
    im = Image.new("RGB", (W, H), (246, 246, 246))
    d = ImageDraw.Draw(im)
    round_box(d, (280, 40, 2035, 870), fill=(246, 246, 246), radius=32, width=3)
    centered_text(d, (0, 55, W, 105), "Overall Framework of DSDFuse", ft(42, True))

    paste_asset(im, "PET_MRI_Input_MRI.png", (70, 250, 225, 390))
    paste_asset(im, "PET_MRI_Input_PET.png", (70, 455, 225, 595))
    centered_text(d, (55, 165, 240, 235), "Structural\nImage\n(MRI)", ft(21, True))
    centered_text(d, (55, 600, 240, 675), "Functional\nImage\n(PET/SPECT)", ft(21, True))

    enc = (340, 180, 750, 585)
    dsb = (780, 180, 1290, 585)
    head = (1380, 180, 1800, 675)
    round_box(d, enc, "Latent Encoding and Forward Diffusion", fill=(250, 250, 250), radius=25, font=ft(21, True))
    round_box(d, dsb, "Dual-Stream Reverse Denoising Backbone\n(DSB)", fill=(250, 250, 250), radius=25, font=ft(20, True))
    round_box(d, head, "Multi-Scale Reliability-Guided\nStructure-Function Fusion Head\n(MS-RGSF)", fill=(255, 232, 208), radius=25, font=ft(19, True))

    arrow(d, (225, 325), (455, 325))
    arrow(d, (225, 530), (455, 530))
    round_box(d, (455, 282, 505, 368), "E_s", fill=(235, 245, 255), radius=0, font=ft(19))
    round_box(d, (460, 487, 510, 573), "E_f", fill=(235, 255, 235), radius=0, font=ft(19))
    cube(d, 565, 292, 38, 54, (194, 229, 242), "z_s^0")
    cube(d, 568, 497, 38, 54, (207, 240, 197), "z_f^0")
    arrow(d, (505, 325), (565, 325))
    arrow(d, (510, 530), (568, 530))
    d.text((615, 306), "q(z_t|z_0)", font=ft(18), fill=(0, 0, 0))
    d.text((618, 511), "q(z_t|z_0)", font=ft(18), fill=(0, 0, 0))
    cube(d, 675, 292, 38, 54, (170, 204, 230), "z_s^t")
    cube(d, 678, 497, 38, 54, (177, 220, 163), "z_f^t")
    arrow(d, (610, 325), (675, 325))
    arrow(d, (613, 530), (678, 530))
    arrow(d, (735, 325), (820, 325))
    arrow(d, (738, 530), (820, 530))

    stream1 = (830, 235, 1242, 365)
    stream2 = (830, 440, 1242, 570)
    round_box(d, stream1, fill=(184, 215, 238), radius=18, width=3)
    round_box(d, stream2, fill=(218, 242, 208), radius=18, width=3)
    for y, color in [(300, (184, 215, 238)), (505, (218, 242, 208))]:
        xs = [852, 958, 1064, 1170]
        labels = ["Local\nMixer", "Mamba\n(Stage 2-3)", "CSI", "RB"]
        for x, lab in zip(xs, labels):
            round_box(d, (x, y - 38, x + 76, y + 38), lab, fill=(255, 255, 255), radius=16, font=ft(18))
        for a, b in zip(xs[:-1], xs[1:]):
            arrow(d, (a + 76, y), (b, y), width=3)
    arrow(d, (1102, 365), (1102, 440), width=3, both=True)

    cube(d, 1310, 292, 38, 54, (184, 215, 238), "z_s^0_hat")
    cube(d, 1310, 497, 38, 54, (218, 242, 208), "z_f^0_hat")
    arrow(d, (1242, 300), (1310, 320))
    arrow(d, (1242, 505), (1310, 525))
    arrow(d, (1362, 320), (1398, 320))
    arrow(d, (1362, 525), (1398, 525))

    round_box(d, (960, 635, 1135, 715), "Timestep\nConditioning t", fill=(255, 245, 220), radius=12, font=ft(21))
    d.line((1048, 635, 1048, 585), fill=(0, 0, 0), width=3)
    arrow(d, (1048, 635), (1048, 585), width=1)
    round_box(d, (1105, 635, 1320, 715), "Selective Multi-Scale\nProcess Bank B_k\nStages: 1, 2, 3", fill=(236, 230, 245), radius=12, font=ft(18, True))
    arrow(d, (1190, 585), (1190, 635), width=3)
    arrow(d, (1320, 675), (1398, 560), width=3)

    round_box(d, (1400, 282, 1480, 330), "S-Proj", fill=(255, 224, 200), radius=10, font=ft(19))
    round_box(d, (1400, 485, 1480, 533), "F-Proj", fill=(230, 250, 225), radius=10, font=ft(19))
    round_box(d, (1400, 595, 1480, 643), "MS Bank\nAgg.", fill=(236, 230, 245), radius=10, font=ft(16))
    op_circle(d, (1515, 420), "C", r=18, font=ft(17, True))
    arrow(d, (1480, 306), (1515, 402), width=3)
    arrow(d, (1480, 509), (1515, 420), width=3)
    arrow(d, (1480, 619), (1515, 438), width=3)
    round_box(d, (1550, 345, 1660, 495), "Reliability\nEstimator\nR_s, R_f, R_b", fill=(255, 255, 255), radius=12, font=ft(18, True))
    round_box(d, (1695, 360, 1775, 480), "Conv\n1x1", fill=(235, 139, 75), radius=10, font=ft(20))
    arrow(d, (1533, 420), (1550, 420), width=3)
    arrow(d, (1660, 420), (1695, 420), width=3)
    cube(d, 1815, 386, 36, 55, (160, 200, 230), "Z_fuse")
    arrow(d, (1775, 420), (1815, 415), width=3)
    round_box(d, (1900, 520, 2010, 595), "SGRR", fill=(236, 232, 246), radius=12, font=ft(19))
    round_box(d, (1900, 365, 1950, 455), "D", fill=(245, 245, 245), radius=0, font=ft(20))
    arrow(d, (1865, 420), (1900, 410), width=3)
    arrow(d, (1925, 455), (1955, 520), width=3)
    arrow(d, (2010, 558), (2090, 558), width=3)
    paste_asset(im, "PET_MRI_Output_Fusion.png", (2080, 485, 2220, 635))
    centered_text(d, (2070, 650, 2235, 705), "Fused Image", ft(21, True))

    legend(d, (1230, 760, 1905, 850), items=("C",))
    im.save(OUT / "fig1_overall_framework_ms_rgsf.png", quality=95)


def draw_overall_process():
    W, H = 2600, 1240
    im = Image.new("RGB", (W, H), (246, 246, 246))
    d = ImageDraw.Draw(im)
    round_box(d, (35, 35, W - 35, H - 35), fill=(246, 246, 246), radius=28, width=3)
    centered_text(d, (0, 45, W, 105), "Overall Framework of DSDFuse", ft(44, True))

    # Inputs
    paste_asset(im, "PET_MRI_Input_MRI.png", (60, 255, 220, 420))
    paste_asset(im, "PET_MRI_Input_PET.png", (60, 555, 220, 720))
    centered_text(d, (45, 165, 240, 235), "Structural Image\n(MRI)", ft(22, True))
    centered_text(d, (45, 725, 240, 785), "Functional Image\n(PET/SPECT)", ft(22, True))

    # Encoder and clean latents.
    round_box(d, (300, 225, 520, 400), "Structural\nLatent Encoder\nE_s", fill=(232, 244, 255), radius=16, font=ft(23, True))
    round_box(d, (300, 525, 520, 700), "Functional\nLatent Encoder\nE_f", fill=(238, 250, 232), radius=16, font=ft(23, True))
    arrow(d, (220, 337), (300, 337), width=4)
    arrow(d, (220, 637), (300, 637), width=4)
    cube(d, 590, 292, 55, 76, (184, 215, 238), "z_s^0")
    cube(d, 590, 592, 55, 76, (218, 242, 208), "z_f^0")
    arrow(d, (520, 337), (590, 337), width=4)
    arrow(d, (520, 637), (590, 637), width=4)

    # Forward diffusion.
    fwd = (710, 210, 1060, 720)
    round_box(d, fwd, "Forward Diffusion", fill=(255, 250, 242), radius=22, font=ft(26, True))
    round_box(d, (755, 285, 885, 385), "Add noise\nq(z_t | z_0)", fill=(255, 245, 220), radius=12, font=ft(20, True))
    round_box(d, (755, 585, 885, 685), "Add noise\nq(z_t | z_0)", fill=(255, 245, 220), radius=12, font=ft(20, True))
    round_box(d, (925, 315, 1025, 365), "epsilon_s", fill=(255, 255, 255), radius=8, font=ft(18))
    round_box(d, (925, 615, 1025, 665), "epsilon_f", fill=(255, 255, 255), radius=8, font=ft(18))
    arrow(d, (660, 337), (755, 337), width=4)
    arrow(d, (660, 637), (755, 637), width=4)
    arrow(d, (885, 337), (925, 337), width=3)
    arrow(d, (885, 637), (925, 637), width=3)
    cube(d, 1105, 292, 55, 76, (155, 190, 225), "z_s^t")
    cube(d, 1105, 592, 55, 76, (176, 220, 160), "z_f^t")
    arrow(d, (1025, 337), (1105, 337), width=4)
    arrow(d, (1025, 637), (1105, 637), width=4)

    # Reverse diffusion panel.
    rev = (1220, 165, 1795, 765)
    round_box(d, rev, "Reverse Diffusion Process", fill=(250, 250, 250), radius=24, font=ft(27, True))
    centered_text(d, (1240, 212, 1775, 250), "for t = T ... 1", ft(22, True))
    round_box(d, (1285, 285, 1735, 615), "Dual-Stream Denoising Backbone (DSB)", fill=(255, 255, 255), radius=20, font=ft(22, True))
    round_box(d, (1325, 350, 1695, 445), "Structure stream: Local Mixer -> Stage-selective Mamba -> CSI -> RB", fill=(224, 239, 252), radius=14, font=ft(18))
    round_box(d, (1325, 485, 1695, 580), "Function stream: Local Mixer -> Stage-selective Mamba -> CSI -> RB", fill=(238, 250, 232), radius=14, font=ft(18))
    arrow(d, (1168, 337), (1285, 392), width=4)
    arrow(d, (1168, 637), (1285, 535), width=4)
    round_box(d, (1250, 675, 1450, 735), "Scheduler update\nz_t -> z_{t-1}", fill=(255, 245, 220), radius=10, font=ft(18, True))
    arrow(d, (1510, 615), (1350, 675), width=3)
    d.line((1350, 735, 1350, 805, 1195, 805, 1195, 485), fill=(0, 0, 0), width=3)
    arrow(d, (1195, 485), (1285, 485), width=3)
    centered_text(d, (1190, 810, 1450, 855), "iterative denoising loop", ft(18, True))
    round_box(d, (1490, 665, 1750, 735), "Predicted noise / denoised\nstructure-function latents", fill=(245, 245, 245), radius=10, font=ft(17))

    # Process bank beneath reverse diffusion.
    bank = (1290, 835, 1750, 970)
    round_box(d, bank, "Selective Multi-Scale Process Feature Bank B_k", fill=(236, 230, 245), radius=16, font=ft(22, True))
    round_box(d, (1325, 895, 1425, 945), "Stage 1", fill=(255, 255, 255), radius=8, font=ft(17, True))
    round_box(d, (1460, 895, 1560, 945), "Stage 2", fill=(255, 255, 255), radius=8, font=ft(17, True))
    round_box(d, (1595, 895, 1695, 945), "Stage 3", fill=(255, 255, 255), radius=8, font=ft(17, True))
    for x in [1375, 1510, 1645]:
        d.line((x, 615, x, 835), fill=(0, 0, 0), width=2)
        arrow(d, (x, 770), (x, 835), width=2)
    centered_text(d, (1260, 785, 1810, 825), "one representative feature per selected stage", ft(18))

    # Final denoised latents.
    cube(d, 1865, 300, 55, 76, (184, 215, 238), "z_s^0_hat")
    cube(d, 1865, 600, 55, 76, (218, 242, 208), "z_f^0_hat")
    arrow(d, (1795, 392), (1865, 337), width=4)
    arrow(d, (1795, 535), (1865, 637), width=4)

    # MS-RGSF head, decoder, SGRR.
    head = (1980, 235, 2320, 740)
    round_box(d, head, "MS-RGSF Head", fill=(255, 232, 208), radius=22, font=ft(28, True))
    round_box(d, (2020, 300, 2110, 355), "S-Proj", fill=(224, 239, 252), radius=10, font=ft(19))
    round_box(d, (2020, 600, 2110, 655), "F-Proj", fill=(238, 250, 232), radius=10, font=ft(19))
    round_box(d, (2020, 485, 2140, 550), "MS Bank\nAggregation", fill=(236, 230, 245), radius=10, font=ft(16, True))
    op_circle(d, (2185, 475), "C", r=20, font=ft(17, True))
    round_box(d, (2215, 390, 2300, 560), "Reliability\nEstimator\nR_s,R_f,R_b\n+\nweighted fusion", fill=(255, 255, 255), radius=12, font=ft(16, True))
    arrow(d, (1930, 337), (2020, 327), width=4)
    arrow(d, (1930, 637), (2020, 627), width=4)
    arrow(d, (1750, 905), (2020, 520), width=3)
    arrow(d, (2110, 327), (2185, 455), width=3)
    arrow(d, (2110, 627), (2185, 475), width=3)
    arrow(d, (2140, 520), (2185, 495), width=3)
    arrow(d, (2205, 475), (2215, 475), width=3)
    cube(d, 2365, 455, 55, 76, (160, 200, 230), "z_fuse")
    arrow(d, (2320, 475), (2365, 492), width=4)

    round_box(d, (2460, 420, 2520, 520), "D", fill=(245, 245, 245), radius=0, font=ft(26))
    arrow(d, (2435, 492), (2460, 470), width=4)
    round_box(d, (2365, 650, 2515, 735), "SGRR\nstructure-guided\nrefinement", fill=(236, 232, 246), radius=12, font=ft(18, True))
    arrow(d, (2490, 520), (2440, 650), width=3)
    arrow(d, (2515, 690), (2550, 690), width=4)
    paste_asset(im, "PET_MRI_Output_Fusion.png", (2440, 780, 2580, 930))
    centered_text(d, (2410, 935, 2600, 985), "Fused Image", ft(22, True))
    d.line((2550, 690, 2550, 780), fill=(0, 0, 0), width=4)
    arrow(d, (2550, 740), (2550, 780), width=4)

    legend(d, (840, 1025, 1700, 1170), items=("C", "x", "+"))
    im.save(OUT / "fig1_overall_framework_process.png", quality=95)


def draw_overall_clean_paper():
    W, H = 2300, 1500
    im = Image.new("RGB", (W, H), (246, 246, 246))
    d = ImageDraw.Draw(im)
    margin = 45
    round_box(d, (margin, 35, W - margin, H - 185), fill=(246, 246, 246), radius=30, width=3)
    d.line((margin, 435, W - margin, 435), fill=(0, 0, 0), width=3)
    centered_text(d, (0, 52, W, 105), "Overall Framework of DSDFuse", ft(42, True))

    # ---------- Forward diffusion ----------
    centered_text(d, (0, 390, W, 430), "Diffusion Forward Process", ft(30, True))
    round_box(d, (130, 165, 225, 250), "I_s", fill=(255, 255, 255), radius=0, font=ft(30, True))
    round_box(d, (130, 280, 225, 365), "I_f", fill=(255, 255, 255), radius=0, font=ft(30, True))
    round_box(d, (305, 150, 545, 380), "Latent\nEncoder\nE_s / E_f", fill=(196, 211, 238), radius=0, font=ft(31))
    arrow(d, (225, 207), (305, 235), width=4)
    arrow(d, (225, 322), (305, 295), width=4)
    cube(d, 640, 195, 54, 82, (200, 225, 242), "z_s^0")
    cube(d, 710, 195, 54, 82, (218, 242, 208), "z_f^0")
    centered_text(d, (630, 295, 800, 335), "clean latents", ft(20))
    arrow(d, (545, 265), (640, 236), width=4)

    d.text((870, 205), "q(z_t | z_0)", font=ft(28), fill=(0, 0, 0))
    d.text((1015, 218), "...", font=ft(35, True), fill=(0, 0, 0))
    arrow(d, (785, 235), (870, 235), width=4)
    arrow(d, (1070, 235), (1150, 235), width=4)
    cube(d, 1150, 195, 54, 82, (120, 120, 120), "z_s^t")
    cube(d, 1220, 195, 54, 82, (150, 150, 150), "z_f^t")
    centered_text(d, (1135, 295, 1305, 335), "noisy latents", ft(20))

    d.text((1385, 205), "q(z_T | z_t)", font=ft(28), fill=(0, 0, 0))
    d.text((1530, 218), "...", font=ft(35, True), fill=(0, 0, 0))
    arrow(d, (1295, 235), (1385, 235), width=4)
    arrow(d, (1585, 235), (1665, 235), width=4)
    cube(d, 1665, 195, 54, 82, (172, 220, 145), "z_s^T")
    cube(d, 1735, 195, 54, 82, (154, 205, 130), "z_f^T")
    centered_text(d, (1645, 295, 1845, 335), "diffused latents", ft(20))

    round_box(d, (1915, 150, 2115, 325), "Selected\nMulti-Scale\nProcess Bank\nB_k", fill=(236, 230, 245), radius=14, font=ft(21, True))
    for i, h in enumerate([92, 74, 56]):
        x = 1940 + i * 36
        y = 238 - h
        draw_col = [(255, 230, 65), (255, 210, 40), (235, 185, 20)][i]
        draw = d
        draw.rectangle((x, y, x + 42, y + h), fill=draw_col, outline=(0, 0, 0), width=2)
    d.line((2005, 150, 2005, 92, 710, 92, 710, 195), fill=(0, 0, 0), width=2)
    arrow(d, (2005, 150), (2005, 151), width=2)

    # ---------- Reverse diffusion ----------
    centered_text(d, (0, 1285, W, 1335), "Diffusion Reverse Process & Inference", ft(30, True))
    cube(d, 120, 610, 62, 96, (172, 220, 145), "z_s^T")
    cube(d, 120, 970, 62, 96, (154, 205, 130), "z_f^T")

    net1 = (275, 545, 1025, 780)
    net2 = (275, 905, 1025, 1140)
    round_box(d, net1, "Dual-Stream Denoising Backbone (DSB)", fill=(224, 239, 252), radius=18, font=ft(23, True))
    round_box(d, net2, "Dual-Stream Denoising Backbone (DSB)", fill=(238, 250, 232), radius=18, font=ft(23, True))
    centered_text(d, (300, 575, 1000, 610), "structure stream", ft(17, True))
    centered_text(d, (300, 935, 1000, 970), "function stream", ft(17, True))
    labels = ["Local\nMixer", "Mamba\nStage 2-3", "CSI", "RB", "Noise\nPred."]
    for base_y in [645, 1005]:
        xs = [325, 480, 650, 790, 915]
        for x, lab in zip(xs, labels):
            round_box(d, (x, base_y, x + 95, base_y + 75), lab, fill=(255, 255, 255), radius=10, font=ft(16, True))
        for x0, x1 in zip(xs[:-1], xs[1:]):
            arrow(d, (x0 + 95, base_y + 38), (x1, base_y + 38), width=3)
    arrow(d, (182, 658), (275, 662), width=4)
    arrow(d, (182, 1018), (275, 1022), width=4)
    d.line((700, 720, 700, 1005), fill=(0, 0, 0), width=3)
    arrow(d, (700, 720), (700, 1005), width=3, both=True)
    centered_text(d, (705, 790, 870, 850), "cross-stream\ninteraction", ft(17, True))

    round_box(d, (1085, 680, 1265, 760), "Scheduler\nUpdate", fill=(255, 245, 220), radius=10, font=ft(21, True))
    round_box(d, (1085, 1040, 1265, 1120), "Scheduler\nUpdate", fill=(255, 245, 220), radius=10, font=ft(21, True))
    arrow(d, (1010, 682), (1085, 720), width=3)
    arrow(d, (1010, 1042), (1085, 1080), width=3)
    d.line((1175, 680, 1175, 505, 250, 505, 250, 640), fill=(0, 0, 0), width=2)
    d.line((1175, 1040, 1175, 865, 250, 865, 250, 1000), fill=(0, 0, 0), width=2)
    arrow(d, (250, 640), (275, 640), width=2)
    arrow(d, (250, 1000), (275, 1000), width=2)
    centered_text(d, (1040, 820, 1305, 875), "iterative denoising\nz_t -> z_{t-1}", ft(18, True))

    cube(d, 1330, 625, 54, 82, (200, 225, 242), "z_s^0_hat")
    cube(d, 1330, 985, 54, 82, (218, 242, 208), "z_f^0_hat")
    arrow(d, (1265, 720), (1330, 666), width=4)
    arrow(d, (1265, 1080), (1330, 1026), width=4)

    # Bank injection from selected stages.
    round_box(d, (1115, 820, 1395, 940), "B_k: selected process features\nfrom stages 1, 2, 3", fill=(236, 230, 245), radius=12, font=ft(18, True))
    for x in [520, 700, 840]:
        d.line((x, 720, x, 820), fill=(0, 0, 0), width=2)
        arrow(d, (x, 790), (1115, 850), width=2)
        d.line((x, 1005, x, 940), fill=(0, 0, 0), width=2)
        arrow(d, (x, 970), (1115, 900), width=2)

    # Fusion head.
    head = (1510, 680, 1840, 1045)
    round_box(d, head, "MS-RGSF\nFusion Head", fill=(255, 232, 208), radius=18, font=ft(26, True))
    round_box(d, (1540, 735, 1635, 790), "S-Proj", fill=(224, 239, 252), radius=8, font=ft(17))
    round_box(d, (1540, 935, 1635, 990), "F-Proj", fill=(238, 250, 232), radius=8, font=ft(17))
    round_box(d, (1540, 835, 1675, 895), "MS Bank\nAgg.", fill=(236, 230, 245), radius=8, font=ft(16, True))
    round_box(d, (1715, 790, 1815, 945), "Reliability\nMaps\nR_s R_f R_b\n+\nWeighted\nFusion", fill=(255, 255, 255), radius=10, font=ft(16, True))
    arrow(d, (1395, 666), (1510, 762), width=4)
    arrow(d, (1395, 1026), (1510, 962), width=4)
    arrow(d, (1395, 880), (1540, 865), width=4)
    arrow(d, (1635, 762), (1715, 850), width=3)
    arrow(d, (1635, 962), (1715, 900), width=3)
    arrow(d, (1675, 865), (1715, 875), width=3)

    cube(d, 1900, 830, 58, 86, (160, 200, 230), "z_fuse")
    arrow(d, (1840, 870), (1900, 870), width=4)
    round_box(d, (2020, 810, 2100, 930), "D", fill=(210, 222, 244), radius=0, font=ft(32, True))
    arrow(d, (1965, 870), (2020, 870), width=4)
    round_box(d, (2145, 810, 2260, 930), "SGRR", fill=(236, 232, 246), radius=12, font=ft(22, True))
    arrow(d, (2100, 870), (2145, 870), width=4)
    arrow(d, (2260, 870), (2320, 870), width=4)
    paste_asset(im, "PET_MRI_Output_Fusion.png", (2320, 795, 2480, 955))
    centered_text(d, (2315, 965, 2490, 1015), "fused image", ft(21, True))

    # Legend.
    leg = (55, H - 160, W - 55, H - 35)
    round_box(d, leg, fill=(255, 255, 255), radius=10, width=2)
    x = 90
    y = H - 125
    cube(d, x, y - 20, 28, 48, (172, 220, 145))
    d.text((x + 65, y - 5), "latent features", font=ft(18), fill=(0, 0, 0))
    round_box(d, (x + 260, y - 22, x + 330, y + 32), "DSB", fill=(224, 239, 252), radius=8, font=ft(16, True))
    d.text((x + 350, y - 5), "denoising backbone", font=ft(18), fill=(0, 0, 0))
    round_box(d, (x + 620, y - 22, x + 730, y + 32), "MS-RGSF", fill=(255, 232, 208), radius=8, font=ft(15, True))
    d.text((x + 750, y - 5), "fusion head", font=ft(18), fill=(0, 0, 0))
    round_box(d, (x + 1010, y - 22, x + 1110, y + 32), "B_k", fill=(236, 230, 245), radius=8, font=ft(17, True))
    d.text((x + 1130, y - 5), "selected process bank", font=ft(18), fill=(0, 0, 0))
    arrow(d, (x + 1470, y + 5), (x + 1540, y + 5), width=3)
    d.text((x + 1560, y - 5), "feature flow", font=ft(18), fill=(0, 0, 0))
    d.line((x + 1760, y + 5, x + 1830, y + 5), fill=(0, 0, 0), width=2)
    d.text((x + 1850, y - 5), "skip / bank injection", font=ft(18), fill=(0, 0, 0))
    im.save(OUT / "fig1_overall_framework_clean.png", quality=95)


def draw_head():
    W, H = 2400, 1040
    im = Image.new("RGB", (W, H), (246, 246, 246))
    d = ImageDraw.Draw(im)
    round_box(d, (15, 8, W - 15, H - 15), fill=(246, 246, 246), radius=14, width=3)
    centered_text(d, (0, 25, W, 120), "Multi-Scale Reliability-Guided Structure-Function\nFusion Head (MS-RGSF)", ft(40, True))
    y1, y2, y3 = 335, 500, 665
    for y, lab, col in [(y1, "z_s^0_hat\n(Structure)", (184, 215, 238)),
                        (y2, "z_f^0_hat\n(Function)", (218, 242, 208))]:
        centered_text(d, (40, y - 55, 190, y + 55), lab, ft(24, True))
        cube(d, 255, y - 42, 55, 84, col)
        arrow(d, (190, y), (255, y), width=3)
    cylinder(d, 245, y3 - 52, 82, 104, fill=(225, 213, 240), text="B_k\nStages 1/2/3")

    round_box(d, (430, y1 - 58, 585, y1 + 58), "S-Proj\n1x1 Conv", fill=(220, 238, 252), radius=10, font=ft(22))
    round_box(d, (430, y2 - 58, 585, y2 + 58), "F-Proj\n1x1 Conv", fill=(230, 250, 225), radius=10, font=ft(22))
    round_box(d, (430, y3 - 58, 610, y3 + 58), "Multi-Scale\nBank Agg.\nB_ms", fill=(236, 230, 245), radius=10, font=ft(20, True))
    arrow(d, (310, y1), (430, y1), width=3)
    arrow(d, (310, y2), (430, y2), width=3)
    arrow(d, (327, y3), (430, y3), width=3)

    diff = (745, 200, 1010, 655)
    rel = (1060, 200, 1320, 760)
    fuse = (1510, 235, 1860, 720)
    refine = (1935, 330, 2140, 705)
    round_box(d, diff, "Difference /\nConsistency Modeling", fill=(246, 246, 246), outline=(74, 101, 105), radius=10, font=ft(20, True), dash=True)
    round_box(d, rel, "Reliability\nEstimator", fill=(246, 246, 246), outline=(74, 101, 105), radius=10, font=ft(20, True), dash=True)
    round_box(d, fuse, "Adaptive Weighted Fusion", fill=(246, 246, 246), outline=(74, 101, 105), radius=10, font=ft(20, True), dash=True)
    round_box(d, refine, "Lightweight Residual\nRefinement", fill=(246, 246, 246), outline=(74, 101, 105), radius=10, font=ft(18, True), dash=True)

    d.text((640, y1 - 10), "S", font=ft(22, True), fill=(0, 0, 0))
    d.text((640, y2 - 10), "F", font=ft(22, True), fill=(0, 0, 0))
    d.text((640, y3 - 10), "B_ms", font=ft(22, True), fill=(0, 0, 0))
    arrow(d, (585, y1), (790, y1), width=3)
    arrow(d, (585, y2), (790, y2), width=3)
    arrow(d, (610, y3), (1070, y3), width=3)
    round_box(d, (795, 282, 965, 370), "Difference\n|S - F|", fill=(255, 255, 255), radius=10, font=ft(22, True))
    round_box(d, (795, 452, 965, 540), "Consistency\nS x F", fill=(255, 255, 255), radius=10, font=ft(22, True))
    arrow(d, (650, y1), (795, 326), width=3)
    arrow(d, (650, y2), (795, 496), width=3)

    round_box(d, (1110, 270, 1275, 360), "Concat\n[S, F, B_ms,\n|S-F|, S x F]", fill=(248, 248, 252), radius=10, font=ft(18))
    round_box(d, (1120, 405, 1265, 455), "Conv 1x1", fill=(248, 248, 252), radius=8, font=ft(20))
    round_box(d, (1120, 500, 1265, 550), "GELU", fill=(248, 248, 252), radius=8, font=ft(20))
    round_box(d, (1120, 595, 1265, 645), "Conv 1x1", fill=(248, 248, 252), radius=8, font=ft(20))
    round_box(d, (1120, 690, 1265, 740), "Softmax", fill=(248, 248, 252), radius=8, font=ft(20))
    for a, b in [((965, 326), (1110, 315)), ((965, 496), (1110, 315)), ((1070, y3), (1110, 315))]:
        arrow(d, a, b, width=3)
    for y in [360, 455, 550, 645]:
        arrow(d, (1192, y), (1192, y + 45), width=3)

    # Reliability maps to fusion.
    for y, lab in [(320, "R_s\nStructure\nreliability"), (500, "R_f\nFunction\nreliability"), (680, "R_b\nBank\nreliability")]:
        arrow(d, (1320, y), (1510, y), width=3)
        centered_text(d, (1360, y - 42, 1500, y + 42), lab, ft(17))
    for y, lab, col in [(320, "S", (220, 238, 252)), (500, "F", (230, 250, 225)), (680, "B_ms", (236, 230, 245))]:
        round_box(d, (1565, y - 34, 1638, y + 34), lab, fill=col, radius=8, font=ft(25, True))
        op_circle(d, (1710, y), "x", r=26)
        arrow(d, (1638, y), (1684, y), width=3)
    op_circle(d, (1810, 500), "+", r=28)
    arrow(d, (1736, 320), (1810, 475), width=3)
    arrow(d, (1736, 500), (1782, 500), width=3)
    arrow(d, (1736, 680), (1810, 525), width=3)
    d.text((1850, 490), "z0", font=ft(25, True), fill=(0, 0, 0))
    arrow(d, (1838, 500), (1935, 500), width=3)

    for i, txt in enumerate(["Conv 1x1", "DWConv 3x3", "GELU", "Conv 1x1"]):
        round_box(d, (1975, 370 + i * 82, 2100, 420 + i * 82), txt, fill=(255, 255, 255), radius=8, font=ft(18))
        if i < 3:
            arrow(d, (2038, 420 + i * 82), (2038, 452 + i * 82), width=3)
    op_circle(d, (2190, 500), "+", r=27)
    arrow(d, (2140, 500), (2163, 500), width=3)
    d.line((1850, 500, 1850, 765, 2190, 765, 2190, 527), fill=(0, 0, 0), width=3)
    arrow(d, (2190, 500), (2290, 500), width=3)
    cube(d, 2290, 460, 55, 84, (244, 188, 157))
    centered_text(d, (2265, 560, 2385, 625), "Z_fuse\n(Fused latent)", ft(22))
    legend(d, (1450, 820, 2220, 985), items=("x", "+"))
    im.save(OUT / "fig2_ms_rgsf_head.png", quality=95)


def draw_sgrr():
    W, H = 1600, 860
    im = Image.new("RGB", (W, H), (246, 246, 246))
    d = ImageDraw.Draw(im)
    round_box(d, (15, 15, W - 15, H - 15), fill=(246, 246, 246), radius=14, width=3)
    centered_text(d, (0, 45, W, 95), "Structure-Guided Reliability Refinement (SGRR)", ft(32, True))
    round_box(d, (70, 165, 205, 285), "I_s\nStructural\nguidance", fill=(224, 239, 252), radius=10, font=ft(18))
    round_box(d, (70, 455, 205, 575), "I_d\nDecoded\nfused image", fill=(234, 248, 228), radius=10, font=ft(18))
    round_box(d, (315, 110, 885, 335), "Structure-guidance branch", fill=(224, 239, 252), radius=12, font=ft(27, True))
    round_box(d, (315, 390, 885, 615), "Residual-refinement branch", fill=(238, 250, 232), radius=12, font=ft(27, True))
    arrow(d, (205, 225), (315, 225), width=3)
    arrow(d, (205, 515), (315, 515), width=3)
    for x, txt in [(360, "1x1\nConv"), (545, "3x3\nConv"), (730, "Sigmoid")]:
        round_box(d, (x, 185, x + 120, 270), txt, fill=(255, 255, 255), radius=10, font=ft(23))
    for x in [480, 665]:
        arrow(d, (x, 225), (x + 65, 225), width=3)
    round_box(d, (940, 165, 1080, 285), "M_s\nstructure\nreliability map", fill=(224, 239, 252), radius=10, font=ft(17))
    arrow(d, (850, 225), (940, 225), width=3)
    for x, txt in [(340, "3x3\nConv"), (475, "Norm"), (610, "GELU"), (745, "3x3\nConv")]:
        round_box(d, (x, 475, x + 100, 555), txt, fill=(255, 255, 255), radius=10, font=ft(21))
    for x in [440, 575, 710]:
        arrow(d, (x, 515), (x + 35, 515), width=3)
    round_box(d, (945, 455, 1085, 575), "R_d\nresidual\nfeature", fill=(238, 250, 232), radius=10, font=ft(18))
    arrow(d, (845, 515), (945, 515), width=3)
    op_circle(d, (1165, 405), "x", r=24)
    op_circle(d, (1230, 535), "+", r=25)
    arrow(d, (1080, 225), (1165, 381), width=3)
    arrow(d, (1085, 515), (1165, 429), width=3)
    arrow(d, (1165, 429), (1230, 510), width=3)
    d.line((205, 575, 205, 690, 1230, 690, 1230, 560), fill=(0, 0, 0), width=3)
    arrow(d, (1230, 535), (1305, 535), width=3)
    round_box(d, (1305, 495, 1395, 575), "3x3\nConv", fill=(255, 244, 220), radius=10, font=ft(22))
    arrow(d, (1395, 535), (1450, 535), width=3)
    round_box(d, (1450, 480, 1545, 590), "I_f\nRefined\nfused\nimage", fill=(255, 244, 220), radius=10, font=ft(20))
    legend(d, (1050, 720, 1560, 835), items=("x", "+"))
    im.save(OUT / "fig3_sgrr.png", quality=95)


def draw_dsb():
    W, H = 1900, 1080
    im = Image.new("RGB", (W, H), (246, 246, 246))
    d = ImageDraw.Draw(im)
    round_box(d, (15, 20, W - 15, 700), fill=(246, 246, 246), radius=14, width=3)
    centered_text(d, (0, 55, W, 95), "Dual-Stream Denoising Backbone (DSB)", ft(30, True))
    round_box(d, (445, 125, 1305, 305), fill=(224, 239, 252), radius=10, width=3)
    round_box(d, (445, 495, 1305, 675), fill=(238, 250, 232), radius=10, width=3)
    centered_text(d, (445, 145, 1305, 175), "Structure stream", ft(17, True))
    centered_text(d, (445, 515, 1305, 545), "Function stream", ft(17, True))
    cube(d, 230, 190, 56, 70, (224, 239, 252), "z_s^t")
    cube(d, 230, 555, 56, 70, (238, 250, 232), "z_f^t")
    arrow(d, (300, 225), (445, 225), width=3)
    arrow(d, (300, 590), (445, 590), width=3)
    for y in [225, 590]:
        for x, txt in [(500, "LCB\nLocal"), (710, "SSB\nStage 2-3"), (925, "CSI"), (1135, "RB")]:
            round_box(d, (x, y - 42, x + 120, y + 42), txt, fill=(255, 255, 255), radius=10, font=ft(24))
        for a, b in [(620, 710), (830, 925), (1045, 1135)]:
            arrow(d, (a, y), (b, y), width=3)
    arrow(d, (985, 267), (985, 548), width=3, both=True)
    cube(d, 1400, 190, 56, 70, (224, 239, 252), "z_s^0_hat")
    cube(d, 1400, 555, 56, 70, (238, 250, 232), "z_f^0_hat")
    arrow(d, (1305, 225), (1400, 225), width=3)
    arrow(d, (1305, 590), (1400, 590), width=3)
    round_box(d, (530, 365, 755, 455), "Timestep\nConditioning t", fill=(255, 245, 220), radius=10, font=ft(22, True))
    d.line((642, 365, 642, 305), fill=(0, 0, 0), width=3)
    d.line((642, 455, 642, 495), fill=(0, 0, 0), width=3)
    round_box(d, (785, 355, 985, 465), "Selective Multi-Scale\nProcess Bank B_k\nStages 1, 2, 3", fill=(236, 230, 245), radius=10, font=ft(18, True))
    arrow(d, (875, 305), (875, 355), width=3)
    arrow(d, (875, 495), (875, 465), width=3)
    d.text((1000, 385), "Cross-stream\ninteraction", font=ft(19, True), fill=(0, 0, 0))
    round_box(d, (1565, 350, 1815, 650), "", fill=(255, 255, 255), radius=8, width=3)
    legend(d, (1565, 350, 1815, 650), items=("x", "+"))

    panels = [
        ((25, 730, 460, 1050), "(a) Local Conv Block (LCB)", ["1x1\nConv", "Norm", "GELU", "3x3\nConv", "+"]),
        ((485, 730, 890, 1050), "(b) State-Space Block (SSB)", ["1x1\nConv", "Flatten\n/ Scan", "Mamba\nor Local", "Reshape", "1x1\nConv"]),
        ((915, 730, 1375, 1050), "(c) Cross-Stream Interaction (CSI)", ["F_s", "Avg/Diff\nContext", "Shared\nGate", "g_s/g_f", "F_s' / F_f'"]),
        ((1400, 730, 1875, 1050), "(d) Refinement Block (RB)", ["1x1\nConv", "3x3\nConv", "Norm", "GELU", "+"]),
    ]
    for box, title, labels in panels:
        round_box(d, box, fill=(232, 244, 255) if "(a)" in title or "(b)" in title else (238, 250, 232), radius=8, width=3)
        centered_text(d, (box[0], box[1] + 8, box[2], box[1] + 38), title, ft(16, True))
    # Panel internals.
    for i, lab in enumerate(["1x1\nConv", "Norm", "GELU", "3x3\nConv"]):
        x = 75 + i * 80
        round_box(d, (x, 820, x + 55, 900), lab, fill=(255, 255, 255), radius=8, font=ft(14))
        if i < 3:
            arrow(d, (x + 55, 860), (x + 80, 860), width=2)
    op_circle(d, (410, 860), "+", r=18, font=ft(16, True))
    d.line((35, 860, 35, 955, 410, 955, 410, 878), fill=(0, 0, 0), width=2)

    for i, lab in enumerate(["1x1\nConv", "Flatten\n/ Scan", "Mamba\n(Stage 2-3)\nLocal otherwise", "Reshape", "1x1\nConv"]):
        x = 500 + i * 75
        round_box(d, (x, 825, x + 62, 900), lab, fill=(255, 255, 255), radius=8, font=ft(12))
        if i < 4:
            arrow(d, (x + 62, 862), (x + 75, 862), width=2)

    round_box(d, (950, 815, 1015, 890), "F_s", fill=(224, 239, 252), radius=8, font=ft(14, True))
    round_box(d, (950, 925, 1015, 1000), "F_f", fill=(238, 250, 232), radius=8, font=ft(14, True))
    round_box(d, (1080, 790, 1175, 865), "Avg/Diff\nContext", fill=(255, 245, 220), radius=8, font=ft(14, True))
    round_box(d, (1080, 905, 1175, 980), "Shared\nGate", fill=(255, 210, 170), radius=8, font=ft(14, True))
    op_circle(d, (1245, 850), "x", r=16, font=ft(14, True))
    op_circle(d, (1245, 960), "x", r=16, font=ft(14, True))
    op_circle(d, (1300, 850), "+", r=16, font=ft(14, True))
    op_circle(d, (1300, 960), "+", r=16, font=ft(14, True))
    round_box(d, (1330, 820, 1365, 880), "F_s'", fill=(224, 239, 252), radius=6, font=ft(12, True))
    round_box(d, (1330, 930, 1365, 990), "F_f'", fill=(238, 250, 232), radius=6, font=ft(12, True))

    for i, lab in enumerate(["1x1\nConv", "3x3\nConv", "Norm", "GELU"]):
        x = 1480 + i * 88
        round_box(d, (x, 830, x + 62, 900), lab, fill=(255, 255, 255), radius=8, font=ft(14))
        if i < 3:
            arrow(d, (x + 62, 865), (x + 88, 865), width=2)
    op_circle(d, (1840, 865), "+", r=18, font=ft(16, True))
    d.line((1420, 865, 1420, 960, 1840, 960, 1840, 883), fill=(0, 0, 0), width=2)
    im.save(OUT / "fig4_dsb_stage_selective_bank.png", quality=95)


if __name__ == "__main__":
    draw_overall()
    draw_overall_process()
    draw_overall_clean_paper()
    draw_head()
    draw_sgrr()
    draw_dsb()
    print(f"Saved figures to {OUT}")
