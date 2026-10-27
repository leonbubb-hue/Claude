"""SCP-096 First-Person Analog-Horror TikTok - prozeduraler Video-Renderer.

Rendert 65 s @ 30 fps in 1080x1920 (9:16). Die 3D-Szenen (Korridor, Raum)
werden per Raycasting in NumPy berechnet (540x960), danach hochskaliert,
mit Bodycam-HUD, Glitch-Text, VHS-Effekten und Filmkorn versehen.

    python3 render.py            -> video_noaudio.mp4
    python3 render.py --still 14.2 15.8 ...  -> Einzelbilder zur Kontrolle
"""
import math
import os
import subprocess
import sys
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FPS = 30
DUR = 65.0
W, H = 540, 960          # Renderaufloesung der 3D-Szenen
OW, OH = 1080, 1920      # Ausgabe
VFOV = math.radians(78)
F = (H / 2) / math.tan(VFOV / 2)
FONT_BOLD = "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-ExtraBold.ttf"
FONT_MED = "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Medium.ttf"
FONT_EMOJI = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
FONT_SERIF_I = "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf"


# ============================================================ utilities
def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def sm(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def lerp(a, b, k):
    return a + (b - a) * k


def hash1(k):
    return (np.sin(np.asarray(k, dtype=np.float64) * 12.9898 + 78.233) * 43758.5453) % 1.0


def wobble(t, seed, freqs=(0.7, 1.9, 4.3, 9.1), amps=(1, 0.6, 0.35, 0.2)):
    """Glatte Pseudo-Zufallsbewegung (Kamerawackeln)."""
    v = 0.0
    for i, (f, a) in enumerate(zip(freqs, amps)):
        v += a * math.sin(2 * math.pi * f * t + 10 * hash1(seed * 7 + i))
    return v / sum(amps)


def fractal_tex(n, beta, seed):
    r = np.random.default_rng(seed)
    fx = np.fft.fftfreq(n)[:, None]
    fy = np.fft.fftfreq(n)[None, :]
    k = np.sqrt(fx ** 2 + fy ** 2)
    k[0, 0] = 1
    spec = np.fft.fft2(r.standard_normal((n, n))) / k ** beta
    spec[0, 0] = 0
    img = np.real(np.fft.ifft2(spec))
    img -= img.min()
    return (img / img.max()).astype(np.float32)


CONC = fractal_tex(512, 1.15, 1)
FINE = fractal_tex(512, 0.55, 2)
STAIN = fractal_tex(256, 1.7, 3)


def samp(tex, u, v):
    n = tex.shape[0]
    i = np.floor(u * n).astype(np.int64) % n
    j = np.floor(v * n).astype(np.int64) % n
    return tex[j, i]


# ============================================================ camera rays
_ys, _xs = np.mgrid[0:H, 0:W].astype(np.float32)
_cx = (_xs - W / 2 + 0.5) / F
_cy = -(_ys - H / 2 + 0.5) / F
_r2 = _cx ** 2 + _cy ** 2
_k = 1 + 0.10 * _r2          # leichte Weitwinkel-/Bodycam-Verzerrung
BASE = np.stack([_cx * _k, _cy * _k, np.ones_like(_cx)], -1)
BASE /= np.linalg.norm(BASE, axis=-1, keepdims=True)
VIGN = np.clip(1.25 - 0.55 * (((_xs - W / 2) / (W / 2)) ** 2 + ((_ys - H / 2) / (H / 2)) ** 2 * 0.6), 0, 1) ** 1.6


def rot(yaw, pitch, roll):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def project(cam, p):
    """Weltpunkt -> (sx, sy, tiefe) in Renderpixeln, oder None."""
    Rm = rot(cam["yaw"], cam["pitch"], cam["roll"])
    v = Rm.T @ (np.asarray(p, float) - np.asarray(cam["pos"], float))
    if v[2] <= 0.05:
        return None
    x, y = v[0] / v[2], v[1] / v[2]
    k = 1 + 0.10 * (x * x + y * y)
    return W / 2 + x * F / k, H / 2 - y * F / k, v[2]


# ============================================================ flicker
def flicker(k, t, chaos=0.0):
    k = np.asarray(k, dtype=np.float64)
    base = 0.8 + 0.2 * hash1(k + 3.1)
    bad = hash1(k) < 0.35 + chaos
    rate = 14 + 30 * chaos
    step = np.floor(t * rate + hash1(k + 1.7) * 100)
    on = hash1(step * 1.31 + k * 7.7) > (0.28 + 0.35 * chaos)
    slow = 0.5 + 0.5 * np.sin(t * (0.9 + hash1(k + 9)) * 6.28)
    pat = np.where(on, 0.75 + 0.25 * slow, 0.04)
    return base * np.where(bad, pat, 1.0)


# ============================================================ raycaster
NRM = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], np.float32)


def render_box(cam, sc, t):
    """Rendert einen quaderfoermigen Raum (Korridor oder Zelle)."""
    Rm = rot(cam["yaw"], cam["pitch"], cam["roll"]).astype(np.float32)
    D = BASE @ Rm.T
    dx, dy, dz = D[..., 0], D[..., 1], D[..., 2]
    cx, cy, cz = cam["pos"]
    x0, x1, y0, y1, z0, z1 = sc["bounds"]
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(dx > 0, (x1 - cx) / dx, (x0 - cx) / dx)
        ty = np.where(dy > 0, (y1 - cy) / dy, (y0 - cy) / dy)
        tz = np.where(dz > 0, (z1 - cz) / dz, (z0 - cz) / dz)
    tx = np.where(np.isfinite(tx) & (tx > 0), tx, 1e9)
    ty = np.where(np.isfinite(ty) & (ty > 0), ty, 1e9)
    tz = np.where(np.isfinite(tz) & (tz > 0), tz, 1e9)
    tt = np.minimum(np.minimum(tx, ty), tz)
    face = np.where(tt == tx, np.where(dx > 0, 1, 0), np.where(tt == ty, np.where(dy > 0, 3, 2), np.where(dz > 0, 5, 4)))
    px, py, pz = cx + dx * tt, cy + dy * tt, cz + dz * tt

    isx = face <= 1
    isy = (face == 2) | (face == 3)
    u = np.where(isx, pz, px)
    v = np.where(isy, pz, py)

    c = samp(CONC, u * 0.23, v * 0.23)
    f = samp(FINE, u * 0.9 + 0.3, v * 0.9)
    tex = 0.55 + 0.35 * c + 0.25 * (f - 0.5)
    alb = np.empty(tt.shape + (3,), np.float32)
    tint = np.array(sc["tints"], np.float32)  # 6x3
    alb[:] = tint[face]
    alb *= tex[..., None]

    em = np.zeros_like(alb)
    # --- Waende: gestrichener Sockel, Streifen, Fugen
    wall = isx | (face >= 4)
    lowpaint = wall & (py < 1.15)
    alb[lowpaint] *= np.array(sc.get("paint", (0.62, 0.72, 0.66)), np.float32)
    stripe = wall & (np.abs(py - 1.15) < 0.035)
    alb[stripe] *= 0.35
    seam = isx & (np.abs(((pz + 1.5) / 3.0) % 1.0 - 0.5) > 0.495)
    alb[seam] *= 0.4
    # Wasserflecken / Schmutz
    st = samp(STAIN, u * 0.11, v * 0.11 + 0.3)
    dirt = np.clip((st - 0.55) * 4, 0, 1) * np.where(isy, 0.55, 0.35)
    alb *= (1 - dirt)[..., None]
    floor = face == 2
    alb[floor] *= 0.55

    # --- Deckenleuchten (Korridor)
    if sc.get("ceiling_lights"):
        ceil = face == 3
        kk = np.round(pz / 6.0)
        fixture = ceil & (np.abs(px) < 0.22) & (np.abs(pz - kk * 6) < 0.75)
        if fixture.any():
            fl = flicker(kk[fixture], t, sc.get("chaos", 0.0)) * sc.get("light_mul", 1.0)
            em[fixture] = np.array([0.85, 1.0, 0.92], np.float32) * (2.2 * fl)[:, None]

    # --- Stahltuer
    door = sc.get("door")
    if door is not None:
        dface, dzpos = door["face"], door["z"]
        dm = (face == dface) & (np.abs(px) < 0.6) & (py < 2.1)
        if door.get("open_frac", 0) > 0:  # Tuer schwingt auf / ist offen
            of = door["open_frac"]
            opening = dm & (px > 0.6 - 1.2 * of) if dface == 5 else dm & (px < -0.6 + 1.2 * of)
            em[opening] = np.array(door.get("beyond", (0.0, 0.0, 0.0)), np.float32)
            alb[opening] = 0
            dm = dm & ~opening
        if door.get("blown"):
            n = samp(FINE, px * 0.7, py * 0.7)
            hole = (face == dface) & (np.abs(px) < 0.72 + 0.25 * n) & (py < 2.25 + 0.3 * n)
            bl = door["beyond_light"]
            haze = 0.6 + 0.4 * samp(CONC, px * 0.3 + t * 0.05, py * 0.3)
            em[hole] = np.array([0.55, 0.6, 0.62], np.float32) * (bl * haze[hole])[:, None]
            alb[hole] = 0
            dm = np.zeros_like(dm)
        if dm.any():
            dxl, dyl = px[dm], py[dm]
            steel = 0.42 + 0.25 * samp(FINE, dxl * 2.5, dyl * 0.4) + 0.12 * samp(CONC, dxl, dyl)
            col = np.stack([steel * 0.78, steel * 0.84, steel * 0.92], -1)
            # Rahmen, Nieten, Querstreben
            edge = (np.abs(dxl) > 0.53) | (dyl > 2.03)
            col[edge] *= 0.55
            rivx = (np.abs(((dxl + 0.6) / 0.2) % 1 - 0.5) < 0.08)
            rivy = (np.abs(((dyl) / 0.35) % 1 - 0.5) < 0.05)
            col[rivx & rivy] *= 1.6
            band = (np.abs(dyl - 0.6) < 0.05) | (np.abs(dyl - 1.6) < 0.05)
            col[band] *= 0.7
            handle = (np.abs(dxl - 0.42 * door.get("hside", 1)) < 0.03) & (np.abs(dyl - 1.05) < 0.14)
            col[handle] = np.array([0.15, 0.15, 0.15]) + 0.6 * col[handle]
            # blutige Handabdruecke
            bm = samp(STAIN, dxl * 0.9 + 0.31, dyl * 0.55 + 0.12)
            blood = np.clip((bm - 0.62) * 6, 0, 1) * ((np.abs(dxl + 0.25) < 0.3) & (dyl > 0.8) & (dyl < 1.7))
            col = col * (1 - blood[:, None]) + blood[:, None] * np.array([0.25, 0.02, 0.02])
            # Riegel
            b = door.get("bolt")
            if b is not None:
                bx0, bx1 = 0.15, 0.15 + 0.62 * b
                bar = (dyl > 1.22) & (dyl < 1.29) & (dxl > bx0) & (dxl < min(bx1, 0.6))
                col[bar] = 0.85 * np.array([0.55, 0.55, 0.52])
            alb[dm] = col
        if sc.get("exit_sign") and dface == 5:
            sign = (face == 5) & (np.abs(px) < 0.25) & (np.abs(py - 2.33) < 0.08)
            em[sign] = np.array([1.6, 0.05, 0.03], np.float32) * (0.7 + 0.3 * math.sin(t * 9))
        if door.get("gap") is not None and not door.get("blown"):
            g = door["gap"]
            gmod = door["gap_mod"]
            gapm = (face == dface) & (np.abs(px) < 0.56) & (py < 0.045)
            em[gapm] = np.array([0.8, 0.95, 1.0], np.float32) * (3.0 * g * gmod(px[gapm]))[:, None]
            spill = (face == 2) & (np.abs(px) < 0.8) & (pz < 1.0)
            k = g * 1.1 * np.exp(-pz[spill] / 0.3) * smooth((0.72 - np.abs(px[spill])) / 0.2) * gmod(px[spill])
            em[spill] += np.array([0.7, 0.85, 0.95], np.float32) * k[:, None]

    # --- Beleuchtung
    light = np.full(alb.shape, sc.get("ambient", 0.01), np.float32)
    nrm = NRM[face]
    for (lp, lc, li, fall) in sc["lights"]:
        if li <= 0.002:
            continue
        Lx, Ly, Lz = lp[0] - px, lp[1] - py, lp[2] - pz
        d2 = Lx * Lx + Ly * Ly + Lz * Lz
        inv = 1 / np.sqrt(d2 + 1e-6)
        nd = np.maximum((nrm[..., 0] * Lx + nrm[..., 1] * Ly + nrm[..., 2] * Lz) * inv, 0)
        k = li * (0.25 + 0.75 * nd) / (1 + d2 * fall)
        light += k[..., None] * np.asarray(lc, np.float32)
    if sc.get("flashlight"):
        fl = sc["flashlight"]
        cosang = (D @ np.asarray(fl["dir"], np.float32))
        spot = smooth((cosang - 0.90) / 0.08) * fl["power"] / (1 + tt * tt * 0.15)
        light += spot[..., None] * np.array([1.0, 0.95, 0.85], np.float32)

    img = alb * light + em
    fog = np.exp(-tt * sc.get("fog", 0.06))[..., None]
    fogc = np.asarray(sc.get("fogcol", (0, 0, 0)), np.float32)
    img = img * fog + fogc * (1 - fog)
    return img, tt


# ============================================================ sprites
SKIN = (214, 208, 198, 255)


def _poly(d, pts, fill):
    d.polygon([tuple(map(float, p)) for p in pts], fill=fill)


def _limb(d, a, b, w0, w1, fill):
    ax, ay = a
    bx, by = b
    ang = math.atan2(by - ay, bx - ax) + math.pi / 2
    ca, sa = math.cos(ang), math.sin(ang)
    _poly(d, [(ax + ca * w0, ay + sa * w0), (bx + ca * w1, by + sa * w1),
              (bx - ca * w1, by - sa * w1), (ax - ca * w0, ay - sa * w0)], fill)
    d.ellipse([bx - w1, by - w1, bx + w1, by + w1], fill=fill)
    d.ellipse([ax - w0, ay - w0, ax + w0, ay + w0], fill=fill)


@lru_cache(None)
def sprite_kneel(reveal):
    """Kniende, ausgemergelte Gestalt (Frontansicht). reveal=Haende vom Gesicht."""
    w, h = 600, 700
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    s = SKIN
    sh = (150, 145, 138, 255)
    # Unterschenkel / Knie am Boden
    _limb(d, (230, 690), (180, 560), 26, 30, sh)
    _limb(d, (370, 690), (420, 560), 26, 30, sh)
    _limb(d, (180, 560), (255, 420), 30, 34, s)
    _limb(d, (420, 560), (345, 420), 30, 34, s)
    # Becken + Rumpf (nach vorne gebeugt, ausgemergelt)
    _poly(d, [(245, 430), (355, 430), (372, 300), (395, 170), (205, 170), (228, 300)], s)
    for i in range(6):  # Rippen
        y = 205 + i * 17
        d.line([(222 + i * 2, y), (300, y + 10), (378 - i * 2, y)], fill=(160, 152, 145, 255), width=4)
    d.line([(300, 180), (300, 420)], fill=(170, 162, 155, 255), width=5)
    if not reveal:
        head = (300, 120)
        d.ellipse([head[0] - 52, head[1] - 75, head[0] + 52, head[1] + 70], fill=s)
        # Arme hoch zum Gesicht, Haende davor
        _limb(d, (212, 180), (150, 300), 17, 14, s)
        _limb(d, (150, 300), (255, 120), 14, 12, s)
        _limb(d, (388, 180), (450, 300), 17, 14, s)
        _limb(d, (450, 300), (345, 120), 14, 12, s)
        for i in range(5):
            _limb(d, (255, 125), (268 + i * 7, 45 + i * 8), 8, 5, s)
            _limb(d, (345, 125), (332 - i * 7, 45 + i * 8), 8, 5, s)
    else:
        head = (300, 105)
        d.ellipse([head[0] - 55, head[1] - 85, head[0] + 55, head[1] + 75], fill=s)
        d.ellipse([268, 70, 284, 86], fill=(20, 18, 18, 255))
        d.ellipse([316, 70, 332, 86], fill=(20, 18, 18, 255))
        d.ellipse([272, 100, 328, 205], fill=(25, 8, 8, 255))  # aufgerissener Mund
        # Arme weggerissen, lange Finger
        _limb(d, (212, 180), (95, 260), 17, 14, s)
        _limb(d, (95, 260), (40, 390), 14, 11, s)
        _limb(d, (388, 180), (505, 260), 17, 14, s)
        _limb(d, (505, 260), (560, 390), 14, 11, s)
        for i in range(4):
            _limb(d, (40, 390), (20 + i * 12, 470 + i * 4), 6, 4, s)
            _limb(d, (560, 390), (580 - i * 12, 470 + i * 4), 6, 4, s)
    im = im.filter(ImageFilter.GaussianBlur(2.2))
    return np.asarray(im).astype(np.float32) / 255


@lru_cache(None)
def sprite_stand():
    """Grosse, gebueckte Silhouette (Gegenlicht)."""
    w, h = 500, 1000
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = (14, 13, 13, 255)
    d.ellipse([205, 40, 300, 175], fill=c)
    _poly(d, [(215, 170), (300, 175), (330, 260), (310, 520), (195, 520), (170, 260)], c)
    _limb(d, (175, 250), (110, 520), 18, 13, c)
    _limb(d, (110, 520), (95, 820), 13, 9, c)
    _limb(d, (330, 250), (385, 520), 18, 13, c)
    _limb(d, (385, 520), (405, 830), 13, 9, c)
    for i in range(4):
        _limb(d, (95, 820), (70 + i * 14, 900), 5, 3, c)
        _limb(d, (405, 830), (420 + i * 10, 905), 5, 3, c)
    _limb(d, (205, 510), (190, 760), 22, 15, c)
    _limb(d, (190, 760), (200, 990), 15, 12, c)
    _limb(d, (300, 510), (318, 760), 22, 15, c)
    _limb(d, (318, 760), (305, 990), 15, 12, c)
    im = im.filter(ImageFilter.GaussianBlur(1.5))
    return np.asarray(im).astype(np.float32) / 255


def composite(img, spr, cx, bottom, height_px, light=1.0, alpha=1.0, rim=None):
    """Sprite (RGBA float) in img einsetzen; cx/bottom/hoehe in Renderpixeln."""
    if height_px < 2:
        return
    sh, sw = spr.shape[:2]
    scale = height_px / sh
    tw, th = max(1, int(sw * scale)), max(1, int(height_px))
    x0 = int(cx - tw / 2)
    y0 = int(bottom - th)
    pil = Image.fromarray((spr * 255).astype(np.uint8), "RGBA").resize((tw, th), Image.BILINEAR)
    s = np.asarray(pil).astype(np.float32) / 255
    ix0, iy0 = max(x0, 0), max(y0, 0)
    ix1, iy1 = min(x0 + tw, img.shape[1]), min(y0 + th, img.shape[0])
    if ix1 <= ix0 or iy1 <= iy0:
        return
    s = s[iy0 - y0: iy1 - y0, ix0 - x0: ix1 - x0]
    a = s[..., 3:4] * alpha
    col = s[..., :3] * light
    if rim is not None:
        col = col + rim
    reg = img[iy0:iy1, ix0:ix1]
    img[iy0:iy1, ix0:ix1] = reg * (1 - a) + col * a


@lru_cache(None)
def hand_layer():
    """Riesige, ueberlange Hand mit Klauen, greift in die Kamera (1080x1920 RGBA)."""
    im = Image.new("RGBA", (OW, OH), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    s = (205, 198, 188, 255)
    sd = (150, 140, 132, 255)
    palm = (540, 1500)
    d.ellipse([300, 1150, 800, 1900], fill=s)
    _limb(d, (540, 2100), palm, 260, 240, s)
    tips = [(-520, 300), (-170, -60), (130, -150), (420, -10), (700, 420)]
    for i, (tx, ty) in enumerate(tips):
        base = (palm[0] + (i - 2) * 95, 1250 if i else 1450)
        mid = ((base[0] + tx + 540) / 2 + (i - 2) * 40, (base[1] + ty) / 2 + 60)
        _limb(d, base, mid, 52, 40, s)
        _limb(d, mid, (tx + 540, ty + 0), 40, 30, s)
        d.line([mid[0] - 25, mid[1], mid[0] + 25, mid[1] + 8], fill=sd, width=6)
        # Klaue
        cx, cy = tx + 540, ty
        _poly(d, [(cx - 26, cy + 10), (cx + 26, cy + 10), (cx + 4, cy - 90)], (35, 20, 18, 255))
        d.ellipse([cx - 40, cy - 10, cx + 40, cy + 60], fill=(120, 25, 20, 230))
    for i in range(14):  # Sehnen / Falten
        y = 1300 + i * 40
        d.line([(380, y), (700, y + 20)], fill=(185, 178, 168, 255), width=3)
    im = im.filter(ImageFilter.GaussianBlur(9))
    a = np.asarray(im).astype(np.float32) / 255
    # Volumen: Rand dunkler (Alpha-Erosion als Schattierung), Hautstruktur, Rotlicht
    alpha = a[..., 3]
    shade = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(40))).astype(np.float32) / 255
    tex = np.asarray(Image.fromarray((CONC * 255).astype(np.uint8)).resize((OW, OH))).astype(np.float32) / 255
    a[..., :3] *= (0.25 + 0.75 * shade ** 1.5)[..., None] * (0.75 + 0.35 * tex)[..., None]
    yy = np.linspace(0, 1, OH)[:, None, None]
    a[..., :3] *= np.array([1.0, 0.72, 0.68]) * (0.55 + 0.45 * yy)
    return a


@lru_cache(None)
def face_base():
    """Das Gesicht - wird beim Zeigen immer stark zerstoert/verzerrt."""
    im = Image.new("RGB", (OW, OH), (4, 4, 4))
    d = ImageDraw.Draw(im)
    skin = (200, 194, 186)
    d.ellipse([90, -350, 990, 2150], fill=skin)
    d.ellipse([260, 330, 450, 470], fill=(30, 28, 28))
    d.ellipse([630, 330, 820, 470], fill=(30, 28, 28))
    d.ellipse([345, 385, 365, 405], fill=(235, 235, 235))
    d.ellipse([715, 385, 735, 405], fill=(235, 235, 235))
    d.ellipse([500, 560, 540, 640], fill=(120, 110, 105))
    d.ellipse([580, 560, 620, 640], fill=(120, 110, 105))
    d.ellipse([230, 700, 850, 1900], fill=(40, 6, 6))       # Maul, grotesk aufgerissen
    d.ellipse([320, 820, 760, 1820], fill=(8, 0, 0))
    for i in range(12):
        x = 260 + i * 50
        d.polygon([(x, 760 + abs(i - 6) * 25), (x + 18, 760 + abs(i - 6) * 25), (x + 9, 850 + abs(i - 6) * 25)], fill=(190, 180, 150))
    im = im.filter(ImageFilter.GaussianBlur(3))
    return np.asarray(im).astype(np.float32) / 255


def face_frame(seed):
    r = np.random.default_rng(seed)
    f = face_base()
    yy = np.arange(OH)[:, None]
    xx = np.arange(OW)[None, :]
    amp = r.uniform(40, 110)
    sx = (xx + amp * np.sin(yy / r.uniform(25, 70) + r.uniform(0, 6)) + r.uniform(-60, 60)).astype(int) % OW
    sy = (yy + 0.4 * amp * np.sin(xx / r.uniform(40, 90))).astype(int) % OH
    g = f[sy, sx]
    out = g.copy()
    off = int(r.uniform(15, 45))
    out[..., 0] = np.roll(g[..., 0], off, 1)
    out[..., 2] = np.roll(g[..., 2], -off, 1)
    for _ in range(12):  # Block-Verschiebungen
        y0 = int(r.uniform(0, OH - 80))
        hh = int(r.uniform(10, 160))
        out[y0:y0 + hh] = np.roll(out[y0:y0 + hh], int(r.uniform(-200, 200)), 1)
    out = np.clip((out - 0.25) * 1.7 + 0.15, 0, 1)
    out *= 0.6 + 0.4 * r.random((OH // 8, OW // 8)).repeat(8, 0).repeat(8, 1)[..., None]
    return out


# ============================================================ polaroid (Szene 1)
@lru_cache(None)
def polaroid():
    """Altes, eingerissenes Polaroid mit 4 eingekreisten Pixeln (RGBA)."""
    pw, ph = 470, 560
    photo_w, photo_h = 410, 410
    cam = dict(pos=(0.3, 1.5, 0), yaw=0.02, pitch=-0.02, roll=0.0)
    sc = corridor_scene(cam, 0.3, chaos=0.0, light_mul=0.9)
    sc["fog"] = 0.05
    img, _ = render_box(cam, sc, 2.0)
    # Ausschnitt quadratisch
    crop = img[H // 2 - W // 2: H // 2 + W // 2, :]
    g = crop.mean(-1)
    g = np.clip(g ** 0.8 * 1.3, 0, 1)
    sep = np.stack([g * 1.0, g * 0.93, g * 0.78], -1)
    photo = Image.fromarray((sep * 255).astype(np.uint8)).resize((photo_w, photo_h), Image.BILINEAR)
    photo = photo.filter(ImageFilter.GaussianBlur(1.1))
    pa = np.asarray(photo).astype(np.float32)
    rng = np.random.default_rng(4)
    pa += rng.normal(0, 14, pa.shape)
    # die vier blassen Pixel am Ende des Ganges
    fx, fy = 214, 196
    pa[fy:fy + 2, fx:fx + 2] = [250, 245, 235]
    pa = np.clip(pa, 0, 255).astype(np.uint8)
    photo = Image.fromarray(pa)

    card = Image.new("RGBA", (pw, ph), (226, 221, 205, 255))
    ca = np.asarray(card).astype(np.float32)
    ca[..., :3] *= (0.82 + 0.18 * np.asarray(Image.fromarray((STAIN * 255).astype(np.uint8)).resize((pw, ph)))[..., None] / 255)
    card = Image.fromarray(ca.astype(np.uint8), "RGBA")
    card.paste(photo, (30, 30))
    d = ImageDraw.Draw(card)
    # roter Marker-Kreis (handgezeichnet, zittrig)
    pts = []
    for i in range(46):
        a = i / 40 * 2 * math.pi
        rr = 21 + 3 * math.sin(i * 1.7)
        pts.append((30 + fx + 1 + rr * math.cos(a) * 1.15, 30 + fy + 1 + rr * math.sin(a)))
    d.line(pts, fill=(185, 20, 20, 255), width=5)
    d.line([(30 + fx + 30, 30 + fy - 22), (30 + fx + 80, 30 + fy - 70)], fill=(185, 20, 20, 255), width=4)
    fnt = ImageFont.truetype(FONT_SERIF_I, 44)
    d.text((95, 470), "DON'T LOOK", font=fnt, fill=(40, 30, 30, 255))
    # Knick und Riss
    d.line([(0, 300), (pw, 255)], fill=(240, 236, 225, 160), width=3)
    tear = [(pw, 0), (pw - 120, 0), (pw - 95, 25), (pw - 70, 18), (pw - 55, 50), (pw - 30, 48), (pw, 95)]
    d.polygon(tear, fill=(0, 0, 0, 0))
    return np.asarray(card).astype(np.float32) / 255


@lru_cache(None)
def thumbs():
    """Daumen, die das Foto halten (RGBA, Kartenkoordinaten +Rand)."""
    im = Image.new("RGBA", (700, 800), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    skin = (150, 112, 92, 255)
    _limb(d, (40, 800), (150, 520), 70, 46, skin)
    _limb(d, (660, 800), (560, 540), 70, 46, skin)
    d.ellipse([120, 480, 175, 540], fill=(175, 140, 120, 255))
    d.ellipse([530, 500, 585, 560], fill=(175, 140, 120, 255))
    # Blut an den Fingern
    d.ellipse([140, 560, 190, 640], fill=(110, 20, 15, 210))
    d.ellipse([520, 600, 560, 660], fill=(110, 20, 15, 190))
    im = im.filter(ImageFilter.GaussianBlur(5))
    return np.asarray(im).astype(np.float32) / 255


# ============================================================ szenen
def corridor_scene(cam, t, chaos=0.0, light_mul=1.0, end_z=None, door=None):
    cz = cam["pos"][2]
    kc = int(round(cz / 6))
    lights = []
    for k in range(kc - 4, kc + 6):
        fl = float(flicker(k, t, chaos)) * light_mul
        lights.append(((0.0, 2.45, k * 6.0), (0.85, 1.0, 0.9), 1.6 * fl, 0.55))
    z1 = end_z if end_z is not None else 1e5
    gray = (0.62, 0.64, 0.6)
    return dict(bounds=(-1.4, 1.4, 0.0, 2.7, -1e5, z1), tints=[gray, gray, (0.5, 0.5, 0.47), (0.5, 0.52, 0.5), gray, gray],
                lights=lights, ceiling_lights=True, chaos=chaos, light_mul=light_mul, fog=0.075,
                ambient=0.012, door=door, exit_sign=end_z is not None)


def room_scene(t, gap, gap_mod, bolt=None, closing=None, blown=False, beyond=0.0, red=1.0):
    gray = (0.55, 0.53, 0.52)
    pulse = 0.55 + 0.45 * math.sin(t * 2.1) ** 2
    lights = [((-2.3, 2.5, 3.2), (1.0, 0.08, 0.05), 1.3 * red * pulse, 0.45),
              ((0.9, 2.55, 0.4), (1.0, 0.12, 0.08), 0.22 * red * pulse, 1.2),
              ((1.6, 1.4, 4.5), (0.55, 0.6, 0.7), 0.10, 0.3),
              ((0.3, 1.8, 2.4), (0.6, 0.65, 0.75), 0.9, 0.5)]
    if blown:
        lights.append(((0.0, 1.4, -0.6), (0.75, 0.82, 0.85), 2.4 * beyond, 0.4))
    elif gap > 0:
        lights.append(((0.0, 0.05, 0.05), (0.7, 0.85, 0.95), 0.6 * gap * float(gap_mod(np.array([0.0]))[0]), 3.0))
    door = dict(face=4, z=0.0, bolt=bolt, gap=gap, gap_mod=gap_mod, blown=blown,
                beyond_light=beyond, hside=-1)
    if closing is not None:
        door["open_frac"] = 1 - closing
        door["beyond"] = (0.55, 0.65, 0.6)
    return dict(bounds=(-2.5, 2.5, 0.0, 2.8, 0.0, 6.0), tints=[gray] * 2 + [(0.45, 0.43, 0.42), (0.4, 0.4, 0.4)] + [gray] * 2,
                paint=(0.7, 0.72, 0.68), lights=lights, fog=0.05, ambient=0.008, door=door)


# ---- Kamera-Pfade
def corridor_z(t):
    """Zurueckgelegte Strecke im Korridor ab t=10."""
    segs = [(10.0, 4.4), (13.3, 1.6), (16.6, 6.8), (26.0, 0.0)]
    z = 0.0
    for (ta, v), (tb, _) in zip(segs[:-1], segs[1:]):
        if t <= ta:
            break
        z += v * (min(t, tb) - ta)
    return z


Z_FIG = 6 * round((corridor_z(13.6) - 12.0) / 6) + 0.6
Z_END = corridor_z(25.92) + 1.05


def corridor_cam(t):
    z = corridor_z(t)
    running = t < 13.3 or t > 16.6
    step = 2.7 if t < 16.6 else 3.4
    amp = 1.0 if running else 0.25
    if 25.3 < t < 26.0:
        amp *= 1.4
    ph = 2 * math.pi * step * t
    bob = 0.07 * amp * abs(math.sin(ph))
    x = 0.07 * amp * math.sin(ph / 2) + 0.15 * wobble(t, 2, (0.3, 0.5), (1, 0.5))
    yaw = 0.05 * amp * math.sin(ph / 2) + 0.04 * wobble(t, 3)
    pitch = -0.05 + 0.05 * amp * math.sin(ph) + 0.025 * wobble(t, 4)
    roll = 0.05 * amp * math.sin(ph / 2) + 0.03 * wobble(t, 5)
    # Blick ueber die Schulter
    look = sm((t - 13.3) / 0.42) - sm((t - 16.62) / 0.22)
    yaw += look * 3.02
    pitch += look * 0.02
    roll += look * 0.06 * math.sin(t * 3)
    if t > 16.6:  # Panik: staerkeres Wackeln
        yaw += 0.07 * wobble(t * 2.3, 6)
        roll += 0.06 * wobble(t * 2.1, 7)
    if 25.0 < t < 26.0:  # Tuer aufreissen: nach rechts reissen
        k = sm((t - 25.45) / 0.3)
        yaw += -0.25 * k
        roll += 0.15 * k
    return dict(pos=(x, 1.62 + bob, z), yaw=yaw, pitch=pitch, roll=roll)


def room_cam(t):
    # 26.0: direkt hinter der Tuer, Blick zur Tuer (yaw=pi)
    shake = 1.0
    if t < 28.0:
        pos = np.array([0.1, 1.5, 1.75])
        yaw, pitch, roll = math.pi + 0.05, -0.12 - 0.08 * sm((t - 26.5) / 0.5), 0.0
    else:
        k = sm((t - 28.0) / 6.5)
        pos = np.array([lerp(0.1, 1.95, k), lerp(1.55, 0.62, sm((t - 29.0) / 5.0)), lerp(1.75, 5.3, k)])
        target = np.array([0.0, 1.0, 0.0])
        dv = target - pos
        yaw = math.atan2(dv[0], dv[2])
        pitch = math.atan2(dv[1], math.hypot(dv[0], dv[2])) * 0.8
        roll = 0.0
    if t < 27.4:
        shake = 2.2
    trem = 0.012 * shake
    yaw += trem * wobble(t * 3, 11) + 0.01 * wobble(t, 15)
    pitch += trem * wobble(t * 3.3, 12)
    roll += trem * wobble(t * 2.7, 13)
    return dict(pos=tuple(pos), yaw=yaw, pitch=pitch, roll=roll)


# ============================================================ post / overlay
def post(img, t, tint=(1, 1, 1), gain=1.0, ab=1, gamma=0.9):
    img = np.clip(img * gain, 0, None)
    img = img / (1 + img * 0.35)  # weiches Tonemapping
    img = img ** gamma
    img *= np.asarray(tint, np.float32)
    img *= VIGN[..., None]
    if ab:
        img[..., 0] = np.roll(img[..., 0], ab, 1)
        img[..., 2] = np.roll(img[..., 2], -ab, 1)
    return np.clip(img, 0, 1)


def tracking_glitch(img, t, strength):
    """VHS-Tracking: verschobene Zeilenbaender + Rauschband."""
    if strength <= 0:
        return img
    r = np.random.default_rng(int(t * 1000) % 2**31)
    h = img.shape[0]
    for _ in range(int(2 + 8 * strength)):
        y0 = int(r.uniform(0, h - 10))
        hh = int(r.uniform(2, 40 * strength + 3))
        img[y0:y0 + hh] = np.roll(img[y0:y0 + hh], int(r.uniform(-50, 50) * strength), 1)
    yb = int(((t * 0.37) % 1.0) * h)
    band = slice(max(0, yb - 6), min(h, yb + 6))
    img[band] = img[band] * 0.4 + r.random(img[band].shape[:2])[..., None] * 0.6 * strength
    return img


def static_frame(t, intensity=1.0, w=270, h=480):
    r = np.random.default_rng(int(t * 997) % 2**31)
    n = r.random((h, w)).astype(np.float32)
    rows = r.random((h, 1)).astype(np.float32)
    n = 0.65 * n + 0.35 * rows
    roll = int((t * 210) % h)
    bar = np.exp(-((np.arange(h) - roll) / 30.0) ** 2)[:, None]
    n = n * (0.75 + 0.5 * bar)
    img = np.repeat(np.repeat(n, OH // h, 0), OW // w, 1)
    img = img * intensity
    return np.repeat(img[..., None], 3, 2)


GRAIN = None


def grain(t):
    global GRAIN
    if GRAIN is None:
        r = np.random.default_rng(99)
        GRAIN = [r.normal(0, 1, (OH // 2, OW // 2)).astype(np.float32) for _ in range(6)]
    g = GRAIN[int(t * FPS) % 6]
    off = int(t * FPS * 37) % 200
    g = np.roll(g, off, 1)
    return np.repeat(np.repeat(g, 2, 0), 2, 1)


SCAN = (0.9 + 0.1 * (np.arange(OH) % 3 != 0)).astype(np.float32)[:, None, None]


@lru_cache(None)
def emoji_img(ch, size):
    try:
        fnt = ImageFont.truetype(FONT_EMOJI, 109)
        im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((10, 10), ch, font=fnt, embedded_color=True)
        bb = im.getbbox()
        if not bb:
            return None
        im = im.crop(bb)
        sc = size / im.height
        return im.resize((int(im.width * sc), size), Image.LANCZOS)
    except Exception:
        return None


@lru_cache(None)
def text_layer(key):
    """Regeltexte als RGBA (OW x Hoehe)."""
    spec = RULES[key]
    lines = spec["lines"]
    head_f = ImageFont.truetype(FONT_BOLD, spec.get("head_size", 100))
    body_f = ImageFont.truetype(FONT_BOLD, spec.get("size", 76))
    im = Image.new("RGBA", (OW, 640), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    y = 20
    for i, (txt, col) in enumerate(lines):
        f = head_f if i == 0 and spec.get("head", True) else body_f
        bb = d.textbbox((0, 0), txt, font=f)
        tw = bb[2] - bb[0]
        x = (OW - tw) // 2
        for ox, oy in [(-4, 0), (4, 0), (0, -4), (0, 4), (-3, -3), (3, 3), (3, -3), (-3, 3)]:
            d.text((x + ox, y + oy), txt, font=f, fill=(0, 0, 0, 255))
        d.text((x, y), txt, font=f, fill=col)
        y += (bb[3] - bb[1]) + 26
    em = spec.get("emoji")
    if em:
        ei = emoji_img(em, 96)
        if ei is not None:
            im.alpha_composite(ei, ((OW - ei.width) // 2, y + 6))
            y += 110
    bb = im.getbbox()
    im = im.crop((0, 0, OW, min(640, (bb[3] if bb else 100) + 12)))
    return np.asarray(im).astype(np.float32) / 255


RED = (235, 30, 30, 255)
WHITE = (240, 238, 232, 255)
RULES = {
    1: dict(lines=[("RULE 1:", RED), ("NEVER look", WHITE), ("at his face.", WHITE)], emoji="👁️"),
    2: dict(lines=[("RULE 2:", RED), ("No photos.", WHITE), ("No videos.", WHITE)], emoji="📵"),
    3: dict(lines=[("RULE 3:", RED), ("If you hear", WHITE), ("him crying,", WHITE), ("RUN.", RED)], emoji="🏃"),
    4: dict(lines=[("RULE 4:", RED), ("Hiding is", WHITE), ("pointless.", WHITE)], emoji="🚪"),
    5: dict(lines=[("RULE 5:", RED), ("There is", WHITE), ("no escape.", WHITE)], emoji="💀"),
    "lost": dict(lines=[("SIGNAL LOST", RED), ("CAM-07 // END OF RECORDING", WHITE)], head_size=70, size=36),
    "q": dict(lines=[("Did YOU see", WHITE), ("his face?", WHITE)], head=False, size=84),
    "follow": dict(lines=[("Follow for more.", RED)], head=False, size=66),
}
# (key, start, ende, y-position in Ausgabepixeln)
TEXT_TIMELINE = [
    (1, 4.2, 9.75, 700), (2, 11.0, 15.3, 640), (3, 23.0, 29.5, 640),
    (4, 40.8, 46.6, 640), (5, 50.3, 54.8, 760),
    ("lost", 55.0, 57.8, 720), ("q", 58.0, 65.0, 760), ("follow", 61.5, 65.0, 1000),
]


def draw_text(out, t):
    for key, t0, t1, ypos in TEXT_TIMELINE:
        if not (t0 <= t < t1):
            continue
        lay = text_layer(key)
        age, left = t - t0, t1 - t
        r = np.random.default_rng(int(t * FPS) * 31 + hash(str(key)) % 1000)
        glitch = 0.0
        if age < 0.45:
            glitch = 1 - age / 0.45
        elif left < 0.25:
            glitch = 1 - left / 0.25
        elif r.random() < 0.06:
            glitch = 0.5
        if age < 0.45 and r.random() < 0.3 * glitch:
            continue  # flackert beim Erscheinen
        L = lay.copy()
        if glitch > 0:
            h = L.shape[0]
            for _ in range(int(3 + 10 * glitch)):
                y0 = int(r.uniform(0, h - 4))
                hh = int(r.uniform(3, 30))
                L[y0:y0 + hh] = np.roll(L[y0:y0 + hh], int(r.uniform(-90, 90) * glitch), 1)
        # Slam-Zoom beim Erscheinen
        sc = 1.0 + 0.25 * max(0.0, 1 - age / 0.12)
        if sc > 1.001:
            pil = Image.fromarray((L * 255).astype(np.uint8), "RGBA")
            nw, nh = int(OW * sc), int(L.shape[0] * sc)
            pil = pil.resize((nw, nh), Image.BILINEAR).crop(((nw - OW) // 2, 0, (nw - OW) // 2 + OW, nh))
            L = np.asarray(pil).astype(np.float32) / 255
        hL = L.shape[0]
        y0 = ypos - hL // 2 + int(r.uniform(-4, 4) * glitch)
        y1 = min(OH, y0 + hL)
        L = L[: y1 - y0]
        a = L[..., 3:4]
        col = L[..., :3]
        ca = int(10 * glitch) + 3
        reg = out[y0:y1]
        rs = np.roll(col, ca, 1)
        bs = np.roll(col, -ca, 1)
        rgb = np.stack([rs[..., 0], col[..., 1], bs[..., 2]], -1)
        out[y0:y1] = reg * (1 - a) + rgb * a


@lru_cache(None)
def hud_fonts():
    return ImageFont.truetype(FONT_BOLD, 40), ImageFont.truetype(FONT_MED, 32)


def draw_hud(out, t):
    if t >= 49.9:
        return
    f1, f2 = hud_fonts()
    im = Image.new("RGBA", (OW, 140), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if int(t * 1.6) % 2 == 0:
        d.ellipse([64, 36, 94, 66], fill=(230, 20, 20, 255))
    d.text((108, 28), "REC", font=f1, fill=(240, 240, 240, 255))
    secs = 3 * 3600 + 14 * 60 + 7 + t
    ts = f"{int(secs // 3600):02d}:{int(secs % 3600 // 60):02d}:{int(secs % 60):02d}"
    d.text((560, 30), f"CAM-07  {ts}", font=f2, fill=(235, 235, 235, 230))
    d.text((64, 84), "D-9341 // SITE-19 // SUBLEVEL 4", font=f2, fill=(220, 220, 220, 200))
    a = np.asarray(im).astype(np.float32) / 255
    y0 = 220
    reg = out[y0:y0 + 140]
    out[y0:y0 + 140] = reg * (1 - a[..., 3:4] * 0.85) + a[..., :3] * a[..., 3:4] * 0.85


def upscale(img):
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    return np.asarray(pil.resize((OW, OH), Image.BILINEAR)).astype(np.float32) / 255


def shift_rot(img, dx, dy, ang):
    """Ganze Ebene verschieben/drehen (Erschuetterung)."""
    if abs(dx) < 0.5 and abs(dy) < 0.5 and abs(ang) < 1e-4:
        return img
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    pil = pil.rotate(math.degrees(ang), resample=Image.BILINEAR, translate=(dx, dy))
    return np.asarray(pil).astype(np.float32) / 255


# ============================================================ szenen-renderer
def scene1(t):
    """0-10 s: Polaroid in zitternden Haenden, Erstarren, Dunkelheit."""
    img = np.zeros((H, W, 3), np.float32)
    # Hintergrund: Taschenlampe auf Betonboden, fast schwarz
    cam = dict(pos=(0, 1.6, 0), yaw=0.0, pitch=-0.35, roll=0.0)
    frozen = t >= 3.8
    te = min(t, 3.8)
    shake_amp = 0.0 if frozen else 1.0
    lower = sm((t - 5.2) / 3.5)
    cam["pitch"] = -0.55 + 0.5 * lower + 0.03 * shake_amp * wobble(te * 4, 1)
    cam["yaw"] = 0.03 * shake_amp * wobble(te * 4, 2)
    sc = dict(bounds=(-2, 2, 0, 2.6, -50, 60), tints=[(0.5, 0.5, 0.48)] * 6, lights=[], fog=0.12, ambient=0.0,
              flashlight=dict(dir=tuple(rot(cam["yaw"], cam["pitch"], 0) @ np.array([0, 0, 1.0])), power=1.2 * (0.8 + 0.2 * math.sin(t * 40) ** 8)))
    bg, _ = render_box(cam, sc, t)
    img += bg
    # unscharfe Gestalt in der Dunkelheit (kaum sichtbar)
    if t > 8.2:
        p = project(cam, (0.0, 0.0, 14.0))
        if p:
            vis = 0.06 * sm((t - 8.2) / 1.2)
            composite(img, sprite_kneel(False), p[0], p[1], 1.4 * F / p[2], light=vis, alpha=0.9)
    # Polaroid + Daumen
    card = polaroid()
    th = thumbs()
    zoom = 1.0 + 0.9 * sm((te - 1.2) / 2.3)
    shx = 7 * shake_amp * wobble(te * 7, 3) + 4 * shake_amp * wobble(te * 15, 8)
    shy = 7 * shake_amp * wobble(te * 7, 4) + 4 * shake_amp * wobble(te * 15, 9)
    ang = 0.03 * shake_amp * wobble(te * 5, 5) - 0.04
    drop = lower * 900
    # Bild: Karte+Daumen gemeinsam als Ebene
    layer = Image.new("RGBA", (700, 800), (0, 0, 0, 0))
    layer.alpha_composite(Image.fromarray((card * 255).astype(np.uint8), "RGBA"), (115, 120))
    layer.alpha_composite(Image.fromarray((th * 255).astype(np.uint8), "RGBA"), (0, 0))
    base_scale = 0.78 * zoom
    lw, lh = int(700 * base_scale), int(800 * base_scale)
    layer = layer.resize((lw, lh), Image.BILINEAR).rotate(math.degrees(ang), resample=Image.BILINEAR, expand=False)
    la = np.asarray(layer).astype(np.float32) / 255
    # Fokuspunkt: eingekreiste Pixel -> bei Zoom ins Bildzentrum
    focus = np.array([(115 + 30 + 215) * base_scale, (120 + 30 + 197) * base_scale])
    center_lay = np.array([lw / 2, lh / 2])
    zk = sm((te - 1.2) / 2.3)
    anchor = center_lay * (1 - zk) + focus * zk
    x0 = int(W / 2 - anchor[0] + shx)
    y0 = int(H / 2 + 40 * (1 - zk) - anchor[1] + shy + drop)
    # Taschenlampenlicht auf dem Foto
    light = 0.95 + 0.05 * math.sin(t * 40) ** 8
    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(W, x0 + lw), min(H, y0 + lh)
    if ix1 > ix0 and iy1 > iy0:
        s = la[iy0 - y0: iy1 - y0, ix0 - x0: ix1 - x0]
        a = s[..., 3:4]
        img[iy0:iy1, ix0:ix1] = img[iy0:iy1, ix0:ix1] * (1 - a) + s[..., :3] * light * a
    if frozen and t < 4.35:  # Bass-Drop-Ruck
        img = np.roll(img, int(18 * (1 - (t - 3.8) / 0.55)), 0)
    return post(img, t, tint=(1.0, 0.97, 0.9), gain=1.0, ab=1 if t < 4.2 else 2)


def scene_corridor(t):
    cam = corridor_cam(t)
    chaos = 0.0
    light_mul = 1.0
    if 15.5 <= t < 16.6:
        chaos = 0.3
    if t >= 16.6:
        chaos = 0.25
    end_z = Z_END if t >= 22.0 else None
    door = None
    if end_z is not None:
        of = sm((t - 25.5) / 0.25)
        door = dict(face=5, z=end_z, hside=1, open_frac=of, beyond=(0.0, 0.0, 0.0))
    sc = corridor_scene(cam, t, chaos, light_mul, end_z, door)
    img, depth = render_box(cam, sc, t)
    # die Gestalt am Ende des Ganges
    if 13.0 < t < 16.9:
        p = project(cam, (0.05, 0.0, Z_FIG))
        if p:
            reveal = t >= 15.5
            kf = float(flicker(round(Z_FIG / 6), t, chaos))
            lit = 0.75 * kf * math.exp(-p[2] * 0.05) + 0.03
            jit = (np.random.default_rng(int(t * FPS)).uniform(-2, 2) if reveal else 0)
            composite(img, sprite_kneel(reveal), p[0] + jit, p[1], 1.45 * F / p[2], light=lit)
    k = 0.0
    if 22.0 <= t < 22.08 or 9.95 <= t < 10.08:
        k = 1.0
    img = post(img, t, tint=(0.88, 1.0, 0.92), gain=1.45, ab=1 if t < 16.6 else 2)
    if t >= 16.6:
        img = tracking_glitch(img, t, 0.25)
    if 15.5 <= t < 16.5:
        img = tracking_glitch(img, t, 0.15)
    return img, k


def gap_mod_factory(t):
    def gm(px):
        m = 0.8 + 0.2 * math.sin(t * 17) ** 2
        if 30.4 <= t < 39.0:
            c1 = 0.25 * math.sin(t * 0.9)
            c2 = c1 + 0.32
            sh = np.exp(-((px - c1) / 0.09) ** 2) + np.exp(-((px - c2) / 0.09) ** 2)
            dim = 0.35 + 0.15 * math.sin(t * 2.3)
            return m * dim * (1 - 0.95 * np.clip(sh, 0, 1))
        if 39.6 <= t < 40.0:
            return 0.0 * px
        return m + 0.0 * px
    return gm


def scene_room(t):
    """26-48.3 s: Zelle hinter der Stahltuer."""
    if t < 35.0:
        cam = room_cam(t)
    else:
        cam = room_cam(34.99)
        cam["yaw"] += 0.012 * wobble(t * 3, 21)
        cam["pitch"] += 0.012 * wobble(t * 3, 22)
        if t < 40:
            push = sm((t - 35) / 5)
            p = np.array(cam["pos"])
            cam["pos"] = tuple(p + np.array([-0.15, 0.0, -0.35]) * push)
    blown = t >= 40.0
    closing = sm((t - 26.0) / 0.28) if t < 26.3 else None
    bolt = sm((t - 26.7) / 0.5) if not blown else None
    gap = 1.0
    beyond = 0.0
    red = 1.0
    if blown:
        beyond = 0.9 * float(flicker(3, t, 0.25)) + 0.1
        red = 0.9 if t < 43 else (0.6 + 0.4 * float(flicker(5, t, 0.4)))
        # Erschuetterung
        e = math.exp(-(t - 40.0) * 2.2)
        cam = dict(cam)
        cam["yaw"] += 0.12 * e * wobble(t * 9, 31)
        cam["pitch"] += 0.10 * e * wobble(t * 9, 32)
        cam["roll"] += 0.14 * e * wobble(t * 8, 33)
    sc = room_scene(t, gap, gap_mod_factory(t), bolt=bolt, closing=closing, blown=blown, beyond=beyond, red=red)
    img, depth = render_box(cam, sc, t)
    fx = {}
    if blown:
        dt = t - 40.0
        # Staub
        dust = 0.35 * math.exp(-dt * 0.35)
        dn = samp(CONC, _xs / W * 0.6 + dt * 0.03, _ys / H * 0.6 - dt * 0.02)
        img = img * (1 - dust * 0.6) + (dust * dn * 0.25)[..., None] * np.array([0.6, 0.55, 0.5])
        # Truemmerteile
        r = np.random.default_rng(5)
        for _ in range(140):
            v = r.normal(0, 1, 3) * np.array([2.0, 1.2, 1.0]) + np.array([0, 1.0, 5.5])
            p0 = np.array([r.uniform(-0.6, 0.6), r.uniform(0.1, 2.0), 0.05])
            p = p0 + v * dt + np.array([0, -4.9, 0]) * dt * dt
            if p[1] < 0.02:
                continue
            pr = project(cam, p)
            if pr and 0 <= pr[0] < W and 0 <= pr[1] < H:
                s = max(1, int(0.06 * F / pr[2]))
                xx, yy = int(pr[0]), int(pr[1])
                img[yy:yy + s, xx:xx + s] = 0.05
        # Gestalt in der Tueroeffnung
        if 45.5 <= t < 48.3:
            vis = float(flicker(3, t, 0.25)) > 0.3 or t > 46.2
            if vis:
                lunge = sm((t - 47.6) / 0.7)
                zf = lerp(-0.4, 4.6, lunge)
                pr = project(cam, (lerp(0.0, 1.6, lunge), lerp(0.0, 0.2, lunge), zf))
                if pr:
                    tw = 0.0
                    if 46.4 < t < 47.6:  # Kopf zuckt
                        tw = 3 * math.sin(t * 40) * (np.random.default_rng(int(t * 30)).random() > 0.6)
                    composite(img, sprite_stand(), pr[0] + tw, pr[1], 2.25 * F / pr[2], light=1.0)
        fx["dust"] = dust
    gain = 1.35
    if 40.0 <= t < 40.07:
        img = img * 0.2 + 0.9
    img = post(img, t, tint=(1.0, 0.92, 0.9), gain=gain, ab=2)
    if blown:
        img = tracking_glitch(img, t, 0.25 * math.exp(-(t - 40.0) * 0.8) + 0.05)
    # fliegende Tuer
    if 40.0 <= t < 40.4:
        k = (t - 40.0) / 0.4
        sw = int(lerp(120, 1400, k ** 1.6))
        sh = int(sw * 1.8)
        door = Image.new("RGB", (60, 108), (55, 58, 64))
        dd = ImageDraw.Draw(door)
        dd.rectangle([2, 2, 57, 105], outline=(30, 31, 35), width=4)
        for yy in (28, 54, 80):
            dd.line([(2, yy), (57, yy)], fill=(38, 40, 45), width=3)
        for yy in range(8, 104, 12):
            dd.point([(6, yy), (53, yy)], fill=(90, 92, 98))
        dd.ellipse([10, 44, 22, 58], fill=(60, 10, 8))
        dd.line([(10, 70), (22, 95)], fill=(60, 10, 8), width=4)
        dpil = door.resize((max(1, sw), max(1, sh)), Image.BILINEAR).rotate(-k * 140, expand=True, resample=Image.BILINEAR)
        dpil = dpil.filter(ImageFilter.BoxBlur(max(1, int(sw * 0.04))))
        da = np.asarray(dpil).astype(np.float32) / 255
        mask = (da.sum(-1) > 0.02)[..., None]
        cx = int(W / 2 - k * k * 700)
        cy = int(H / 2 - 40 - k * 200)
        hh, ww = da.shape[:2]
        x0, y0 = cx - ww // 2, cy - hh // 2
        ix0, iy0, ix1, iy1 = max(0, x0), max(0, y0), min(W, x0 + ww), min(H, y0 + hh)
        if ix1 > ix0 and iy1 > iy0:
            s = da[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] * 0.55
            m = mask[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
            img[iy0:iy1, ix0:ix1] = np.where(m, s, img[iy0:iy1, ix0:ix1])
    return img


# ============================================================ frame dispatcher
def frame(i):
    t = i / FPS
    glitch_cut = 0.0
    shake = (0, 0, 0)
    if t < 10.0:
        img = scene1(t)
        if 9.75 <= t:
            img = tracking_glitch(img, t, 1.0)
        out = upscale(img)
    elif t < 26.0:
        img, k = scene_corridor(t)
        out = upscale(img)
        glitch_cut = k
        if t >= 16.5:
            e = 1.0 if t < 17.0 else 0.35
            shake = (14 * e * wobble(t * 6, 41), 14 * e * wobble(t * 6, 42), 0.012 * e * wobble(t * 5, 43))
    elif t < 48.3:
        img = scene_room(t)
        out = upscale(img)
        if 26.25 <= t < 26.6:
            e = 1 - (t - 26.25) / 0.35
            shake = (20 * e * wobble(t * 20, 51), 20 * e * wobble(t * 20, 52), 0.02 * e * wobble(t * 20, 53))
        if 40.0 <= t < 42.5:
            e = math.exp(-(t - 40.0) * 2.0)
            shake = (60 * e * wobble(t * 25, 61), 60 * e * wobble(t * 25, 62), 0.05 * e * wobble(t * 20, 63))
    else:
        out = scene_end(t)
    if shake != (0, 0, 0):
        out = shift_rot(out, *shake)
    if glitch_cut > 0:
        out = out * 0.3 + static_frame(t, 0.9) * 0.7
    draw_hud(out, t)
    draw_text(out, t)
    out = out * SCAN
    out = out + grain(t)[..., None] * 0.035
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


FACE_FRAMES = {49.30: 1, 49.33: 2, 49.43: 3, 49.47: 3, 49.57: 4, 49.60: 5, 49.70: 6, 49.80: 7}


def scene_end(t):
    """48.3 s+: Griff, Hochreissen, Gesicht, Rauschen, Outro."""
    if t < 49.3:
        # Raum, Kamera wird hochgerissen; Hand fuellt das Bild
        cam = room_cam(34.99)
        lift = sm((t - 48.75) / 0.5)
        p = np.array(cam["pos"])
        cam["pos"] = tuple(p + np.array([-0.6, 1.9, -0.8]) * lift)
        cam["pitch"] += -0.9 * lift + 0.15 * wobble(t * 12, 71)
        cam["roll"] += 1.1 * lift + 0.1 * wobble(t * 12, 72)
        cam["yaw"] += 0.6 * lift
        sc = room_scene(t, 0, gap_mod_factory(t), blown=True, beyond=0.6, red=0.8)
        img, _ = render_box(cam, sc, t)
        img = post(img, t, tint=(1.0, 0.9, 0.9), gain=1.6, ab=3)
        img = tracking_glitch(img, t, 0.6)
        out = upscale(img)
        hand = hand_layer()
        k = sm((t - 48.3) / 0.35)
        sc_ = lerp(0.35, 1.25, k)
        hw, hh = int(OW * sc_), int(OH * sc_)
        hp = Image.fromarray((hand * 255).astype(np.uint8), "RGBA").resize((hw, hh), Image.BILINEAR)
        hp = hp.rotate(-25 * lift, resample=Image.BILINEAR)
        ha = np.asarray(hp).astype(np.float32) / 255
        x0 = (OW - hw) // 2 + int(80 * wobble(t * 9, 73))
        y0 = (OH - hh) // 2 + int(lerp(500, 250, k)) + int(400 * lift)
        ix0, iy0, ix1, iy1 = max(0, x0), max(0, y0), min(OW, x0 + hw), min(OH, y0 + hh)
        if ix1 > ix0 and iy1 > iy0:
            s = ha[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
            a = s[..., 3:4]
            out[iy0:iy1, ix0:ix1] = out[iy0:iy1, ix0:ix1] * (1 - a) + s[..., :3] * 0.85 * a
        out = shift_rot(out, 40 * wobble(t * 20, 74), 40 * wobble(t * 20, 75), 0.04 * wobble(t * 15, 76))
        return out
    if t < 49.9:
        key = round(math.floor(t * FPS) / FPS, 2)
        seed = FACE_FRAMES.get(key)
        if seed is not None:
            return face_frame(seed)
        return static_frame(t, 1.0)
    # Rauschen bis zum Ende, mit unterschwelligem Gesichts-Frame
    inten = 1.0 if t < 55 else 0.55
    out = static_frame(t, inten)
    if 63.0 <= t < 63.07:
        out = out * 0.5 + face_frame(9) * 0.5
    if t >= 55:
        out *= 0.75
    return out


# ============================================================ main
def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--still":
        os.makedirs("stills", exist_ok=True)
        for s in sys.argv[2:]:
            i = int(round(float(s) * FPS))
            Image.fromarray(frame(i)).save(f"stills/f_{float(s):05.2f}.png")
            print("still", s)
        return
    nframes = int(DUR * FPS)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
         "-pix_fmt", "yuv420p", "video_noaudio.mp4"], stdin=subprocess.PIPE)
    with Pool(4) as pool:
        for n, fr in enumerate(pool.imap(frame, range(nframes), chunksize=4)):
            ff.stdin.write(fr.tobytes())
            if n % 150 == 0:
                print(f"frame {n}/{nframes}", flush=True)
    ff.stdin.close()
    ff.wait()
    print("fertig")


if __name__ == "__main__":
    main()
