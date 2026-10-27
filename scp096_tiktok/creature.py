"""SCP-096 als 3D-Skelett-Rig mit schattierten Kapsel-Segmenten.

Posen (lokal, Meter, y oben, Blickrichtung +z):
    pose_kneel(t, reveal)  kniend, Haende vor dem Gesicht / weggerissen
    pose_stand(t)          gebueckt in der Tuer, zuckend
    pose_run(t, phase)     Sprint mit fuchtelnden Armen
Gerendert wird per Raster in NumPy mit Z-Buffer, Skin-Shading, Rippen,
Gesicht (Augen, Maul mit Kiefer-Parameter) und Klauen.
"""
import math

import numpy as np


def _v(*a):
    return np.array(a, dtype=np.float64)


def _n(v):
    return v / (np.linalg.norm(v) + 1e-9)


def _roty(v, a):
    c, s = math.cos(a), math.sin(a)
    return _v(c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2])


def _add_jaw(j):
    """Unterkiefer haengt entlang der Kopfachse herab - je weiter offen, desto laenger das Maul."""
    up = _n(j["crown"] - j["chin"])
    j["jawtip"] = j["chin"] - up * (0.05 + 0.3 * j["jaw"]) + _v(0, 0, 0.02)
    j["ribs"] = j["spine"] + (j["chest"] - j["spine"]) * 0.08      # Brustkorb beginnt knapp ueber der Taille
    return j


def _hand(j, side, wrist, d, curl, spread=0.55, length=0.24):
    """Lange Finger mit Klauen, aufgefaechert um Richtung d."""
    d = _n(d)
    perp = _n(np.cross(d, _v(0, 0, 1)) if abs(d[2]) < 0.9 else np.cross(d, _v(0, 1, 0)))
    fwd = _n(np.cross(perp, d))
    knuck = wrist + d * 0.15
    j[f"knuckle_{side}"] = knuck
    for i in range(4):
        off = (i - 1.5) / 1.5
        base = knuck + perp * off * 0.035
        mid = base + _n(d + perp * off * spread * 0.4 + fwd * curl * 0.5) * length * 0.55
        tip = mid + _n(d + perp * off * spread * 0.5 + fwd * curl * 1.3) * length * 0.5
        j[f"fb{i}_{side}"], j[f"fm{i}_{side}"], j[f"ft{i}_{side}"] = base, mid, tip
        j[f"fc{i}_{side}"] = tip + _n(tip - mid) * length * 0.22


# ------------------------------------------------------------------ posen
def pose_run(t, phase, lean=0.5, stride=1.0, lift=1.0, arm=1.0):
    p = phase % 1.0
    j = {}
    L1, L2, LF = 0.62, 0.60, 0.20

    def leg(ph):
        a = 0.12 * stride + 0.78 * stride * math.sin(2 * math.pi * ph)
        kf = 0.3 * lift + 1.45 * lift * max(0.0, math.cos(2 * math.pi * ph)) ** 1.4
        b = a - kf
        knee = _v(0, -math.cos(a), math.sin(a)) * L1
        ankle = knee + _v(0, -math.cos(b), math.sin(b)) * L2
        f = b + 1.25 + 0.35 * stride * math.sin(2 * math.pi * ph)
        toe = ankle + _v(0, -math.cos(f), math.sin(f)) * LF
        return knee, ankle, toe

    kl, al, tl = leg(p)
    kr, ar, tr = leg(p + 0.5)
    ground = min(al[1], ar[1], tl[1], tr[1])
    py = -ground + 0.04 + 0.05 * stride * math.cos(4 * math.pi * p)
    sway = 0.07 * stride * math.sin(2 * math.pi * p)
    pel = _v(sway, py, 0)
    twist = 0.3 * stride * math.sin(2 * math.pi * p)
    for side, sx, (k, a, to) in (("l", -1, (kl, al, tl)), ("r", 1, (kr, ar, tr))):
        hip = pel + _roty(_v(0.11 * sx, -0.02, 0), -twist * 0.6)
        j[f"hip_{side}"] = hip
        j[f"knee_{side}"] = hip + k + _v(0.03 * sx, 0, 0)
        j[f"ankle_{side}"] = hip + a + _v(0.02 * sx, 0, 0)
        j[f"toe_{side}"] = hip + to + _v(0.02 * sx, 0, 0)
    th = lean + 0.06 * math.sin(4 * math.pi * p)
    j["pelvis"] = pel
    j["spine"] = pel + _v(0, math.cos(th), math.sin(th)) * 0.32
    th2 = th + 0.2
    j["chest"] = j["spine"] + _v(0, math.cos(th2), math.sin(th2)) * 0.38
    for side, sx in (("l", -1), ("r", 1)):
        j[f"shoulder_{side}"] = j["chest"] + _roty(_v(0.245 * sx, -0.05 + 0.06 * sx * math.sin(2 * math.pi * p), -0.02), twist)
    j["neck"] = j["chest"] + _v(0, math.cos(th + 0.75), math.sin(th + 0.75)) * 0.17
    jit = 0.12 * math.sin(t * 23) * (math.sin(t * 3.1) > 0.6)
    j["chin"] = j["neck"] + _v(-sway * 0.6, -0.12, 0.13)
    tilt = 0.1 + jit
    j["crown"] = j["chin"] + _n(_v(math.sin(jit) * 0.3 - sway, math.cos(tilt), math.sin(tilt))) * 0.40
    for side, sx, off in (("l", -1, 0.0), ("r", 1, 0.5)):
        ph = 2 * math.pi * (p + off) + 0.35          # Arme gegen die Beine, mit Verzoegerung
        al_ = -1.15 * arm * math.sin(ph) + 0.25 * arm
        ab = 0.2 + 0.1 * math.sin(ph * 2 + sx)
        sh = j[f"shoulder_{side}"]
        d1 = _v(math.sin(ab) * sx, -math.cos(al_) * math.cos(ab), math.sin(al_) * math.cos(ab))
        el = sh + _n(d1) * 0.58
        e = 0.2 * arm + 0.7 * arm * max(0.0, math.sin(ph + 0.9))
        al2 = al_ + e
        d2 = _v(math.sin(ab) * sx * 0.6, -math.cos(al2) * math.cos(ab), math.sin(al2) * math.cos(ab))
        wr = el + _n(d2) * 0.58
        j[f"elbow_{side}"], j[f"wrist_{side}"] = el, wr
        _hand(j, side, wr, _n(d2 + _v(0, -0.2, 0.1)), curl=0.35 + 0.3 * math.sin(ph))
    j["jaw"] = 0.85
    return _add_jaw(j)


def pose_walk(t, phase):
    """Langsames, schweres Gehen mit haengenden, schleifenden Armen."""
    return pose_run(t, phase, lean=0.28, stride=0.42, lift=0.3, arm=0.22)


def blend(a, b, k):
    """Lineare Ueberblendung zweier Posen (gleiche Gelenke)."""
    return {key: a[key] * (1 - k) + b[key] * k for key in a}


def pose_kneel(t, reveal):
    """reveal: 0 Haende vor dem Gesicht, 1 weggerissen, Kopf hoch, Maul offen."""
    j = {}
    sob = 0.025 * max(0.0, math.sin(t * 7.3)) * (1 - reveal)
    j["pelvis"] = _v(0, 0.44, -0.18)
    for side, sx in (("l", -1), ("r", 1)):
        j[f"hip_{side}"] = _v(0.12 * sx, 0.42, -0.16)
        j[f"knee_{side}"] = _v(0.22 * sx, 0.07, 0.45)
        j[f"ankle_{side}"] = _v(0.16 * sx, 0.06, -0.15)
        j[f"toe_{side}"] = _v(0.14 * sx, 0.03, -0.36)
    hunch = 0.55 * (1 - reveal) + 0.05
    j["spine"] = j["pelvis"] + _v(0, math.cos(hunch * 0.7), math.sin(hunch * 0.7)) * 0.32 + _v(0, sob, 0)
    j["chest"] = j["spine"] + _v(0, math.cos(hunch), math.sin(hunch)) * 0.38 + _v(0, sob, 0)
    for side, sx in (("l", -1), ("r", 1)):
        j[f"shoulder_{side}"] = j["chest"] + _v(0.245 * sx, -0.04 + sob, 0)
    j["neck"] = j["chest"] + _v(0, math.cos(hunch + 0.6), math.sin(hunch + 0.6)) * 0.17
    j["chin"] = j["neck"] + _v(0, -0.06 * (1 - reveal) - 0.1 * reveal, 0.12)
    tilt = lerp(1.1, -0.15, reveal)
    shake = 0.05 * math.sin(t * 61) * reveal
    j["crown"] = j["chin"] + _n(_v(shake, math.cos(tilt), math.sin(tilt))) * (0.30 + 0.1 * reveal)
    face = (j["chin"] + j["crown"]) / 2 + _v(0, 0, 0.12)
    for side, sx in (("l", -1), ("r", 1)):
        sh = j[f"shoulder_{side}"]
        el_c = sh + _v(0.22 * sx, -0.38, 0.22)
        wr_c = face + _v(0.07 * sx, -0.12, 0.02)
        el_r = sh + _v(0.32 * sx, -0.48, 0.12)
        wr_r = sh + _v(0.62 * sx, -0.98, 0.25)
        el = el_c * (1 - reveal) + el_r * reveal
        wr = wr_c * (1 - reveal) + wr_r * reveal
        j[f"elbow_{side}"], j[f"wrist_{side}"] = el, wr
        d = (_n(face + _v(-0.05 * sx, 0.1, 0.03) - wr) * (1 - reveal)
             + _n(wr - el + _v(0, -0.3, 0)) * reveal)
        _hand(j, side, wr, d, curl=0.15 + 0.6 * reveal, spread=0.3 + 0.5 * reveal, length=0.16 + 0.08 * reveal)
    j["jaw"] = 0.1 + 0.8 * reveal
    return _add_jaw(j)


def pose_stand(t):
    j = {}
    sway = 0.03 * math.sin(t * 1.3)
    j["pelvis"] = _v(sway, 1.22, 0)
    for side, sx in (("l", -1), ("r", 1)):
        j[f"hip_{side}"] = _v(0.12 * sx + sway, 1.2, 0)
        j[f"knee_{side}"] = _v(0.2 * sx, 0.64, 0.1)
        j[f"ankle_{side}"] = _v(0.25 * sx, 0.07, 0.0)
        j[f"toe_{side}"] = _v(0.3 * sx, 0.03, 0.18)
    j["spine"] = j["pelvis"] + _v(0, 0.31, 0.06)
    j["chest"] = j["spine"] + _v(0, 0.34, 0.16)
    for side, sx in (("l", -1), ("r", 1)):
        j[f"shoulder_{side}"] = j["chest"] + _v(0.245 * sx, -0.06, -0.02)
    j["neck"] = j["chest"] + _v(0, 0.12, 0.13)
    twitch = 0.0
    k = math.sin(t * 2.7) + math.sin(t * 5.3)
    if k > 1.2:
        twitch = 0.45 * math.sin(t * 50)
    j["chin"] = j["neck"] + _v(0, -0.06, 0.1)
    j["crown"] = j["chin"] + _n(_v(math.sin(0.35 + twitch), math.cos(0.35 + twitch) * 1.0, 0.25)) * 0.34
    for side, sx in (("l", -1), ("r", 1)):
        sh = j[f"shoulder_{side}"]
        sw = 0.04 * math.sin(t * 1.1 + sx)
        el = sh + _v(0.2 * sx, -0.54, 0.06 + sw)
        wr = el + _v(0.17 * sx, -0.53, 0.04 + sw)
        j[f"elbow_{side}"], j[f"wrist_{side}"] = el, wr
        _hand(j, side, wr, _v(0.25 * sx, -1, 0.1), curl=0.4 + 0.2 * math.sin(t * 3 + sx), spread=0.45, length=0.27)
    j["jaw"] = 0.6
    return _add_jaw(j)


def lerp(a, b, k):
    return a + (b - a) * k


# ------------------------------------------------------------------ segmente
def segments():
    S = [("pelvis", "spine", 0.1, 0.06, "belly"), ("spine", "ribs", 0.06, 0.13, "belly"),
         ("ribs", "chest", 0.15, 0.19, "torso"),
         ("hip_l", "hip_r", 0.068, 0.068, "skin"),
         ("shoulder_l", "shoulder_r", 0.042, 0.042, "skin"),
         ("chest", "neck", 0.08, 0.045, "skin"), ("neck", "chin", 0.045, 0.065, "skin"),
         ("chin", "crown", 0.084, 0.088, "head"), ("jawtip", "chin", 0.05, 0.076, "jaw")]
    for s in ("l", "r"):
        S += [(f"hip_{s}", f"knee_{s}", 0.062, 0.046, "skin"), (f"knee_{s}", f"knee_{s}", 0.056, 0.056, "knee"),
              (f"knee_{s}", f"ankle_{s}", 0.048, 0.03, "skin"), (f"ankle_{s}", f"toe_{s}", 0.034, 0.02, "skin"),
              (f"shoulder_{s}", f"elbow_{s}", 0.04, 0.028, "skin"), (f"elbow_{s}", f"elbow_{s}", 0.034, 0.034, "skin"),
              (f"elbow_{s}", f"wrist_{s}", 0.028, 0.02, "skin"), (f"wrist_{s}", f"knuckle_{s}", 0.032, 0.03, "hand")]
        for i in range(4):
            S += [(f"fb{i}_{s}", f"fm{i}_{s}", 0.011, 0.009, "hand"), (f"fm{i}_{s}", f"ft{i}_{s}", 0.009, 0.0075, "hand"),
                  (f"ft{i}_{s}", f"fc{i}_{s}", 0.0075, 0.0055, "tip")]
    return S


SEGS = segments()


def hand_segments(side="r"):
    """Nahaufnahme-Hand: Unterarm, flaechige Handflaeche, Fingerglieder mit Klauen."""
    S = [(f"elbow_{side}", f"wrist_{side}", 0.034, 0.028, "skin"),
         (f"fb0_{side}", f"fb3_{side}", 0.017, 0.017, "skin"),
         (f"wrist_{side}", f"knuckle_{side}", 0.042, 0.036, "hand")]
    for i in range(4):
        S += [(f"wrist_{side}", f"fb{i}_{side}", 0.03, 0.019, "hand"),
              (f"fb{i}_{side}", f"fm{i}_{side}", 0.0115, 0.0095, "hand"),
              (f"fm{i}_{side}", f"ft{i}_{side}", 0.0095, 0.008, "hand"),
              (f"ft{i}_{side}", f"fc{i}_{side}", 0.008, 0.006, "tip")]
    return S


def pose_grab(k, close):
    """Hand im Kameraraum (Kamera im Ursprung, Blick +z). k: Annaeherung, close: Zugreifen."""
    j = {}
    wr = _v(lerp(0.06, 0.0, k), lerp(-0.5, -0.24, k), lerp(1.4, 0.36, k))
    j["elbow_r"] = wr + _v(0.12, -0.45, 0.45)
    d = _n(_v(-0.05, 1.0, -0.25 - 0.3 * close))
    _hand(j, "r", wr, d, curl=-(0.1 + 1.0 * close), spread=1.2 - 0.3 * close, length=0.26)
    return j


def to_world(j, pos, yaw, scale=1.0):
    out = {}
    for k, v in j.items():
        if k == "jaw":
            out[k] = v
            continue
        out[k] = _roty(v * scale, yaw) + np.asarray(pos, float)
    return out


# ------------------------------------------------------------------ raster
def rasterize(img_shape, cam, jw, project, rot, F, light, tex, fine, t=0.0, segs=None):
    """Gibt (rgb, alpha, depth) in Bildgroesse zurueck."""
    Hh, Ww = img_shape[:2]
    col = np.zeros((Hh, Ww, 3), np.float32)
    alpha = np.zeros((Hh, Ww), np.float32)
    zb = np.full((Hh, Ww), 1e9, np.float32)
    R = rot(cam["yaw"], cam["pitch"], cam["roll"])
    key_c = R.T @ _n(np.asarray(light["key_dir"], float))
    key_c = _v(key_c[0], key_c[1], -key_c[2])  # Konvention: Normale zur Kamera = -z
    half = _n(key_c + _v(0, 0, -1))
    rim_c = light.get("rim", 0.0)
    pp = {}
    for k, v in jw.items():
        if k == "jaw":
            continue
        pp[k] = project(cam, v)
    pel = pp.get("pelvis")
    ppm = F / pel[2] if pel else 200.0
    ox, oy = (pel[0], pel[1]) if pel else (0, 0)
    jaw = jw.get("jaw", 0.3)
    skin_lit = np.array([0.85, 0.83, 0.84], np.float32)
    skin_sh = np.array([0.26, 0.22, 0.27], np.float32)
    blood_c = np.array([0.42, 0.04, 0.05], np.float32)
    for a, b, ra, rb, mat in (segs or SEGS):
        A, B = pp.get(a), pp.get(b)
        if A is None or B is None:
            continue
        ax, ay, za = A
        bx, by, zb_ = B
        sra, srb = ra * F / za, rb * F / zb_
        if max(sra, srb) < 0.35:
            continue
        x0 = int(max(0, math.floor(min(ax - sra, bx - srb)) - 1))
        x1 = int(min(Ww, math.ceil(max(ax + sra, bx + srb)) + 2))
        y0 = int(max(0, math.floor(min(ay - sra, by - srb)) - 1))
        y1 = int(min(Hh, math.ceil(max(ay + sra, by + srb)) + 2))
        if x1 <= x0 or y1 <= y0:
            continue
        Y, X = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        X += 0.5
        Y += 0.5
        abx, aby = bx - ax, by - ay
        l2 = abx * abx + aby * aby
        if l2 < 1e-6:
            h = np.zeros_like(X)
        else:
            h = np.clip(((X - ax) * abx + (Y - ay) * aby) / l2, 0, 1)
        dx = X - (ax + abx * h)
        dy = Y - (ay + aby * h)
        d = np.sqrt(dx * dx + dy * dy)
        r = sra + (srb - sra) * h
        cov = np.clip(r - d + 0.5, 0, 1)
        m = cov > 0
        if not m.any():
            continue
        s = np.clip(d / np.maximum(r, 1e-3), 0, 1)
        c = np.sqrt(1 - s * s)
        nx = np.where(d > 1e-4, dx / np.maximum(d, 1e-4), 0) * s
        ny = -np.where(d > 1e-4, dy / np.maximum(d, 1e-4), 0) * s
        nz = -c
        wr = ra + (rb - ra) * h
        depth = (za + (zb_ - za) * h) - c * wr
        win = m & (depth < zb[y0:y1, x0:x1])
        if not win.any():
            continue
        ndl = nx * key_c[0] + ny * key_c[1] + nz * key_c[2]
        diff = np.clip((ndl + 0.12) / 1.12, 0, 1) ** 1.3
        u = (X - ox) / ppm * 1.3
        v = (Y - oy) / ppm * 1.3
        tx = tex(u, v)
        fn = fine(u * 3.0, v * 3.0)
        alb = 0.78 + 0.45 * (tx - 0.5)
        vein = np.clip(1 - np.abs(fn - 0.5) / 0.02, 0, 1) * 0.5
        base = skin_sh[None, None] + (skin_lit - skin_sh)[None, None] * diff[..., None]
        rgb = base * alb[..., None]
        rgb = rgb * (1 - vein[..., None] * np.array([0.12, 0.32, 0.04], np.float32))   # violette Adern
        spec_k = 0.18
        if mat == "torso":
            qq = (dx * aby - dy * abx) / max(math.sqrt(l2), 1e-3) / np.maximum(r, 1e-3)
            ribs = np.clip(np.cos((h + 0.07 * qq * qq) * 2 * math.pi * 6.0), 0, 1) ** 4 * ((h > 0.3) & (h < 0.92)) * (1 - s) ** 0.7 * (np.abs(qq) > 0.12)
            stern = np.exp(-(((dx * aby - dy * abx) / max(math.sqrt(l2), 1e-3)) / max((sra + srb) * 0.06, 0.3)) ** 2)
            rgb *= (1 - 0.48 * ribs * (0.4 + 0.6 * diff) - 0.15 * stern * (h > 0.4))[..., None]
            under = np.exp(-((h - 0.22) / 0.07) ** 2) * (1 - s)                  # Rippenbogen
            rgb *= (1 - 0.35 * under)[..., None]
        elif mat == "belly":
            rgb *= (0.78 + 0.22 * s)[..., None]                                   # eingefallener Bauch
        elif mat == "head" and light.get("face", True):
            q = (dx * aby - dy * abx) / max(math.sqrt(l2), 1e-3) / np.maximum(r, 1e-3)
            wr_ = np.clip(np.cos(h * 2 * math.pi * 13), 0, 1) ** 3 * (h > 0.74) * (1 - s)   # Stirnfalten
            rgb *= (1 - 0.22 * wr_)[..., None]
            for ex in (-0.36, 0.36):
                e = ((q - ex) / 0.105) ** 2 + ((h - 0.6) / 0.042) ** 2
                rgb *= (1 - 0.7 * np.clip(2.2 - e * 0.28, 0, 1))[..., None]       # tiefe, dunkle Augenhoehlen
                eye = np.array([0.74, 0.74, 0.77], np.float32) * (0.6 + 0.4 * diff[..., None])
                rgb = np.where((e < 1)[..., None], eye * (0.85 + 0.15 * (1 - e[..., None])), rgb)  # milchig, ohne Pupille
            mw = 0.18 + 0.2 * jaw
            top = 0.18 + 0.25 * jaw
            mouth = (h < top) & (np.abs(q) < mw * np.sqrt(np.clip(1 - (h / top) ** 2, 0, 1)) + 0.06 * (h < top * 0.4))
            rgb = np.where(mouth[..., None], np.array([0.05, 0.01, 0.015], np.float32), rgb)
            fold = np.exp(-((np.abs(q) - mw - 0.12) / 0.05) ** 2) * (h < top + 0.1)     # gedehnte Falten neben dem Maul
            rgb *= (1 - 0.3 * fold)[..., None]
            nose = (np.abs(np.abs(q) - 0.06) < 0.03) & (np.abs(h - top - 0.06) < 0.022)
            rgb = np.where(nose[..., None], rgb * 0.3, rgb)
        elif mat == "jaw":
            q = (dx * aby - dy * abx) / max(math.sqrt(l2), 1e-3) / np.maximum(r, 1e-3)
            if light.get("face", True):
                mw = 0.5 + 0.2 * jaw
                inner = (np.abs(q) < mw) & (h > 0.12)
                depth_k = np.clip(1 - np.abs(q) / mw, 0, 1)
                mc = np.array([0.09, 0.015, 0.02], np.float32) * (1 - 0.85 * depth_k[..., None])
                rgb = np.where(inner[..., None], mc, rgb)
                lip = (np.abs(q) >= mw) & (np.abs(q) < mw + 0.18)
                rgb = np.where(lip[..., None], rgb * 0.62, rgb)
            rgb *= (1 - 0.25 * np.clip(1 - h / 0.2, 0, 1))[..., None]
        elif mat in ("hand", "tip", "knee"):
            amt = {"hand": 0.55, "tip": 0.9, "knee": 0.35}[mat]
            bl = np.clip((tx - 0.5) * 4 + amt * 2 - 0.6, 0, 1) * amt
            rgb = rgb * (1 - bl[..., None]) + blood_c * (0.35 + 0.65 * diff[..., None]) * bl[..., None]
            spec_k = 0.35 if mat != "knee" else 0.18
        spec = np.clip(nx * half[0] + ny * half[1] + nz * half[2], 0, 1) ** 28 * spec_k
        rgb = rgb * light["key"] + light.get("ambient", 0.03) * rgb
        rgb = rgb + spec[..., None] * light["key"]
        if rim_c:
            rim = (1 - c) ** 2.5 * rim_c
            rgb = rgb + rim[..., None] * np.asarray(light.get("rim_col", (0.8, 0.85, 0.9)), np.float32)
        rgb = rgb * np.asarray(light.get("tint", (1, 1, 1)), np.float32)
        fog = math.exp(-((za + zb_) / 2) * light.get("fog", 0.0))
        rgb = rgb * fog
        sub = col[y0:y1, x0:x1]
        sub[win] = rgb[win]
        al = alpha[y0:y1, x0:x1]
        al[win] = np.maximum(al[win], cov[win])
        zz = zb[y0:y1, x0:x1]
        zz[win] = depth[win]
    return col, alpha, zb


def composite_into(img, col, alpha, strength=1.0):
    a = (alpha * strength)[..., None]
    img[:] = img * (1 - a) + col * a


def contact_shadow(img, cam, jw, project, strength=0.55):
    """Weicher Bodenschatten unter den Fuessen."""
    pts = [jw[k] for k in ("ankle_l", "ankle_r", "toe_l", "toe_r", "knee_l", "knee_r") if k in jw]
    cx = np.mean([p[0] for p in pts])
    cz = np.mean([p[2] for p in pts])
    Hh, Ww = img.shape[:2]
    mask = np.zeros((Hh, Ww), np.float32)
    ring = []
    for i in range(24):
        a = i / 24 * 2 * math.pi
        pr = project(cam, (cx + 0.45 * math.cos(a), 0.005, cz + 0.45 * math.sin(a)))
        if pr is None:
            return
        ring.append(pr[:2])
    ring = np.array(ring)
    x0, y0 = np.floor(ring.min(0)).astype(int)
    x1, y1 = np.ceil(ring.max(0)).astype(int)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(Ww, x1), min(Hh, y1)
    if x1 <= x0 or y1 <= y0:
        return
    ccx, ccy = ring.mean(0)
    rx = (ring[:, 0].max() - ring[:, 0].min()) / 2 + 1e-3
    ry = (ring[:, 1].max() - ring[:, 1].min()) / 2 + 1e-3
    Y, X = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    e = ((X - ccx) / rx) ** 2 + ((Y - ccy) / ry) ** 2
    mask[y0:y1, x0:x1] = np.clip(1 - e, 0, 1) ** 1.5
    img *= (1 - strength * mask)[..., None]
