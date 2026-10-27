"""Sounddesign fuer das SCP-096 TikTok (komplett synthetisch, kein Voice-over).

Erzeugt audio.wav (48 kHz, Stereo, 65 s).
"""
import numpy as np
import wave

SR = 48000
DUR = 65.0
N = int(SR * DUR)
rng = np.random.default_rng(96)

L = np.zeros(N)
R = np.zeros(N)
CLIP_BUS = np.zeros(N)  # wird NACH dem Limiter addiert -> echtes Clipping


# ----------------------------------------------------------------- helpers
def secs(n):
    return np.arange(n) / SR


def noise(n):
    return rng.standard_normal(n)


def fft_filter(x, lo=None, hi=None, slope=1.0):
    """Weicher Bandpass im Frequenzbereich (offline, zeitinvariant)."""
    n = len(x)
    size = 1 << int(np.ceil(np.log2(n + 1)))
    X = np.fft.rfft(x, size)
    f = np.fft.rfftfreq(size, 1 / SR)
    g = np.ones_like(f)
    if lo:
        g *= 1 / (1 + (lo / np.maximum(f, 1e-3)) ** (4 * slope))
    if hi:
        g *= 1 / (1 + (f / hi) ** (4 * slope))
    return np.fft.irfft(X * g, size)[:n]


def formant(x, freqs, bws, gains):
    n = len(x)
    size = 1 << int(np.ceil(np.log2(n + 1)))
    X = np.fft.rfft(x, size)
    f = np.fft.rfftfreq(size, 1 / SR)
    g = np.zeros_like(f)
    for fc, bw, ga in zip(freqs, bws, gains):
        g += ga / (1 + ((f - fc) / (bw / 2)) ** 2)
    return np.fft.irfft(X * g, size)[:n]


def convolve(x, ir):
    n = len(x) + len(ir) - 1
    size = 1 << int(np.ceil(np.log2(n)))
    y = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)
    return y[:n]


def reverb(x, decay=1.8, wet=0.35, pre=0.012, bright=6000, seed=0):
    r = np.random.default_rng(seed)
    n = int(SR * decay * 1.6)
    tt = secs(n)
    ir = r.standard_normal(n) * np.exp(-tt * 6.9 / decay)
    ir = fft_filter(ir, 120, bright)
    ir[: int(pre * SR)] = 0
    ir /= np.sqrt(np.sum(ir ** 2)) + 1e-9
    y = convolve(x, ir)
    out = np.zeros(len(y))
    out[: len(x)] += x * (1 - wet)
    out += y * wet * 1.4
    return out


def env_ad(n, a, d, curve=1.0):
    t = secs(n)
    e = np.where(t < a, t / max(a, 1e-6), np.exp(-(t - a) / max(d, 1e-6)))
    return e ** curve


def place(sig, start, gain=1.0, pan=0.0, bus=None):
    """pan -1 = links, +1 = rechts (equal power)."""
    i0 = int(start * SR)
    if i0 >= N:
        return
    s = sig[: max(0, N - i0)] * gain
    if i0 < 0:
        s = s[-i0:]
        i0 = 0
    a = (pan + 1) * np.pi / 4
    if bus is not None:
        bus[i0:i0 + len(s)] += s
        return
    L[i0:i0 + len(s)] += s * np.cos(a)
    R[i0:i0 + len(s)] += s * np.sin(a)


def curve(points, n=N):
    """Lineare Automationskurve aus (zeit, wert)-Paaren."""
    tt = secs(n)
    ts, vs = zip(*points)
    return np.interp(tt, ts, vs)


def harmonic(f0, nh=30, rolloff=1.0, jitter=0.0):
    """Bandbegrenzter Saegezahn aus einer (zeitvariablen) f0-Kurve."""
    if jitter:
        f0 = f0 * (1 + jitter * fft_filter(noise(len(f0)), None, 40) * 3)
    ph = 2 * np.pi * np.cumsum(f0) / SR
    out = np.zeros(len(f0))
    for k in range(1, nh + 1):
        amp = 1 / k ** rolloff
        out += amp * np.sin(k * ph) * (k * f0 < SR / 2.2)
    return out


# ----------------------------------------------------------------- sounds
def sub_drop(dur=2.8, f_start=110, f_end=28):
    n = int(SR * dur)
    t = secs(n)
    f = f_end + (f_start - f_end) * np.exp(-t * 3.2)
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * env_ad(n, 0.004, dur * 0.42)
    s = np.tanh(s * 2.2)
    hit = fft_filter(noise(n), None, 900) * env_ad(n, 0.001, 0.12) * 0.9
    return s + hit


def riser(dur=0.9):
    n = int(SR * dur)
    t = secs(n)
    s = fft_filter(noise(n), 2500, 12000) * (t / dur) ** 3
    return s * 0.5


def breath_track(t0, t1, rate, gain, seed=1, ragged=0.25):
    """Panisches Atmen: abwechselnd Ein-/Ausatmen, gefiltertes Rauschen."""
    r = np.random.default_rng(seed)
    t = t0
    inhale = True
    while t < t1:
        period = 1 / rate * (1 + ragged * (r.random() - 0.5))
        d = period * (0.48 if inhale else 0.52)
        n = int(SR * d)
        tt = secs(n)
        shape = np.sin(np.pi * np.clip(tt / d, 0, 1)) ** (0.7 if inhale else 1.3)
        src = noise(n)
        if inhale:
            s = formant(src, [900, 1800, 3200], [700, 900, 1500], [1.0, 0.7, 0.35])
        else:
            s = formant(src, [600, 1300, 2600], [600, 900, 1500], [1.0, 0.6, 0.25])
            # leichter Stimmanteil (Stoehnen)
            f0 = 140 + 20 * r.random() + 0 * tt
            s += 0.08 * formant(harmonic(f0, 20), [500, 1400], [200, 400], [1, 0.5])
        s = s * shape
        s /= np.max(np.abs(s)) + 1e-9
        place(s, t, gain * (0.8 + 0.4 * r.random()), pan=r.uniform(-0.1, 0.1))
        t += d
        inhale = not inhale


def sob_track(t0, t1, gain, seed=7, dist_lp=None, pan=0.0):
    """Unnatuerliches Schluchzen: zu tief, zu lang, leicht ringmoduliert."""
    r = np.random.default_rng(seed)
    total = int((t1 - t0) * SR)
    out = np.zeros(total)
    t = 0.0
    while t < (t1 - t0) - 0.3:
        d = r.uniform(0.35, 1.1)
        n = min(int(SR * d), total - int(t * SR))
        if n <= 100:
            break
        tt = secs(n)
        base = r.uniform(230, 330)
        f0 = base * (1 + 0.25 * np.exp(-tt * 5)) * (1 - 0.35 * (tt / d) ** 2)
        f0 *= 1 + 0.025 * np.sin(2 * np.pi * 7.5 * tt)
        v = harmonic(f0, 24, 1.1, jitter=0.01)
        v = formant(v, [380, 900, 2500], [160, 300, 600], [1.0, 0.45, 0.15])
        # Oktave tiefer, verstimmt -> unmenschlich
        v2 = harmonic(f0 * 0.497, 30, 1.0)
        v2 = formant(v2, [300, 700], [200, 300], [1.0, 0.4])
        v = v + 0.5 * v2
        v *= 1 + 0.35 * np.sin(2 * np.pi * 41 * tt)  # Ringmod / Flattern
        # Schluchz-Stoesse
        hic = 0.55 + 0.45 * (np.sin(2 * np.pi * r.uniform(5, 8) * tt) > -0.2)
        e = np.sin(np.pi * np.clip(tt / d, 0, 1)) ** 0.6 * hic
        s = v * e
        s /= np.max(np.abs(s)) + 1e-9
        i0 = int(t * SR)
        out[i0:i0 + n] += s * r.uniform(0.6, 1.0)
        t += d
        # nasses Einatmen / Schniefen
        gn = int(SR * r.uniform(0.18, 0.35))
        g = formant(noise(gn), [1500, 3500], [900, 1500], [1, 0.6])
        g *= np.sin(np.pi * secs(gn) / (gn / SR)) ** 2
        g /= np.max(np.abs(g)) + 1e-9
        i1 = int(t * SR)
        out[i1:i1 + gn] += g[: max(0, total - i1)] * 0.45
        t += gn / SR + r.uniform(0.05, 0.3)
    if dist_lp:
        out = fft_filter(out, None, dist_lp)
    out = reverb(out, decay=2.6, wet=0.55, seed=seed)
    out /= np.max(np.abs(out)) + 1e-9
    place(out, t0, gain, pan)


def scream(dur, seed=3, pitch=1.0):
    r = np.random.default_rng(seed)
    n = int(SR * dur)
    t = secs(n)
    # Tonhoehe: steiler Anstieg, dann zitterndes Plateau mit Rissen
    f0 = (170 + 520 * (1 - np.exp(-t * 4.0))) * pitch
    wob = fft_filter(r.standard_normal(n), None, 12)
    wob /= np.max(np.abs(wob)) + 1e-9
    f0 *= 1 + 0.09 * wob + 0.03 * np.sin(2 * np.pi * 11 * t)
    cracks = (fft_filter(r.standard_normal(n), None, 3) > 0.004).astype(float)
    f0 *= 1 + 0.45 * fft_filter(cracks, None, 30)
    v = harmonic(f0, 40, 0.8, jitter=0.02)
    v = formant(v, [760, 1250, 2700, 3600], [180, 260, 500, 700], [1.0, 0.8, 0.5, 0.3])
    low = harmonic(f0 * 0.5 * 1.013, 40, 0.9)
    low = formant(low, [500, 1100], [300, 400], [1.0, 0.5])
    screech = np.sin(2 * np.pi * np.cumsum(2400 + 600 * wob) / SR
                     + 4 * np.sin(2 * np.pi * np.cumsum(f0) / SR))
    hiss = fft_filter(noise(n), 1200, 6000)
    s = v + 0.7 * low + 0.12 * screech + 0.5 * hiss * (0.6 + 0.4 * wob)
    s = np.tanh(s / (np.std(s) + 1e-9) * 1.6)
    e = np.clip(t / 0.08, 0, 1) * np.clip((dur - t) / 0.25, 0, 1)
    return s * e


def footsteps(t0, t1, rate0, rate1, gain0, gain1, barefoot=True, seed=5, pan0=0.0, pan1=0.0):
    r = np.random.default_rng(seed)
    t = t0
    while t < t1:
        k = (t - t0) / max(t1 - t0, 1e-6)
        rate = rate0 + (rate1 - rate0) * k
        g = gain0 + (gain1 - gain0) * k
        n = int(SR * 0.25)
        tt = secs(n)
        thump = np.sin(2 * np.pi * (60 + 40 * np.exp(-tt * 40)) * tt) * env_ad(n, 0.002, 0.05)
        if barefoot:
            slap = fft_filter(noise(n), 700, 4500) * env_ad(n, 0.001, 0.018)
            s = thump * 1.0 + slap * 0.9
        else:
            click = fft_filter(noise(n), 1500, 9000) * env_ad(n, 0.0005, 0.01)
            s = thump * 0.7 + click * 0.8
        s /= np.max(np.abs(s)) + 1e-9
        s = reverb(s, decay=1.3, wet=0.3, seed=int(r.integers(1e6)))
        place(s, t, g * r.uniform(0.75, 1.0), pan=pan0 + (pan1 - pan0) * k)
        t += 1 / rate * r.uniform(0.85, 1.12)


def heartbeat(t0, t1, bpm0, bpm1, gain):
    t = t0
    while t < t1:
        k = (t - t0) / max(t1 - t0, 1e-6)
        bpm = bpm0 + (bpm1 - bpm0) * k
        for off, f, a in ((0.0, 52, 1.0), (0.17, 64, 0.7)):
            n = int(SR * 0.3)
            tt = secs(n)
            s = np.sin(2 * np.pi * f * tt * (1 + 0.6 * np.exp(-tt * 30))) * env_ad(n, 0.006, 0.07)
            s += 0.3 * np.sin(2 * np.pi * f * 2 * tt) * env_ad(n, 0.004, 0.04)
            place(np.tanh(s * 1.8), t + off, gain * a)
        t += 60 / bpm


def metal_hit(dur=2.2, partials=(173, 411, 689, 1123, 1730, 2390), seed=11, bright=1.0):
    r = np.random.default_rng(seed)
    n = int(SR * dur)
    t = secs(n)
    s = np.zeros(n)
    for i, p in enumerate(partials):
        p *= r.uniform(0.97, 1.03)
        s += np.sin(2 * np.pi * p * t + r.random() * 6) * np.exp(-t * (1.8 + i * 0.9)) / (1 + i * 0.5)
    crack = fft_filter(noise(n), 200, 5000 * bright) * env_ad(n, 0.0005, 0.06)
    body = fft_filter(noise(n), None, 400) * env_ad(n, 0.001, 0.25)
    out = 0.6 * s + 1.2 * crack + 1.0 * body
    return out / (np.max(np.abs(out)) + 1e-9)


def explosion(dur=4.5):
    n = int(SR * dur)
    t = secs(n)
    f = 24 + 70 * np.exp(-t * 2.5)
    boom = np.tanh(3 * np.sin(2 * np.pi * np.cumsum(f) / SR)) * env_ad(n, 0.003, 1.1)
    blast = fft_filter(noise(n), None, 2500) * env_ad(n, 0.001, 0.35)
    blast = np.tanh(blast * 3)
    crunch = np.zeros(n)
    r = np.random.default_rng(40)
    for _ in range(260):  # brechender Beton (granular)
        st = r.exponential(0.5)
        if st > dur - 0.2:
            continue
        gn = int(SR * r.uniform(0.004, 0.03))
        g = fft_filter(noise(gn), r.uniform(300, 1500), r.uniform(2000, 7000)) * env_ad(gn, 0.0005, 0.006)
        i = int(st * SR)
        crunch[i:i + gn] += g * r.uniform(0.2, 1.0) * np.exp(-st * 1.2)
    tear = metal_hit(3.0, (97, 233, 389, 571, 811, 1290, 1777), seed=4)
    tear = np.pad(tear, (0, n - len(tear)))
    screech = np.sin(2 * np.pi * np.cumsum(900 + 500 * np.sin(2 * np.pi * 3 * t)) / SR) * env_ad(n, 0.01, 0.4) * 0.3
    s = 1.4 * boom + 1.0 * blast + 0.8 * crunch + 0.7 * tear + screech
    return reverb(s / np.max(np.abs(s)), decay=2.2, wet=0.3, seed=9)


def glitch_blip(dur=0.18, seed=0):
    r = np.random.default_rng(seed)
    n = int(SR * dur)
    out = np.zeros(n)
    i = 0
    while i < n:
        seg = int(SR * r.uniform(0.008, 0.03))
        f = r.choice([180, 440, 1250, 2900, 5200, 80])
        tt = secs(seg)
        if r.random() < 0.5:
            s = np.sign(np.sin(2 * np.pi * f * tt))
        else:
            s = np.round(noise(seg) * 2) / 2
        out[i:i + seg] += s[: n - i] * r.uniform(0.3, 1.0)
        i += seg
    return out * 0.6


def static(dur, seed=0, crackle=True):
    r = np.random.default_rng(seed)
    n = int(SR * dur)
    s = fft_filter(r.standard_normal(n), 250, 11000)
    s *= 0.7 + 0.3 * np.sign(np.sin(2 * np.pi * 59.94 * secs(n)))  # Brummen/Zeilen
    if crackle:
        k = r.random(n) < 0.0008
        s += k * r.standard_normal(n) * 6
    return np.clip(s / (np.std(s) * 3.2 + 1e-9), -1, 1)


def drip():
    n = int(SR * 0.4)
    t = secs(n)
    f = 700 + 1400 * (1 - np.exp(-t * 60))
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * env_ad(n, 0.001, 0.03)
    return reverb(s, decay=2.5, wet=0.6, seed=3)


def crinkle(dur, seed=2):
    r = np.random.default_rng(seed)
    n = int(SR * dur)
    s = fft_filter(r.standard_normal(n), 2000, 12000)
    mask = fft_filter((r.random(n) < 0.004).astype(float), None, 400)
    return s * np.clip(mask * 40, 0, 1)


def tinnitus(dur=4.0):
    n = int(SR * dur)
    t = secs(n)
    return np.sin(2 * np.pi * 4150 * t) * np.exp(-t * 0.9) * np.clip(t / 0.05, 0, 1)


# ----------------------------------------------------------------- timeline
def build():
    t = secs(N)

    # ---- Drone (Sub-Bass-Teppich) mit Automation
    drone = (np.sin(2 * np.pi * 36 * t) + 0.8 * np.sin(2 * np.pi * 36.6 * t)
             + 0.4 * np.sin(2 * np.pi * 54.3 * t + 1) + 0.25 * np.sin(2 * np.pi * 72.1 * t))
    rumble = fft_filter(noise(N), 20, 140)
    rumble /= np.max(np.abs(rumble))
    drone = drone / 2.4 * (0.75 + 0.25 * np.sin(2 * np.pi * 0.13 * t)) + 0.6 * rumble
    dg = curve([(0, 0.25), (3.7, 0.3), (4.2, 0.65), (9.8, 0.5), (10, 0.35), (15.4, 0.45),
                (15.5, 0.0), (16.45, 0.0), (16.5, 0.6), (26, 0.6), (35, 0.65), (39.2, 0.75),
                (39.25, 0.0), (40.0, 0.0), (40.01, 0.9), (44, 0.55), (48.3, 0.7),
                (49.9, 0.7), (50.0, 0.0), (65, 0.0)])
    place(drone * dg, 0, 0.35)

    # ---- Szene 1: Polaroid, Atmen, Erstarren, Schluchzen, Bass-Drop
    breath_track(0.0, 3.7, 2.1, 0.55, seed=1)
    place(crinkle(3.6), 0.05, 0.25, 0.2)
    gasp = formant(noise(int(SR * 0.35)), [900, 2200], [800, 1200], [1, 0.6])
    gasp *= np.sin(np.pi * secs(len(gasp)) / 0.35) ** 0.5
    place(gasp / np.max(np.abs(gasp)), 3.72, 0.7)
    sob_track(3.9, 15.5, 0.62, seed=7)  # laeuft bis die Haende fallen
    place(riser(0.5), 3.75, 0.35)
    place(sub_drop(3.5), 4.2, 1.0)
    place(glitch_blip(0.25, 1), 4.2, 0.35)
    breath_track(6.2, 9.9, 1.0, 0.35, seed=2, ragged=0.1)  # flaches, zitterndes Atmen
    place(glitch_blip(0.2, 2), 9.85, 0.5)

    # ---- Szene 2: Rennen, Blick zurueck, Stille, Schrei
    place(static(0.12, 1), 9.95, 0.35)
    breath_track(10.0, 13.4, 2.6, 0.6, seed=3)
    footsteps(10.0, 13.4, 2.6, 2.6, 0.45, 0.45, barefoot=False, seed=6)
    place(glitch_blip(0.2, 3), 11.0, 0.35)
    footsteps(13.4, 15.4, 1.4, 1.0, 0.3, 0.2, barefoot=False, seed=8)
    breath_track(13.5, 15.4, 1.2, 0.25, seed=4)
    # 15.5: Weinen bricht ab -> 1 s Totenstille -> 16.5 Schrei
    scr1 = scream(9.6, seed=3)
    scr1 = reverb(scr1, decay=2.0, wet=0.4, seed=12)[: int(9.6 * SR)]
    sg = curve([(0, 0.55), (1.0, 0.7), (5, 0.8), (9.5, 1.0)], len(scr1))
    place(scr1 * sg, 16.5, 0.85)
    place(sub_drop(2.0, 90, 30), 16.5, 0.6)
    breath_track(16.7, 25.9, 3.0, 0.6, seed=5)
    footsteps(16.8, 25.9, 3.2, 3.6, 0.45, 0.5, barefoot=False, seed=9)
    footsteps(17.4, 25.9, 4.5, 6.5, 0.15, 0.85, barefoot=True, seed=10, pan0=-0.2, pan1=0.0)
    place(static(0.1, 2), 21.95, 0.35)
    place(glitch_blip(0.2, 4), 23.0, 0.35)

    # ---- Szene 3: Tuer, Riegel, Herzschlag, Schrei hinter der Tuer
    place(metal_hit(2.5, seed=13), 26.28, 1.0)
    place(sub_drop(1.2, 80, 35), 26.28, 0.6)
    n = int(SR * 0.55)
    scrape = fft_filter(noise(n), 1500, 5000) * np.sin(np.pi * secs(n) / 0.55) ** 0.5
    place(scrape * 0.5, 26.7)
    place(metal_hit(0.8, (1150, 2310, 3470, 5200), seed=14, bright=1.6), 27.22, 0.8)
    scr2 = scream(9.8, seed=21, pitch=0.97)
    scr2 = fft_filter(scr2, None, 900)  # gedaempft durch Stahl
    scr2 = reverb(scr2, decay=1.2, wet=0.25, seed=22)[: int(9.8 * SR)]
    sg2 = curve([(0, 0.35), (4.0, 0.45), (4.5, 1.0), (9.6, 1.0), (9.8, 0.0)], len(scr2))
    place(scr2 * sg2, 26.4, 1.0, pan=0.0)
    breath_track(26.5, 34.9, 2.4, 0.45, seed=11)
    heartbeat(27.0, 39.2, 120, 165, 0.95)
    footsteps(26.4, 30.5, 6.0, 2.0, 0.6, 0.3, barefoot=True, seed=12)

    # ---- Szene 4: Stille, Explosion
    breath_track(35.0, 38.0, 1.6, 0.3, seed=12, ragged=0.1)
    place(drip(), 37.6, 0.35, 0.3)
    place(explosion(5.0), 40.0, 1.15)
    place(sub_drop(3.5, 120, 24), 40.0, 1.0)
    place(metal_hit(2.5, (130, 340, 610, 980, 1500), seed=17), 40.42, 0.9, pan=-0.75)
    place(tinnitus(4.5), 40.15, 0.12)
    place(glitch_blip(0.2, 5), 40.8, 0.3)
    rumble2 = fft_filter(noise(int(SR * 8)), 20, 90)
    rumble2 = rumble2 / np.max(np.abs(rumble2)) * np.exp(-secs(len(rumble2)) * 0.4)
    place(rumble2, 40.3, 0.7)
    debris = np.zeros(int(SR * 4))
    rr = np.random.default_rng(3)
    for _ in range(40):
        st = rr.uniform(0.3, 3.5)
        h = metal_hit(0.4, (rr.uniform(600, 1500), rr.uniform(1800, 3500)), seed=int(rr.integers(1e5)))
        i = int(st * SR)
        debris[i:i + len(h)] += h[: len(debris) - i] * rr.uniform(0.05, 0.3) * np.exp(-st)
    place(debris, 40.3, 0.6, pan=0.2)
    breath_track(42.0, 47.4, 2.8, 0.4, seed=13)
    place(drip(), 45.4, 0.4, -0.2)

    # ---- Szene 5: Sprung, Griff, Gesicht, Clipping, Rauschen
    scr3 = scream(2.6, seed=31, pitch=1.08)
    place(scr3, 47.6, 0.95)
    place(sub_drop(2.0, 100, 30), 48.3, 0.9)
    final = scream(0.7, seed=41, pitch=1.15)
    place(np.clip(final * 6.0, -1, 1), 49.25, 1.0, bus=CLIP_BUS)  # absichtliches Clipping
    st = static(15.2, 7)
    sgc = curve([(0, 1.0), (4.9, 0.85), (5.0, 0.45), (15.1, 0.35), (15.2, 0.0)], len(st))
    place(st * sgc, 49.85, 0.75)
    place(glitch_blip(0.25, 6), 50.3, 0.4)
    place(glitch_blip(0.25, 7), 55.0, 0.3)
    place(glitch_blip(0.2, 8), 58.0, 0.3)
    place(glitch_blip(0.2, 9), 61.5, 0.3)
    sub = scream(0.12, seed=51)
    place(sub, 63.0, 0.5)  # Unterschwelliges Gesicht


def limiter(x, thr=0.89, block=240):
    nb = len(x) // block + 1
    pad = np.pad(np.abs(x), (0, nb * block - len(x)))
    peaks = pad.reshape(nb, block).max(1)
    peaks = np.maximum(peaks, np.concatenate([[0], peaks[:-1]]))
    peaks = np.maximum(peaks, np.concatenate([peaks[1:], [0]]))
    g = np.minimum(1.0, thr / (peaks + 1e-9))
    # schnelle Attack, langsames Release
    for i in range(1, nb):
        g[i] = min(g[i], g[i - 1] + 0.004)
    gi = np.interp(np.arange(len(x)), np.arange(nb) * block + block / 2, g)
    return x * gi


def main():
    build()
    # Harte Totenstille (auch Hall-Fahnen werden abgeschnitten)
    gate = curve([(0, 1), (15.48, 1), (15.52, 0), (16.48, 0), (16.5, 1),
                  (39.22, 1), (39.27, 0), (39.99, 0), (40.0, 1), (65, 1)])
    L[:] *= gate
    R[:] *= gate
    st = np.stack([L, R], 1)
    st /= np.percentile(np.abs(st), 99.95) + 1e-9
    st *= 1.25
    st[:, 0] = limiter(st[:, 0])
    st[:, 1] = limiter(st[:, 1])
    gmax = np.max(np.abs(st))
    st = st / max(gmax, 1e-9) * 0.89
    st += CLIP_BUS[:, None]
    st = np.clip(st, -1.0, 1.0)
    pcm = (st * 32767).astype(np.int16)
    with wave.open("audio.wav", "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print("audio.wav geschrieben", pcm.shape)


if __name__ == "__main__":
    main()
