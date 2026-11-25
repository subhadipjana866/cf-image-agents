"""
typography_ocr.py
----------------------------------------------------------------------
OPTIONAL companion to typography.py. Needs EasyOCR (and therefore PyTorch),
which is why it is a separate module: the main path -- typeset before
generation, register after -- needs no OCR at all.

Used for exactly two things:
  match_user_font_image(path)   nearest Kognie font to the lettering in a user's
                                own image (role "font"). Approximate by design:
                                5/7 correct on posters with known fonts; rounded
                                sans and brush scripts are the weak spots.
  measure_element(img, element) where registration cannot place an element (it
                                moved > ~120 px, or user-font-image mode), read its
                                real geometry from the pixels: line text, x,
                                baseline, font size (vertical-profile fit of the real
                                font), letter-spacing, colour, rotation. Validated
                                on ground truth: size 2.1%, baseline/x 0.3 px.

If EasyOCR is not installed, `available()` is False and the pipeline marks such
elements "unverified" instead of failing.
"""

from __future__ import annotations

import difflib
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw

import typography as ty

_reader = None
_ocr_cache: dict = {}


def available() -> bool:
    try:
        import easyocr  # noqa: F401
        return True
    except Exception:
        return False


def ocr(img: np.ndarray):
    global _reader
    key = (img.shape, int(img[::97, ::89].sum()))
    if key in _ocr_cache:
        return _ocr_cache[key]
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    res = _reader.readtext(img, paragraph=False)
    _ocr_cache.clear()
    _ocr_cache[key] = res
    return res


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", s.lower()).strip()


def best_span(det: str, lines: list[str]) -> tuple[float, str, int]:
    t = norm(det)
    best = (0.0, "", -1)
    for li, line in enumerate(lines):
        words = line.split()
        for i in range(len(words)):
            for j in range(i + 1, len(words) + 1):
                span = " ".join(words[i:j])
                r = difflib.SequenceMatcher(None, t, norm(span)).ratio()
                if r > best[0]:
                    best = (r, span, li)
    return best


# ---- masks ------------------------------------------------------------------

def main_band(m: np.ndarray) -> np.ndarray:
    """Keep only the dominant text line (OCR boxes clip neighbouring lines)."""
    prof = m.sum(1)
    if prof.max() == 0:
        return m
    on = prof >= 0.08 * prof.max()
    bands, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            bands.append((prof[i:j].sum(), i, j))
            i = j
        else:
            i += 1
    _, b0, b1 = max(bands)
    for _s, s0, s1 in bands:
        if s1 <= b0 and b0 - s1 <= 0.15 * (b1 - b0):
            b0 = s0
        if s0 >= b1 and s0 - b1 <= 0.15 * (b1 - b0):
            b1 = s1
    n, cc, _st, cent = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    out = np.zeros_like(m)
    for k in range(1, n):
        if b0 <= cent[k][1] < b1:
            out |= cc == k
    return out


def ink_mask(crop: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    _, labels, _ = cv2.kmeans(lab, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                              3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(crop.shape[:2])
    border = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    m = labels != np.bincount(border, minlength=2).argmax()
    n, cc, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= max(4, 0.0006 * m.size):
            keep |= cc == i
    return main_band(keep)


def crop_to_ink(m):
    ys, xs = np.where(m)
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if len(ys) else m


def text_angle(crop: np.ndarray) -> float:
    m = ink_mask(crop)
    pts = np.column_stack(np.where(m))[:, ::-1].astype(np.float32)
    if len(pts) < 20:
        return 0.0
    (_, _), (w, h), ang = cv2.minAreaRect(pts)
    if w < h:
        ang -= 90
    return float(((ang + 45) % 90) - 45)


# ---- user font image -> nearest Kognie font ------------------------------------

def _normalise(m, H=48):
    if m.size == 0 or not m.any():
        return np.zeros((H, 1), bool)
    cols = m.any(0)
    runs, i = [], 0
    while i < len(cols):
        if cols[i]:
            j = i
            while j < len(cols) and cols[j]:
                j += 1
            runs.append(m[:, i:j])
            i = j
        else:
            i += 1
    gap = np.zeros((m.shape[0], max(1, int(0.12 * m.shape[0]))), bool)
    parts = []
    for k, r in enumerate(runs):
        if k:
            parts.append(gap)
        parts.append(r)
    m2 = np.concatenate(parts, 1)
    w = max(1, int(round(m2.shape[1] * H / m2.shape[0])))
    return cv2.resize(m2.astype(np.uint8) * 255, (w, H), interpolation=cv2.INTER_AREA) > 100


def _render(text, f):
    l, t, r, b = f.getbbox(text)
    im = Image.new("L", (max(4, r - l + 20), max(4, b - t + 20)), 0)
    ImageDraw.Draw(im).text((10 - l, 10 - t), text, font=f, fill=255)
    return crop_to_ink(np.array(im) > 127)


def _degrade(m, h):
    if h >= m.shape[0]:
        return m
    w = max(1, int(round(m.shape[1] * h / m.shape[0])))
    g = cv2.GaussianBlur(cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA), (0, 0), 0.6)
    return crop_to_ink(g > 0.45)


def _distance(a, b):
    H = a.shape[0]
    bw = cv2.resize(b.astype(np.uint8) * 255, (a.shape[1], H), interpolation=cv2.INTER_AREA) > 100
    if not a.any() or not bw.any():
        return 9.0
    dta = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, 3)
    dtb = cv2.distanceTransform((~bw).astype(np.uint8), cv2.DIST_L2, 3)
    chamfer = (dtb[a].mean() + dta[bw].mean()) / 2 / H
    return float(chamfer + 0.35 * abs(np.log(a.shape[1] / b.shape[1])) + 0.5 * abs(a.mean() - bw.mean()))


def match_user_font_image(path) -> dict:
    img = np.array(Image.open(path).convert("RGB"))
    runs = []
    for pts, txt, conf in ocr(img):
        t = txt.strip()
        if conf < 0.5 or len(re.sub(r"[^A-Za-z]", "", t)) < 3:
            continue
        pts = np.array(pts)
        x0, y0 = pts.min(0).astype(int)
        x1, y1 = pts.max(0).astype(int)
        runs.append((y1 - y0, t, (x0, y0, x1, y1), float(conf)))
    if not runs:
        return {"nearest": None, "confidence": None, "approximate": True,
                "reason": "no readable lettering in the image"}
    votes, detail = {}, []
    for h, t, (x0, y0, x1, y1), conf in sorted(runs, reverse=True)[:5]:
        pad = max(2, int(0.1 * h))
        mask = crop_to_ink(ink_mask(img[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]))
        if mask.shape[0] < 8:
            continue
        a = _normalise(mask)
        d = {k: min(_distance(a, _normalise(_degrade(_render(v, ty.load_font(k, 96)), mask.shape[0])))
                    for v in {t, t.upper()}) for k in ty.FONTS}
        ranked = sorted(d, key=d.get)
        detail.append({"text": t, "ranked": ranked[:3]})
        for i, k in enumerate(ranked[:3]):
            votes[k] = votes.get(k, 0) + h * conf * (3 - i)
        if len(detail) == 1:
            s = sorted(d.values())
            first_conf = round(float(1 - s[0] / s[1]), 3) if s[1] else None
    if not detail:
        return {"nearest": None, "confidence": None, "approximate": True, "reason": "lettering too small"}
    ranked = sorted(votes, key=votes.get, reverse=True)
    return {"nearest": ranked[0], "runner_up": ranked[1] if len(ranked) > 1 else None,
            "confidence": first_conf, "approximate": True, "runs": detail}


# ---- measuring an element from pixels -------------------------------------------

def _stroke_width(mask):
    if not mask.any():
        return 0.0
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    ridge = dt[(dt >= cv2.dilate(dt, np.ones((3, 3)))) & mask]
    return float(2 * np.median(ridge)) if len(ridge) else 0.0


def _profile_fit(R, M, k0):
    Mn = M / (np.linalg.norm(M) + 1e-9)
    best = (-1.0, k0, 0)
    for k in k0 * np.linspace(0.7, 1.3, 61):
        L = max(2, int(round(len(R) * k)))
        Rk = np.interp(np.linspace(0, len(R) - 1, L), np.arange(len(R)), R)
        F = len(M) + 2 * L
        Mp = np.zeros(F)
        Mp[L:L + len(M)] = Mn
        for off in range(-L // 2, len(M) - L // 2 + 1):
            Rp = np.zeros(F)
            Rp[L + off:L + off + L] = Rk
            sc = float(Mp @ Rp) / (np.linalg.norm(Rp) + 1e-9)
            if sc > best[0]:
                best = (sc, L / len(R), off)
    return best[1], best[2]


def _fit_line(key, weight, text, ink, mask):
    """font size (vertical-profile fit), letter-spacing, origin, baseline."""
    x0, y0, x1, y1 = ink
    ref = 200
    f = ty.load_font(key, ref, weight)
    l, t, r, b = f.getbbox(text, anchor="ls")
    pad = 8
    im = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 0)
    ImageDraw.Draw(im).text((pad - l, pad - t), text, font=f, fill=255, anchor="ls")
    rm = np.array(im) > 96
    ox, oy = pad - l, pad - t
    ys, xs = np.where(rm)
    if not len(ys) or not mask.any():
        return None
    k, off = _profile_fit(rm[ys.min():ys.max() + 1].sum(1).astype(float), mask.sum(1).astype(float),
                          mask.shape[0] / (ys.max() + 1 - ys.min()))
    size = ref * k
    ls = ((x1 - x0) - (xs.max() + 1 - xs.min()) * k) / max(1, len(text) - 1)
    return {"font_size_px": round(size, 2), "letter_spacing_px": round(ls, 2),
            "x": round(x0 - (xs.min() - ox) * k, 1), "baseline_y": round(y0 + off + (oy - ys.min()) * k, 1)}


def measure_element(img: np.ndarray, e: dict) -> dict | None:
    """Re-read one element from the pixels. Returns a planned-shaped dict or None."""
    p = e["planned"]
    target = ty.apply_transform(e["text"], p["text_transform"])
    runs = []
    for pts, txt, conf in ocr(img):
        ratio, span, _ = best_span(txt, [target])
        if conf < 0.1 or ratio < 0.6 or len(norm(span).replace(" ", "")) < 2:
            continue
        pts = np.array(pts)
        runs.append({"ocr": txt, "conf": float(conf), "ratio": ratio,
                     "box": [int(v) for v in (*pts.min(0), *pts.max(0))]})
    anchors = [r for r in runs if len(norm(r["ocr"]).replace(" ", "")) >= 3 and r["ratio"] >= 0.8]
    if not anchors:
        return None
    hm = float(np.median([r["box"][3] - r["box"][1] for r in anchors]))
    runs = [r for r in runs if (r["box"][3] - r["box"][1]) <= 2.2 * hm and (r in anchors or any(
        min(r["box"][3], a["box"][3]) - max(r["box"][1], a["box"][1]) > 0.5 * min(r["box"][3] - r["box"][1],
                                                                                    a["box"][3] - a["box"][1])
        for a in anchors))]
    lines = []
    for r in sorted(runs, key=lambda r: (r["box"][1] + r["box"][3]) / 2):
        for ln in lines:
            ly0, ly1 = min(x["box"][1] for x in ln), max(x["box"][3] for x in ln)
            if min(r["box"][3], ly1) - max(r["box"][1], ly0) > 0.5 * min(r["box"][3] - r["box"][1], ly1 - ly0):
                ln.append(r)
                break
        else:
            lines.append([r])
    H, W = img.shape[:2]
    key, weight = e["font"]["key"], e["font"]["weight"]
    out_lines, sizes, spacings, colours, angles = [], [], [], [], []
    for ln in lines:
        ln = sorted(ln, key=lambda r: r["box"][0])
        x0, y0 = min(r["box"][0] for r in ln), min(r["box"][1] for r in ln)
        x1, y1 = max(r["box"][2] for r in ln), max(r["box"][3] for r in ln)
        pad = max(3, int(0.12 * (y1 - y0)))
        cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
        crop = img[cy0:min(H, y1 + pad), cx0:min(W, x1 + pad)]
        if min(crop.shape[:2]) < 10:
            continue
        m = ink_mask(crop)
        ys, xs = np.where(m)
        if len(ys) < 20:
            continue
        ink = (cx0 + xs.min(), cy0 + ys.min(), cx0 + xs.max() + 1, cy0 + ys.max() + 1)
        _, span, _ = best_span(" ".join(r["ocr"] for r in ln), [target])
        fit = _fit_line(key, weight, span, ink, m[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
        if not fit:
            continue
        bg = np.median(crop[~m], 0)
        px = crop[m].astype(float)
        dist = np.linalg.norm(px - bg, axis=1)
        px = px[dist >= np.percentile(dist, 60)] if len(px) > 50 else px
        colours.append(np.median(px, 0))
        sizes.append(fit["font_size_px"])
        spacings.append(fit["letter_spacing_px"])
        angles.append(text_angle(crop))
        out_lines.append({"text": span, "x": fit["x"], "baseline_y": fit["baseline_y"],
                          "advance_width": float(ink[2] - ink[0]), "ink_box": [float(v) for v in ink]})
    if not out_lines:
        return None
    xs0 = [l["ink_box"][0] for l in out_lines]
    xs1 = [l["ink_box"][2] for l in out_lines]
    cx = [(a + b) / 2 for a, b in zip(xs0, xs1)]
    tol = 0.015 * W
    align = ("center" if len(out_lines) > 1 and np.ptp(cx) <= tol else
             "left" if len(out_lines) > 1 and np.ptp(xs0) <= tol else
             "right" if len(out_lines) > 1 and np.ptp(xs1) <= tol else p["align"])
    size = float(np.median(sizes))
    bases = [l["baseline_y"] for l in out_lines]
    lh = float(np.median(np.diff(bases))) / size if len(bases) > 1 else p["line_height"]
    box = [min(xs0), min(l["ink_box"][1] for l in out_lines), max(xs1), max(l["ink_box"][3] for l in out_lines)]
    return {"font_size_px": round(size, 2), "letter_spacing_px": round(float(np.median(spacings)), 2),
            "line_height": round(lh, 3), "align": align, "text_transform": p["text_transform"],
            "color_hex": ty.rgb2hex(np.median(colours, 0)), "rotation_deg": round(float(np.median(angles)), 1),
            "box": {"x": box[0], "y": box[1], "w": box[2] - box[0], "h": box[3] - box[1]},
            "lines": out_lines, "overflow": False, "min_font_px": p.get("min_font_px"),
            "legibility_met": size >= (p.get("min_font_px") or 0) - 0.5}
