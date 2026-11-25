# Typography in the Kognie generation pipeline (v3.3)

**What it does.** Every graphic's text is set in **real fonts**, meets **WCAG AA contrast** and
**platform-specific legibility**, and ships with **`text_layers`**: the metadata an HTML canvas needs to
recreate each text element as an editable object instead of baked-in pixels.

**Why it is built this way.** This is the short version of a month of experiments (`experiments/font_adherence/`):

| Approach | Result |
|---|---|
| Name a font in the prompt | 0/4 lines in the requested font (GPT Image 2.5); models draw a generic look-alike |
| Attach an image of the font's alphabet | 24/28 lines in the requested font; 7/8 with 2-3 fonts per graphic |
| Attach a **typeset layout image** (copy already set in the real fonts) | text lands within **0-1 px** of plan on 2:3; 10-70 px drift on other canvases, so we **register** afterwards |
| Ask a vision LLM which font it is | 42% correct, so never use one to verify fonts |

---

## 1. The flow

```
GROUND        extract facts + locked text + NAMED FONTS (font_requests)
  |
SUBJECT
  |
  +-- TYPOGRAPHER   picks REAL fonts per role from the Kognie library  ]
  +-- ART DIRECTOR                                                     ] parallel
  +-- COMPOSITOR                                                       ]
  |
TYPESET  (new)      1 reasoning call: exact copy, frames, sizes, colours
                    + deterministic enforcement (typography.enforce):
                      font rules · named/unsupported fonts · user font image ·
                      character set · legibility floors · WCAG AA · typesetting
                    -> pre-generation metadata + layout image
  |
DIRECTOR            told the typography is pre-set: composes the artwork AROUND it,
                    keeps the text areas calm, does not restate text or typefaces
  |
RENDER              images = [layout image, font samples (or the user's font image),
                              client references...]   model: gpt-image-2.5-flare
  |
TEXT QA  (new)      per element: registration (+/-120 px, scale 0.85-1.15, ~0.1 s, no OCR)
                      confirmed | corrected (shift/scale applied) | measured (OCR) | unverified
                    + WCAG contrast against the artwork actually painted behind the text
  |
result["text_layers"]   (schema kognie.text_layers.v2), also written as <stem>.text_layers.json
```

**No text → no typography.** If GROUND finds nothing that must appear, TYPESET is skipped, no layout image
is sent, and the request runs exactly as in v3.2 (`text_layers` absent).

**Failure is never fatal.** If the TYPESET call fails, the request falls back to v3.2 behaviour
(model-drawn text, no `text_layers`) and emits `node_failed`.

`create_more()` re-runs TYPESET for the variation's new composition. Copy, fonts and colours carry over;
placement follows the new layout. It writes its own `text_layers`.

---

## 2. Product rules (enforced in code, not left to the LLM)

| Rule | Where |
|---|---|
| Text is optional; zero elements is valid | GROUND / TYPESET prompt; v5.4 copywriter prompt updated too |
| Fonts per graphic: **default 2, max 3** | `enforce()` merges extra families |
| **Title always has its own font**; subheading/body/CTA/caption may share one | `enforce()` reassigns sharers |
| User names a **supported** font → used for the role they named (title if unclear); its sample image is the reference | GROUND `font_requests` + alias table (`pasifico` → Pacifico, `monstreat` → Montserrat) |
| User names an **unsupported** font → **notice** + closest supported style | `result["typography"]["user_notices"]`, e.g. *"Helvetica Neue" isn't available on Kognie yet. We used the closest supported style instead: Lexend and Montserrat.* |
| User attaches their own image for the font style (**reference role `"font"`**) → that image is the **only** font reference (no Kognie sample attached). Metadata uses the **nearest Kognie font**, flagged `approximate_font` | `typeset_render_inputs()`, `typography_ocr.match_user_font_image()` |
| Copy uses only characters on the font sheets (A–Z a–z 0–9 `. , !`); exotic symbols are replaced, `? ' - : & % "` are allowed but flagged | `sanitize()` |
| Decorative fonts (script, hand, display) only for titles and short lines (≤ 12 words) | `enforce()` |
| **WCAG 2.x AA**: 4.5:1, or 3:1 for large text (≥ 24 CSS px, or ≥ 18.66 px bold, **at display size**). Failing colours are fixed before generation (lightness moved, hue kept) and checked again against the rendered artwork | `fix_contrast()`, `post_contrast()` |
| **Legibility is platform-specific**: minimum size per role from each platform's display width, corrected for x-height and light weights; text is never typeset below it (the frame grows instead) | `PLATFORMS`, `min_font_px()`, `typeset()` |

### Platform table (`typography.PLATFORMS`, editable)

| Platform | Display width (CSS px) | Min body / title (CSS px) | Text-free margins (top, right, bottom, left) |
|---|---|---|---|
| Instagram Square / Portrait Post | 390 | 12 / 20 | 5% all round |
| Instagram Story | 390 | 12 / 20 | 14%, 6%, 20%, 6% (app UI) |
| Facebook Feed Landscape | 390 | 12 / 20 | 5%, 4%, 5%, 4% |
| Pinterest Pin | **236** (grid column) | 12 / 20 | 5%, 6%, 10%, 6% |
| LinkedIn Feed Post | 390 | 12 / 20 | 5% |
| YouTube Thumbnail | **246**, max 6 words | 12 / 20 | 5%, 5%, 14%, 5% (timestamp) |
| Presentation Slide | 960 | **32 / 48** (24 pt body) | 7%, 7%, 10%, 7% |
| Digital Signage Portrait | 405 (55" at ~15 ft) | **16 / 28** | 8% |
| X (Twitter) In-Stream Post | 360 | 12 / 20 | 5%, 4%, 5%, 4% |

---

## 3. API changes (backwards compatible)

`run()` and `create_more()` keep their signatures.

**New reference role `"font"`**:
```python
pipeline.run(raw_prompt="... take the lettering style from my image", goal=..., platform=...,
             reference_images=[("user_lettering.png", "font"), ("logo.png", "logo")])
```

**New result keys**

| Key | Meaning |
|---|---|
| `result["typography"]` | `fonts_used`, `user_notices` (show these to the user), `policy_log` (every automatic fix), `user_font_image` (match result) |
| `result["text_layers"]` | the editable-text metadata (below). Absent when the graphic has no text |
| `result["layout_image"]` | the typeset layout PNG sent to the renderer (debugging) |

**New events**: `typeset_done` (elements, fonts, notices) and `text_qa_done` (per-element statuses,
`contrast_fail` ids). The `node_start` / `node_done` events now also include `"typeset"`.

---

## 4. `text_layers` (schema `kognie.text_layers.v2`)

```jsonc
{
  "schema": "kognie.text_layers.v2", "image": "event_announcement__..png",
  "canvas": {"width": 1024, "height": 1280, "units": "px", "origin": "top-left"},
  "fonts_used": ["Pacifico", "Montserrat"],
  "user_notices": [], "policy_log": ["date: 38px below the ... minimum -> raised", "..."],
  "elements": [{
    "id": "title", "role": "title", "text": "Summer Jazz Night", "text_transform": "none",
    "font_family": "Pacifico", "css_font_family": "'Pacifico', cursive", "font_weight": 400,
    "font_style": "normal", "font_key": "pacifico",
    "font_metrics": {"units": "em", "ascent": 1.303, "descent": 0.374, "cap_height": 0.695, "x_height": 0.49, ...},
    "font_size_px": 58.39, "letter_spacing_px": 0.0, "line_height": 1.1, "align": "left",
    "color_hex": "#fff6e5", "rotation_deg": 0,
    "lines": [{"text": "Summer Jazz", "x": 81.0, "baseline_y": 176.3, "advance_width": 311.2, "ink_box": [..]},
              {"text": "Night", "x": 81.0, "baseline_y": 240.5, ...}],
    "box": {"x": 81, "y": 131, "w": 314, "h": 128}, "box_norm": {"x": 0.079, "y": 0.102, ...},
    "contrast": {"plan": {"ratio": 16.9, "target": 3.0, "level": "AA large", ...},
                 "after_render": {"ratio": 16.47, "pass": true, ...}},
    "qa": {"status": "confirmed", "source": "planned", "registration": {"dx": 1, "dy": 0, "scale": 1.0, "score": 0.96}},
    "approximate_font": false
  }]
}
```

**Recomposing.** For each line, call `ctx.fillText(text, x, baseline_y)` with
`ctx.font = "{font_style} {font_weight} {font_size_px}px {css_font_family}"` and
`ctx.letterSpacing = letter_spacing_px + "px"`. Apply `text_transform` at render time; `text` keeps the
natural case for editing. For DOM text boxes, use `top = baseline_y − font_metrics.ascent × font_size_px`.
**Keep the stored line breaks**; only re-wrap when the user edits the text.
`examples/recompose_canvas.html` does both (canvas, and editable DOM text).

**Use the same font files and the same text engine as the editor.** The typesetter uses Pillow. A browser
canvas lays text out with fractional advances and full kerning, so long lines can differ by about 1% in
width. If the editor uses Fabric.js, Konva or skia, swapping `typography.typeset()` for that engine's
measurements removes the gap (proven in `font_adherence/run_textmeta/recompose/typeset.html`: 10/12 lines
within 1 px).

### QA statuses: what the editor should do

| `qa.status` | Meaning | Editor action |
|---|---|---|
| `confirmed` | text is where the metadata says (≤ 2 px, same size) | use as-is |
| `corrected` | the model moved or scaled the block; the metadata was shifted/scaled to match | use as-is |
| `measured` | registration couldn't place it; geometry re-read from pixels (OCR) | use; slightly less exact |
| `unverified` | couldn't confirm (OCR not installed, or text missing) | editable, but may not line up with the baked text; flag for review or regenerate |

`contrast.after_render.pass == false` means the artwork behind the text is too busy or too close in tone.
Add a scrim behind the editable text, or regenerate.

---

## 5. Configuration and dependencies

| Variable | Default | |
|---|---|---|
| `KOGNIE_TYPOGRAPHY` | `on` | `off` restores v3.2 behaviour exactly |
| `KOGNIE_TYPOGRAPHY_ASSETS` | `./typography_assets` | fonts + samples |
| `ARMC3_IMAGE_MODEL` | `gpt-image-2.5-flare` | **changed in v3.3** (was `gpt-image-2`). Flare is the model the typography results were measured on, and it's about 2× faster than Sunburst at equal font fidelity |

- **Required:** `numpy`, `opencv-python-headless` (registration, contrast), `Pillow`.
- **Optional:** `easyocr` (pulls in PyTorch). It's only needed for user-font-image matching and for measuring
  elements registration can't place. Without it those elements are `unverified`, never a crash.
- **Courier New is not redistributable** and isn't shipped. It's loaded from the OS (Windows/macOS have it;
  on Linux install `ttf-mscorefonts-installer`), or drop `cour.ttf` into `typography_assets/fonts/courier_new/`.
  For the web editor, consider *Courier Prime* (OFL) as the embeddable equivalent.
- The other six fonts are SIL OFL; each folder carries its `OFL.txt`, which must ship with the font.

**Cost and latency per graphic with text:** +1 reasoning call (TYPESET, thinking=medium, about 10–25 s) and
+0.1–2 s of QA (OCR fallback adds about 5–15 s on CPU when used).

---

## 6. Adding a font to the Kognie library

1. Put the TTF/OTF in `typography_assets/fonts/<key>/`, with its licence.
2. Put an alphabet sample image in `typography_assets/samples/<key>.png`: A–Z, a–z, 0–9 and `. ! ,` in clean
   rows on white. This is what the renderer copies letterforms from; bigger and cleaner is better.
3. Add an entry to `typography.FONTS`: family, weight (one weight per entry), file, generic CSS family,
   category, `readability` (`text` or `display`), aliases (including common misspellings) and `caps_only`
   if it has no lowercase.
4. Run `python typography_smoke_test.py E1`, then check the rendered text against the sample.

---

## 7. Known limits

- **Text-block drift on non-2:3 canvases.** The model moved blocks 10–70 px on 5:4/4:5 in testing.
  Registration corrects shifts up to ±120 px and ±15% scale; beyond that the element is measured (OCR) or
  `unverified`. An open experiment: render at a standard size (1024×1024, 1536×1024 or 1024×1536), then
  crop or extend to the platform shape.
- **User-font-image matching is approximate.** It was right on 5 of 7 posters with known fonts; rounded
  sans (Comfortaa vs Lexend) and brush scripts are the weak spots. Low confidence (< 0.5) triggers a
  user notice.
- **`create_more` prompt precedence.** The variation node writes its prompt before the text is re-set, so
  the renderer is told that "any wording in the description is the text already in image 1". Watch for
  doubled text in variations.
- **Gemini fallback renderer.** If gpt-image-2.5-flare fails, Nano Banana renders at its own size. The
  metadata is rescaled to the output size, and registration still runs.
- **Weight drift.** Light fonts sometimes render bolder. The metadata keeps the specified weight, which is
  the design intent.

---

## 8. Testing

```bash
python typography_smoke_test.py           # E1-E5: supported font, unsupported font, no text, user font image, create_more
python typography_smoke_test.py E1 E3     # subset
```
Each case writes the PNG, the layout PNG, `*.text_layers.json`, and the full result JSON with events under
`out/typography_smoke/`.

**Results at release** (2026-11-17, gpt-image-2.5-flare, quality medium): see `TEST_RESULTS` below.

| Case | Goal / platform | Outcome |
|---|---|---|
| **E1** named supported font ("use Pacifico for the title") | Event Announcement / Instagram Portrait | Pacifico title + Montserrat (2 fonts). 5 sizes raised to the platform floor. QA: 2 confirmed, 3 corrected (3 px shift). Contrast after render 11–19:1. 156 s, $0.13 |
| **E2** named unsupported font ("Helvetica Neue") | Product Launch / LinkedIn | Notice shown; Lexend title + Montserrat. A planned overlap was auto-resolved (block moved 12 px). QA: 2 corrected, 1 measured. Contrast 6.4–15:1. 188 s |
| **E3** "no text at all" | Brand Awareness / Instagram Story | No typography: no TYPESET, no layout image, plain v3.2 path. 95 s, $0.10 |
| **E4** user's own lettering image (role `font`) | Grand Opening / Instagram Square | Matched to Amatic SC (confidence 0.67). The user image was the only font reference. 5/5 elements measured via OCR. Contrast 6.7–16:1. 170 s |
| **E5** `create_more` on E1 | → `hero_panel\|right` | Re-typeset for the new layout: **5/5 confirmed at 0–2 px**. Contrast 11–18:1. 78 s, $0.085 |

Two defects were found and fixed by this run, and both are covered by the code above:
- **`create_more` described text differently from the layout image.** Its first run had elements moved
  30–67 px and shrunk about 10%; it is now told the typography is set after it.
- **A planned text overlap was flagged but not fixed.** `enforce()` now pushes the lower block down.
