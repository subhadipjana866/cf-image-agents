# Kognie generation pipeline v3.2 — integration guide

Base generation model: **gpt-image-2**. Two entry points: `run()` for a first
graphic, `create_more()` for each "show me another" click on top of it.

Everything in `kognie-generation-v32.zip` is copied verbatim from the tested
source or transformed by a named rule in `make_package_v32.py`. Nothing is
hand-copied, because hand-copying is how v3.1 ended up shipping a 4-option
composition menu while the experiments were still reading a 3-option one.

---

## 1. Quick start

```bash
unzip kognie-generation-v32.zip && cd kognie-generation-v32
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                              # fill in the two keys
python example_run.py
```

`example_run.py` generates one graphic, then runs one create-more on it. If you
get two PNGs in `./out`, the install is good.

### Credentials

| Variable | Required | Used for |
|---|---|---|
| `GEMINI_API_KEY` | yes | the six reasoning calls, and the create-more call |
| `OPENAI_API_KEY` | yes | gpt-image-2, the base renderer |

Both are read from the environment at call time. Nothing is written to disk and
nothing is baked into the package.

Without `OPENAI_API_KEY` the pipeline still compiles a prompt but **every render
silently falls through to the fallback renderer**, and the output is not what was
tested. Treat a missing key as a misconfiguration, not a degraded mode — assert
on `result["render"]["rendered_by"]` if you want to catch it.

---

## 2. What changed since v3.1

| | Why |
|---|---|
| **Hard timeouts** | A reasoning call hung for 15 minutes on 2.25s of CPU. `CALL_TIMEOUT_S` (180s) is enforced twice — at the SDK's HTTP layer and again as a thread ceiling — so a wedged connection degrades into the fallback-model retry instead of a stuck worker. |
| **Supplied photos are integrated, not pasted** | A hero image came back as a gold-bordered card on a separately generated scene. The hero rules spoke in layout vocabulary ("framing", "placement", "crop") while the product rules spoke in world vocabulary — and the product one worked. Both now say the canvas is one continuous photograph. |
| **Actions render whole, with hierarchy** | `HARVEST30` rendered in a bare box with no "use code". The Director had read "reproduce locked strings exactly" as forbidding it from adding words around one. It is now told the locked string is a *substring* of the line it renders — and that the token outranks its instruction words. |
| **Attributed quotations get drawn quote marks** | The first fix failed because "set it in quotation marks" was read as the prompt's own delimiters. The rule now says the marks are glyphs the viewer sees. |
| **Branded packaging when a logo is supplied** | Blank surfaces exist to stop the pipeline *inventing* a wordmark. Once the client supplies theirs, dressing the scene with it is correct. Conditional on `has_logo`; invented label copy (origins, specs, tasting notes) stays banned. |
| **`other_stated_facts`** | `locked_text` is keyed by schema `field_id`, so a fact with no field — "live set from the Bootham Brass Band at noon" — was dropped at extraction. There is now a bucket for it. |
| **One source of layout data** | `layout_guidance.json`, read once at import. See §6. |
| **`create_more()`** | New. See §5. |

---

## 3. The two calls

### First generation

```python
import generation_pipeline as pipeline

result = pipeline.run(
    raw_prompt="...",                    # the user's brief, free text
    goal="Flash Sale",                   # must match a known design goal
    platform="Instagram Story",          # must match a known platform
    reference_images=[                   # optional, max 5 — see §4
        {"path": "logo.png",    "role": "logo"},
        {"path": "product.jpg", "role": "product"},
    ],
    generate_image=True,                 # False = compile the prompt only
    out_dir=Path("out"),                 # PNG + JSON written here
    on_event=None,                       # optional live progress callback
    cta_requested=None,                  # None = detect from raw_prompt
    copywriter=None,                     # optional, added 29 Oct 2026: copywriter.make_hook(...)
)
```

`copywriter`: on by default since 6 Nov 2026. It runs beside SUBJECT and its copy
deck replaces GROUND's locked text and other facts for every node after it. `None`
builds the default one from `raw_prompt`; the app passes
`v32_bridge.copywriter_hook(...)`, which gives it the user's own words, their field
answers and the brand profile. `False` or `ARMC3_COPYWRITER=0` turns it off. If it
fails, or runs `ARMC3_COPY_TIMEOUT_S` (60 s) past SUBJECT, the run keeps GROUND's text.

`pipeline.combos()` returns all 170 servable `(goal, platform)` pairs;
`goals()` and `platforms()` return each axis. An unknown pair raises
`BriefNotFound`.

### What comes back

```python
result["status"]          # "ok"  |  "needs_input"
result["image"]           # path to the PNG, absent if the render failed
result["final_prompt"]    # the compiled prompt the renderer was given
result["negative_prompt"] # defect list, appended to the prompt as prose
result["aspect_ratio"]    # from the guidance, e.g. "9:16"
result["subject"]         # what the pipeline decided the graphic depicts
result["ground"]          # extracted facts: locked_text, other_stated_facts, redemption
result["render"]          # {ok, rendered_by, size, quality, attempts, routing}
result["wall_s"], result["cost_total"], result["events"]
result["copy"]            # copywriter output (deck, problems, attempts, ...), None without one
result["locked_text"]     # only when a copy deck was used: the text the graphic carries
result["typography"]      # fonts_used, user_notices (show these), policy_log; None without text
result["text_layers"]     # editable-text metadata (kognie.text_layers.v2); absent without text
result["layout_image"]    # the typeset layout PNG sent to the renderer as image 1
result["antislop"]        # SLOP GUARD: every conditional decision for this graphic: active and
                          # exempted rules, in_frame, body_parts, drawn, text_restored,
                          # negatives {gated_out, final, kept, tailor_chose, tailor_left_out},
                          # positives {offered, chosen}
```

### SLOP GUARD (added 2 Nov 2026)

`antislop.py` adds rules against the defaults that make a graphic read as AI slop, before
anything is rendered. These are glow and decorative gradients, WordArt, sparkles and
flares, filler ornaments, invented scene writing, plastic skin, haze, clutter and
catalogue people.

- **No model call.** The rules are built in code from the brief in about 1–2 ms. They
  go to the TYPOGRAPHER, ART DIRECTOR, COMPOSITOR and DIRECTOR, and to `create_more()`.
  Their negative terms are appended to the Director's list in code.
- **The brief outranks the guard.** A rule steps aside when the user or their chosen
  style asks for that thing ("neon rim light", "confetti", "gold foil lettering"). The
  check ignores a style's `Avoid:` line and phrases like "no gradients".
- **On by default.** `ARMC3_ANTISLOP=0` turns it off, and the prompts are then the same
  as before.
- **Short prompt (added 4 Nov 2026).** With the guard on, the Director writes labelled
  lines (Scene, Camera, Light, Text, Layout, Style) instead of a 340-520 word paragraph.
  The negative list is kept, except the Director's whole-sentence clichés.
  `ARMC3_SHORT_PROMPT=0` goes back to the long paragraph.
- **Who is in the frame.** SUBJECT proposes `in_frame` (`none`, `part` or `people`) and the
  ART DIRECTOR may change it. A person is allowed, never required. The matching line and
  negatives are added in code. A supplied hero or product image skips this. The result
  records it in `result["antislop"]["in_frame"]`.
- **Anatomy (added 6 Nov 2026).** The specialists also list `body_parts`, and the code reads
  the Scene line as well. Only the parts in this image get a line (hands, feet, mouth, face,
  ears, legs, arms; at most `MAX_PARTS`, 3) and matching negatives. SUBJECT and the ART
  DIRECTOR are steered to views generators draw well. A stylised figure gets one line for
  the whole body, not the photographic ones.
- **The medium is open.** The image can be a photograph, an illustration, a type-led poster
  or anything else. The photographic register is added only when the prompt describes a
  photograph, and a drawing is not offered a photograph's negatives or positives.
- **Writing in the scene.** SUBJECT is told not to build a scene around readable writing. If
  the Scene still names a paper, screen or sign, a line and a negative keep it unreadable.
  A brief asking for a screenshot or dashboard switches this off.
- **Every word arrives.** A locked string missing from the Director's prompt is added back in
  code, and softening never changes a quoted string.
- **Negatives in code, before the TAILOR.** Template headings are removed, near-synonyms
  become one term, a requested CTA loses its ban, and terms that cannot apply (body terms
  with no body, group terms with no group, a part not in frame) go.
- **TAILOR node.** One fast call after the Director picks the negatives (at most 8) and
  positives (at most 3) that fit this prompt. Garbled-text, branding, CTA, user-exclusion and
  anatomy terms always stay. On a failure or after `ARMC3_TAILOR_TIMEOUT_S` (20 s) the
  list from the code step is used. `ARMC3_TAILOR=0` turns it off.
- **No checks after the render.** Everything above acts before the image is generated.

### TYPOGRAPHY (2026-11-17 drop, merged on top of the SLOP GUARD)

Read `TYPOGRAPHY.md` for the full design. Text is set in **real fonts** from the Kognie
library, meets **WCAG AA contrast** and per-platform legibility, and every graphic with text
returns `result["text_layers"]` (schema `kognie.text_layers.v2`) so the editor can recreate
each text element as an editable object.

- **TYPESET node** after the three specialists, before the DIRECTOR: one reasoning call plus
  the deterministic `typography.enforce()`. It produces the pre-generation metadata and a
  typeset layout PNG, which goes to the renderer as image 1, followed by the font sample
  sheets (or the client's own `font` reference) and then the client's references.
- **TEXT QA** after the render: registration (±120 px, ±15% scale) confirms or corrects each
  element, then contrast and legibility are checked against the painted artwork. OCR
  (`typography_ocr.py`, needs `easyocr`) is optional; without it unplaced elements are
  `unverified`, never a crash.
- **TYPOGRAPHER** picks fonts by library key. **GROUND** returns `font_requests` (typefaces
  the user named). New reference role **`font`**: the client's lettering sample, used as the
  only font reference.
- **With the SLOP GUARD.** The typeset text reaches the renderer in the layout image, so the
  Director is told not to restate it. The guard's "every word arrives" step is therefore
  skipped when TYPESET produced a layout, otherwise each string would be drawn twice. The
  short prompt's Text and Layout lines are swapped for `DIRECTOR_FINAL_SHORT_TYPESET`.
  `create_more()` does the same, and if its re-typeset fails, the locked strings are added
  back to its prompt before the render.
- **The copywriter's deck is what gets typeset**: TYPESET reads the locked text after the
  copy deck and the CTA hygiene have replaced GROUND's.
- **No text, no typography.** With nothing to set, TYPESET is skipped and the run is the
  same as before. A failed TYPESET call degrades to model-drawn text (`node_failed`).
- New result keys: `typography` (`fonts_used`, `user_notices` to show the user,
  `policy_log`, `user_font_image`), `text_layers`, `layout_image`. New events:
  `typeset_done`, `text_qa_done`.
- `KOGNIE_TYPOGRAPHY=off` turns it all off. `KOGNIE_TYPOGRAPHY_ASSETS` points at the fonts
  and sample sheets (default `./typography_assets`). Courier New is not shipped (licence).

**Keep the whole result JSON.** `create_more()` takes it as its input, and it is
the only record of what was sent to the renderer.

`status == "needs_input"` means GROUND found the brief unusable and returned
`result["question"]`. Rare by design — a thin brief is not a blocker.

### Progress events

`on_event` fires synchronously as each node completes:
`{"event": ..., "node": ..., "t": seconds_since_start}` with `node_start`,
`parallel_start`, `node_done`, `node_retry`, `node_failed`; with a copywriter,
also `copywriter_wait_start` / `copywriter_wait_done` (no `node`); one `antislop`
event with the active and exempted rules; and, with text, `node_start` for `typeset`,
`typeset_done` (elements, fonts, notices) and `text_qa_done` (statuses, `contrast_fail`,
`legibility_fail`).
`pipeline._live` is a working reference implementation.

---

## 4. Reference images — what to do in each case

A reference is not just "an image". **What it is *for* changes how every node
treats it**, and getting the role wrong is worse than supplying nothing. Pass
`{"path": ..., "role": ...}`; a bare path defaults to `hero`. Maximum 5.

| Role | What the pipeline does | The failure it prevents |
|---|---|---|
| `logo` | Placed **exactly as supplied** — same shapes, proportions, colours, lettering. Prominent position, clear space ≥ the mark's own height. The prompt is forbidden from *describing* it. | A model asked to "draw the logo" invents a different one every time. |
| `hero` | **Is** the subject. The finished graphic is one continuous photograph: the supplied pixels untouched, the rest of the canvas its own space continued outward in the same light. | Being pasted in as a framed card on a separately generated scene. |
| `product` | Same, at a higher bar — no invented features, finishes, ports or branding. Stands **in** the scene, on a real surface, with a real contact shadow. | An invented feature on a real product is a factual error, not a style choice. |
| `layout` | **Arrangement only** — grid, hierarchy, placement of masses, reading order. Its subject and palette must not leak in. | The graphic quietly becoming about the reference's subject. |
| `style` | **Treatment only** — palette, light quality, texture, finish. Nothing structural, nothing of its subject. | Its layout or subject arriving with the treatment. |

### Rules that follow from a supplied reference

**Every supplied reference reaches the renderer, logo included.** An earlier
build withheld the logo while the prompt still said "the supplied logo,
reproduced exactly as provided" — both renderers had no option but to invent
one. If the prompt talks about a file, the renderer has that file.

**A supplied logo unlocks branded packaging.** With no logo, every surface in the
scene is blank or turned away. With one, packaging, cups, bags, garments and
signage **may** carry that mark, and the primary package in shot **should**.
Invented label copy — product names, origins, weights, specs, tasting notes —
stays banned either way: a rendered label reading `ETHIOPIA YIRGACHEFFE,
1,900–2,200 MASL` is a factual claim about a product that does not exist.

**A hero or product image lifts the likeness guard.** When the brief names a real
person and no photo is supplied, the pipeline refuses to synthesise a face and
represents them typographically instead. A supplied hero or product lifts that;
a logo, layout or style reference does **not**, because it says nothing about the
person.

---

## 5. The create-more flow

```python
from create_more import create_more

lineage = [{"archetype": "original", "style_signature": "as first generated"}]

more = create_more(
    first_result,                 # the whole result dict from run()
    out_dir=Path("out/more1"),
    lineage=lineage,              # everything shown so far
)
lineage.append({"archetype": more["chosen_archetype"],
                "style_signature": more["style_signature"]})
```

### What it does and does not re-do

Everything upstream of the Director is already settled and is **reused as-is** —
the extracted facts, the locked strings, the subject. One call re-does the
Director's job under a different constraint.

```
first generation   6 reasoning calls   then render
create more        1 reasoning call    then render
```

That matters for correctness as much as cost: because it is not re-extracting
from the brief, it cannot quietly lose a fact the way a fresh run could.

### Two mechanisms, both required

**Content lock** is a list, not an instruction. Every locked string comes across
verbatim and exactly once, and the subject carries over. Measured 5/5 across a
full four-graphic lineage.

**Layout change is enforced in code.** The node must name the composition the
previous graphic used and commit to a different one from the menu; the code then
checks the choice against **the whole lineage**, not just the previous entry, and
retries with an explicit exclusion if it collides. Comparing only against the
last graphic lets click 3 return to click 1's composition.

### The four-click rule

Each combination's menu holds 3–4 compositions. `more["exhausted"]` goes `True`
when the lineage has used them all — **that is the signal to disable the button.**
`more["used_before"]` and `more["menu_size"]` let you show "3 of 4" in the UI.

Beyond the menu the node will improvise a composition rather than repeat one, so
a fifth click still returns something new; it is simply no longer drawn from the
catalogue.

### What the renderer is and is not given

User-supplied references go to the renderer as normal. **The previous generated
graphic does not.** Handing an image model the thing it is meant to diverge from
returns a near-copy — that is what `images.edit` is for and it is the opposite of
the request. The previous graphic reaches the node as text: a three-line lineage
digest of composition + style signature per prior graphic.

### Known limit — style converges

Composition is guaranteed unique. **Style is not.** Across a measured four-graphic
lineage all four came back dark-ground, because the subject was a candlelit
dinner and dark is the best-fitting answer every time. Type register varied on
its own; ground value did not.

This is the model converging on the optimum, not being lazy — and forcing a
bright variant would have misrepresented the event. If four visually distinct
*styles* matter for your UI, that needs a variation plan that separates
brief-pinned axes from brief-free ones; it is designed but not built.

---

## 6. `layout_guidance.json` — precomputed, held in memory

All 170 `design goal × platform` combinations in **one 0.93 MB file, read once at
import** into `generation_pipeline.GUIDANCE`.

**No S3. No per-request disk I/O. No storage credentials. No network on the
generation path.** The data is version-locked to the code that reads it, so
"which guidance produced this image" is answerable from the deployment version.

`KOGNIE_LAYOUT_GUIDANCE` overrides the path if you want to mount it elsewhere;
the default resolves against the module file, not the process working directory,
so a service started from any directory finds it.

### What each record holds

| Key | Carries |
|---|---|
| `aspect_ratio`, `canvas_px` | the canvas |
| `must_communicate` / `may_include` | what the graphic must land, never how it looks |
| `composition_menu` | 3–4 compositions that suit this combination, each with **when it fits** |
| `guidance.format_character` | what this format is for |
| `guidance.platform_craft` | what this canvas demands — chrome, viewing distance, legible size |
| `guidance.art_direction_prompts` | five art-direction provocations |
| `defect_tags` | defect vocabulary for the negative prompt — never a subject ban |
| `authored` | a hand-written house brief, on the 10 combinations that have one |

`load_guidance()` resolves `authored` for you: hand-written wins where it exists,
otherwise the derived guidance fills the slot, so all 170 carry real format
character and art direction.

**Guidance, not specification.** There is deliberately no zone geometry, no
typography scale and no reading order — prescribing those is what produced ten
variations of one centred layout. `hard_rules`, `creative_latitude` and
`asset_policy` were dropped because nothing read them; the safety rules they
carried live in the prompts, where they are enforced.

It is regenerated offline by `build_layout_guidance.py` in the source repo from
the Phase 0 factory profiles. Those profiles are build input, not a runtime
dependency.

---

## 7. The flow

```
GROUND      extract facts                          thinking = low
   |        locked text, other stated facts, redemption mechanism.
   |        Deliberately the LOWEST budget: it copies strings, and a
   |        reasoning budget on an extraction task invites embellishment.
   v
SUBJECT     what is this graphic actually of?       thinking = high
   |        Concrete nouns, before anyone designs. Reference-role rules
   |        and the likeness guard are applied here.
   v
   +--> TYPOGRAPHER    type, hierarchy, CTA, actions, quotations  ]
   +--> ART DIRECTOR   imagery, light, colour, material           ]  concurrent
   +--> COMPOSITOR     placement, safe area, focal path           ]
   |        Each has its own model fallback: a failed provider degrades
   |        one discipline rather than the whole request.
   v
DIRECTOR    arbitrate, compile ONE prompt           thinking = medium
   |        Decides where specialists conflict; never averages them.
   v
RENDER      gpt-image-2
            Canvas is a parameter, not a sentence. Falls back to the
            secondary renderer only when the shape cannot be hit.

CREATE MORE  reuses GROUND + SUBJECT, re-does the DIRECTOR    thinking = medium
```

Locked strings are de-duplicated before compiling — the schemas can mandate two
overlapping fields (a milestone year *and* a headline opening with that year),
which is how "25 YEARS" once rendered twice. `create_more` applies the same
dedupe.

### Canvas sizing

gpt-image-2 takes pixels, not a ratio. The pipeline computes the **smallest exact
fit** at or above the platform's real short edge (multiples of 16, bounded
640–3840). Smallest on purpose: aspect accuracy is what matters and extra pixels
cost render time without being seen — a YouTube thumbnail displays ~210px wide.

If the ratio cannot be hit within 2% relative error the job routes to the
fallback renderer, which takes the ratio string directly.
`result["render"]["routing"]` records which path was taken and why.

---

## 8. Measured cost and latency

| | reasoning | wall | cost |
|---|---|---|---|
| First generation | ~60s | **~110s** | **~$0.104** |
| Create more | **24.9s** | **62.7s** | **$0.084** |

Create-more is 29% faster and 18% cheaper end to end. Two things bought that:
the previous compiled prompt was replaced by a three-line digest (reasoning time
tracked input length almost exactly), and thinking dropped from `high` to
`medium` — justified because this node never reasons about the subject.

Reasoning cost is exact; token counts come back from the API and thinking tokens
are frequently the majority. **Image cost is a per-quality-tier estimate** —
`usage.output_tokens` is recorded in every result JSON, so one billing line
converts it.

First generation is ~10s slower than v3.1 measured at ~100s: the added rules made
the prompts longer. That was a deliberate trade for the correctness fixes in §2.

Levers, in order of effect: `ARMC3_IMAGE_QUALITY` (`high` roughly quadruples
image cost for the same size), then per-node thinking. Canvas size is **not** a
lever — 4× the pixels costs about 13% more time.

---

## 9. Errors to catch

| Exception | When | What to do |
|---|---|---|
| `BriefNotFound` | `goal`×`platform` unknown | 400 — the pair is not supported |
| `MissingCredential` | `GEMINI_API_KEY` unset | 500 — a deploy problem |
| `ValueError` | more than 5 reference images | 400 |

Both custom types subclass `brief_store.PipelineError`, so `except PipelineError`
catches the pair. `RuntimeError` means a node that cannot be skipped (GROUND, SUBJECT or
the DIRECTOR) failed on both models; a DIRECTOR reply with no prompt is retried once
first. A dead specialist, copywriter or TAILOR degrades the run and never raises.
GROUND gets `ARMC3_GROUND_TIMEOUT_S` (60 s) per model, other nodes `ARMC3_CALL_TIMEOUT_S`
(180 s), and the render `ARMC3_RENDER_TIMEOUT_S` (300 s). A **render** failure does not raise — it returns
`result["render"]["ok"] is False` with every attempt recorded, because a compiled
prompt is still worth returning.

---

## 10. Known limits — read before shipping

- **No render retries.** A failed generation is reported, not retried. Text
  accuracy rests entirely on prompting. Add retries at the call site if you want
  them.
- **`run()` is synchronous and takes ~110 seconds.** Do not call it inside a
  request handler — queue it, and drive progress from `on_event`.
- **Both calls write to `out_dir`.** Give them a real path.
- **Style converges across a lineage** — see §5.
- **160 of 170 combinations have derived rather than hand-written guidance.**
  They carry real platform craft and art direction, but `format_character` is
  factual rather than evocative. Authoring the rest is a content task.
- **Image cost is a tier estimate**, not billed figures.
- **Brand kits are not in this package.** Reference images are; brand colour
  profiles are not.

---

## 11. Rebuilding

From the source repo's `experiments/`:

```bash
python build_layout_guidance.py && python make_package_v32.py
```

Re-run after any change. The packager re-extracts, re-applies every named
transform, refuses to build if an anchor has moved, scans each shipped text file
for anything shaped like an API key, and rewrites the zip.
