"""
create_more.py
----------------------------------------------------------------------
"Show me another one" — a different graphic carrying identical content.

THE PROBLEM
    A user has a graphic they broadly like and wants to see it laid out a
    different way. Re-running the whole pipeline is wrong twice over: it costs
    six reasoning calls to re-derive facts that are already known, and because
    reasoning is stochastic it can come back with a different SUBJECT, different
    copy emphasis, even different extracted facts. The user asked for another
    layout, not another interpretation.

THE APPROACH
    Everything upstream of the director is already settled and is reused as-is:
    the extracted facts, the locked strings, the subject. One call re-does the
    director's job under a different constraint. So:

        full pipeline   6 reasoning calls  ~56s   then render
        create more     1 reasoning call   ~20s   then render

    The content is LOCKED and the layout is FORCED to change. Those are two
    different mechanisms and both matter:

    CONTENT LOCK is a list, not an instruction. Every locked string comes across
    verbatim, the subject matter is carried over, and the node is told it has no
    licence to re-write, re-emphasise or drop any of it. It is not re-extracting
    from the brief, so it cannot quietly lose a fact the way a fresh run could.

    LAYOUT CHANGE is enforced by construction rather than by asking nicely. The
    combination's composition menu holds 3-4 genuinely different archetypes
    ("full_bleed|centered", "hero_panel|right", "inset|centered",
    "typographic|centered"). The node must name the archetype the previous
    graphic used and commit to a DIFFERENT one; the code then checks that the
    two differ and retries once with an explicit exclusion if they do not.
    Asking a model to "make it different" without that check produces the same
    composition with the colours moved around.

WHAT THE RENDERER IS AND IS NOT GIVEN
    User-supplied references -- logo, product, hero -- still go to the renderer,
    because those are the client's assets and must survive every variation.

    The PREVIOUS GENERATED GRAPHIC is deliberately NOT sent. Handing an image
    model the thing it is meant to diverge from produces a near-copy: that is
    what images.edit is for and it is the opposite of the request. The previous
    graphic reaches this node as TEXT -- its compiled prompt, which already
    states its placement explicitly -- so the node knows precisely what to avoid
    without the renderer being anchored to it.

    python create_more.py --run blind_final/final_run_*.json --only F02 F03 F05
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from pathlib import Path

import generation_pipeline as v31
import antislop
import reference_roles as refroles

# medium, not high. This node does NOT reason about the subject from scratch --
# ground, subject and the locked strings all arrive settled. It arbitrates a
# layout inside a fixed content lock, which is the same class of work the main
# DIRECTOR does at medium. Measured against the high-thinking baseline below.
CREATE_MORE_THINKING = os.environ.get("KOGNIE_MORE_THINKING", "medium")

CREATE_MORE = """You are the design director, and you are producing ANOTHER VERSION of a graphic
that already exists. Same message, same facts, same subject. Different design.

DESIGN GOAL: {goal}
PLATFORM: {platform} ({aspect_ratio}, {w}x{h}px)

WHAT THIS GRAPHIC IS ABOUT — carry this over unchanged:
{subject}

CONTENT LOCK — every one of these appears in your new version, verbatim, and EXACTLY ONCE.
You are not re-writing the copy and you are not deciding what deserves space. It all does:
{locked}

ALSO STATED BY THE CLIENT — same standing as the locked text:
{other_facts}

USER'S OWN DIRECTION — still binding on this version:
{directives}

USER EXCLUSIONS: {exclusions}
{people_note}
{reference_rules}
{typography_note}

WHAT THIS LINEAGE HAS ALREADY SHOWN — every one of these is spent. Your version
must not read like any of them, in composition OR in styling:
{lineage}

COMPOSITIONS AVAILABLE FOR THIS GOAL AND PLATFORM:
{menu}

YOUR JOB

1. NAME WHAT THE PREVIOUS VERSION DID. Identify its archetype from the menu above — where the
   imagery sat, how the type was aligned, what carried the eye. Be specific and honest; if it
   used something not on the menu, say so in the same vocabulary.

2. CHOOSE A DIFFERENT ONE. Not a variation of the same idea — a different structure. If the
   previous was a full-bleed photograph with centred type, do not return a full-bleed photograph
   with type nudged left. Move the imagery somewhere structurally different, change what the eye
   hits first, change what carries the message. The reader should be able to tell at a glance
   that these are two designs, not two exports of one design.

3. THE STYLE MAY MOVE TOO. Palette, light, type pairing, material and mood are all yours to
   change, and changing them is usually right — this is a genuinely different take, not a
   re-skin. The one thing that may not move is what the graphic SAYS and what it DEPICTS.

4. KEEP AN ACTION WHOLE. Where the content lock contains something the reader acts on — a code,
   a deadline, a booking step — the instruction, the token and what it yields stay together and
   adjacent. Adding the instruction words around a locked token does NOT breach the content lock:
   locked "HARVEST30" becomes "USE CODE HARVEST30 AT CHECKOUT" and the locked string is intact.
   A bare code in a box is not an action a reader can complete.

   THE TOKEN OUTRANKS ITS INSTRUCTION. The code is the dominant element of that unit — larger,
   heavier or in the accent — and the words around it are smaller and lighter. The eye lands on
   the code and picks up "use code" and "at checkout" second.

   ATTRIBUTED QUOTATIONS CARRY QUOTATION MARKS. If a locked string is something a named person or
   source SAID, the artwork DRAWS a visible opening quotation mark before its first word and a
   visible closing one after its last. These are glyphs the viewer sees, NOT the delimiters you
   use when writing the string down — you already quote every locked string with " when you state
   it, and "set it in quotation marks" has been read as meaning your own punctuation and then
   vanished from the render. Say it so the renderer cannot mistake it. The attribution goes
   visually subordinate, usually after an em dash. The marks do not breach the content lock: the
   locked string is intact between them. Without them a person's words read as the brand's own
   slogan.

5. SAY EACH THING ONCE. Every locked string once, in the place it works hardest in THIS layout.
   And each FACT once: the headline and sub-headline (and any eyebrow, badge or body line) carry
   different information. A sub-headline never restates the headline's offer, number, product or
   date in other words; if two strings say the same thing, render it once, in the stronger
   element, and drop the other line rather than invent replacement copy.

{cta_rule}

6. GUARD. No invented facts, no invented dates or prices, no invented logos or marks, and never
   a real third-party trademark. {branding_note} Nothing crosses the frame edge; all text sits
   inside the safe area with real contrast behind it. {safe_area}

{slop_director}

Return ONLY strict JSON:
{{
  "previous_archetype": "<label from the menu vocabulary, e.g. full_bleed|centered>",
  "chosen_archetype": "<a DIFFERENT label>",
  "what_changed": "<one sentence: how this version is structurally different>",
  "style_signature": "<six words or fewer naming this version's GROUND VALUE (dark/light/mid), its
     type register and its photographic treatment, e.g. 'dark ground, serif display, wide interior'>",
  "final_prompt": "<{final_spec}>",
  "negative_prompt": "<comma-separated defects to avoid, starting from: {defect_tags}>"
}}"""

CREATE_MORE_FINAL = """one flowing paragraph, 340-520 words, ready for an image model. Natural
     language only — no coordinates, no zone numbers, no pixel values, no real font names. It
     must carry: the subject and its concrete physical details; every locked string exactly as
     given; the type pairing and hierarchy; the palette and its value structure; the light and
     materials; and the SPECIFIC, directional placement of every element. End by stating it is a
     finished, production-quality graphic."""

# The same short prompt as run(); a variation that went back to 500 words would undo it.
CREATE_MORE_FINAL_SHORT = ("a short plain brief for an image model: 120-200 words, not counting the text "
                           "strings. No coordinates, no zone numbers, no pixel values, no real font names. "
                           "Use exactly these labelled lines, separated by newlines:\n"
                           + v31.SHORT_PROMPT_LINES.replace('"', "'"))


CM_TYPOGRAPHY_NOTE = """TYPOGRAPHY IS SET SEPARATELY, IN REAL FONTS, AFTER YOU. The fonts, copy and colours of the
original carry over and will be typeset to follow the placement you describe, then supplied to the
renderer as a finished typographic layout image. So in final_prompt: state WHERE each text block sits
in this new composition (directional, as usual), but do NOT quote the strings, do NOT name or describe
typefaces and do NOT give type sizes. Include the sentence: 'All typography is supplied, already set, in
the attached typographic layout (image 1): reproduce it exactly and add no other text anywhere.'
This overrides the instructions below about writing the locked strings and the type pairing into the
prompt. Every other requirement stands."""


def _menu_block(guidance: dict) -> str:
    items = guidance.get("composition_menu") or []
    if not items:
        return "  (none listed — choose any structure that suits the canvas)"
    return "\n".join(
        f"  - {c['shape']}: {c['idea']} — {c.get('note', '')}" for c in items)


def create_more(prev: dict, *, out_dir: Path, generate_image: bool = True,
                on_event=None, attempt_limit: int = 3,
                lineage: list | None = None) -> dict:
    """One more version of `prev`, same content, different layout.

    `prev` is a previous result: anything carrying goal, platform, aspect_ratio,
    final_prompt, and the ground facts. A blind-test record works directly.
    """
    import os
    from google import genai
    from google.genai import types

    events = []
    T0 = time.time()

    def emit(e):
        e["t"] = round(time.time() - T0, 2)
        events.append(e)
        if on_event:
            on_event(e)

    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise v31.MissingCredential(
            "GEMINI_API_KEY is not set - create_more cannot reason without it")
    client = genai.Client(
        api_key=key,
        http_options=types.HttpOptions(timeout=v31.CALL_TIMEOUT_S * 1000))

    goal, platform = prev["goal"], prev["platform"]
    guidance = v31.load_guidance(goal, platform)
    aspect = prev.get("aspect_ratio") or guidance["aspect_ratio"]
    w, h = (guidance.get("canvas_px") or [1080, 1080])[:2]

    ground = prev.get("ground") or {}
    # dedupe_locked, not the raw extraction. The schemas mandate overlapping
    # fields -- brand_name "Marrow & Mend" is a substring of attribution "Ines
    # Roca, founder of Marrow & Mend" -- and run() drops the shorter one before
    # compiling. Reading ground["locked_text"] here skipped that step, so the
    # brand name came back as BOTH an eyebrow and part of the attribution.
    locked = prev.get("locked_text") or ground.get("locked_text") or {}
    locked, dropped_locked = v31.dedupe_locked(locked)
    locked_s = "\n".join(f'  {k}: "{v}"' for k, v in locked.items()) or "  (none)"

    # A copy deck from run()'s copywriter hook carries every fact that goes on the
    # graphic; GROUND's other facts still hold what the deck chose to leave out.
    from_deck = bool((prev.get("copy") or {}).get("locked"))
    other = [] if from_deck else [str(x).strip() for x in (ground.get("other_stated_facts") or [])
                                  if str(x).strip()]
    # Same hygiene as run(): a fact that only restates the locked text came back
    # as a sub-headline repeating the headline's offer.
    other, dropped_facts = v31.dedupe_facts(other, locked)
    dropped_locked = dropped_locked + dropped_facts
    # A call to action only when the user asked for one; run() records the
    # answer on its result, and a prev without it (older, or read back from an
    # image) gets none.
    cta_requested = bool(prev.get("cta_requested"))
    if not cta_requested:
        locked, other, dropped_cta = v31.strip_cta(locked, other)
        dropped_locked = dropped_locked + dropped_cta
        locked_s = "\n".join(f'  {k}: "{v}"' for k, v in locked.items()) or "  (none)"
    red = ground.get("redemption") or {}
    if red.get("token") and red.get("how") and not from_deck:
        other.append(f'the code {red["token"]} is used {red["how"]}')
    other_s = "\n".join(f"  - {x}" for x in other) or (
        "  (none — the copy deck above carries every fact that goes on the graphic)"
        if from_deck else "  (none)")

    directives = ground.get("visual_directives") or []
    dir_s = "; ".join(directives) if directives else "none — the art direction is yours"
    ex = (ground.get("user_exclusions") or {})
    ex_terms = list(ex.get("explicit_absences") or []) + list(ex.get("style_contradictions") or [])
    ex_s = ", ".join(ex_terms) if ex_terms else "none"

    people = [p.get("name") for p in (ground.get("named_people") or []) if p.get("name")]
    typed = refroles.normalise(prev.get("refs") or prev.get("reference_images") or [])
    protect = bool(people) and not any(t["role"] in ("hero", "product") for t in typed)
    people_note = ""
    if protect:
        people_note = (f"NAMED PERSON WITHOUT A PHOTO ({', '.join(people)}): do NOT render a "
                       "photorealistic face or an invented likeness.")

    # Same inversion as the main pipeline: blank surfaces guard against INVENTING
    # a mark, which stops being the right answer once the client supplies theirs.
    has_logo = any(t["role"] == "logo" for t in typed)
    branding_note = (
        "Packaging, cups, bags, garments and signage in the scene MAY carry the client's supplied "
        "logo, reproduced from the attached file exactly as given and reading as printed on the "
        "material; invent no other wordmark, sub-brand or label copy."
        if has_logo else
        "No brand name is printed on any depicted object — not the client's own and not a real "
        "third-party brand; those surfaces are blank or turned away.")

    subject = prev.get("subject")
    if isinstance(subject, dict):
        subject = subject.get("subject", "")
    subject_s = subject or "(carry over the subject of the previous version)"

    # Comparing against the immediately previous graphic only guarantees that
    # click N differs from click N-1 -- variation 3 could legally return to
    # variation 1's composition. With four options per combination the lineage
    # has to be carried, or the third click starts recycling.
    # The lineage is carried as a DIGEST, not as the previous compiled prompt.
    # That prompt ran 340-520 words and reasoning time tracked its length almost
    # exactly (27s / 43s / 57s across the last batch). Three short fields per
    # prior graphic say the same thing in a fraction of the tokens -- and say it
    # about STYLE as well as layout, which the prose only implied.
    hist = list(lineage or [])
    used = [h.get("archetype") for h in hist if h.get("archetype")]
    menu_labels = [c["shape"] for c in (guidance.get("composition_menu") or [])]
    remaining = [m for m in menu_labels if m.lower() not in {u.lower() for u in used}]

    if hist:
        rows = []
        for n, h in enumerate(hist):
            tag = "the original" if n == 0 else f"variation {n}"
            bits = [h.get("archetype") or "?"]
            if h.get("style_signature"):
                bits.append(h["style_signature"])
            rows.append(f"  - {tag}: " + " — ".join(bits))
        lineage_block = "\n".join(rows)
        lineage_block += ("\n\nSTILL AVAILABLE from the menu: "
                          + ", ".join(remaining) if remaining else
                          "\n\nEvery listed composition is spent. Build "
                          "something structurally distinct from all of them.")
    else:
        lineage_block = "  (nothing yet — this is the first variation)"

    # Same SLOP GUARD as run(), rebuilt from the brief text it recorded, since this call may
    # change the style. A prev without that record (older) gets no guard rather than one
    # blind to what its brief asked for.
    slop = (prev.get("antislop") or {}).get("brief_text")
    guard = antislop.build(slop or "", enabled=v31.ANTISLOP and slop is not None,
                           short_prompt=v31.SHORT_PROMPT)
    short_prompt = v31.SHORT_PROMPT and bool(guard["active"])

    # The text goes through TYPESET again when the original was typeset: the prompt then places the
    # text blocks without quoting them, and the strings reach the renderer in the layout image.
    prev_ts = (prev.get("text_layers") or {}).get("elements")
    canvas_size, _ = v31.openai_size(aspect, platform)
    typeset_planned = bool(v31.ty.ENABLED and prev_ts and canvas_size)

    used_block = ""
    banned = used_block
    result_json, meta = None, None
    for attempt in range(1, attempt_limit + 1):
        prompt = CREATE_MORE.format(
            goal=goal, platform=platform, aspect_ratio=aspect, w=w, h=h,
            subject=subject_s, locked=locked_s, other_facts=other_s,
            directives=dir_s, exclusions=ex_s, people_note=people_note,
            reference_rules=refroles.block(typed, refroles.DIRECTOR_RULE),
            typography_note=CM_TYPOGRAPHY_NOTE if typeset_planned else "",
            branding_note=branding_note,
            lineage=lineage_block,
            menu=_menu_block(guidance) + banned,
            safe_area=v31.safe_area(platform), slop_director=guard["director"],
            final_spec=CREATE_MORE_FINAL_SHORT if short_prompt else CREATE_MORE_FINAL,
            cta_rule=(v31.CTA_DIRECTOR_ON if cta_requested else v31.CTA_DIRECTOR_OFF)
                     .replace("5. ", "   ", 1),
            defect_tags=", ".join((guidance.get("defect_tags") or [])
                                  + ([] if cta_requested else [v31.CTA_NEGATIVES])))
        emit({"event": "node_start", "node": "create_more", "attempt": attempt})
        result_json, meta = v31._call(client, prompt, "create_more", emit,
                                      thinking=CREATE_MORE_THINKING, temp=0.9)
        prev_arch = (result_json.get("previous_archetype") or "").strip()
        new_arch = (result_json.get("chosen_archetype") or "").strip()
        clash = {prev_arch.lower(), *(u.lower() for u in used)}
        if new_arch and new_arch.lower() not in clash:
            break
        # Asking for "different" is not enough on its own -- this is the check
        # that makes the difference real rather than hoped for.
        emit({"event": "node_retry", "node": "create_more",
              "error": f"chose an already-used archetype ({new_arch!r}); excluding it"})
        used.append(new_arch)
        banned = used_block + (f"\n\nYou just returned {new_arch!r}, which this lineage has "
                                f"already shown. It is now FORBIDDEN too. Choose a structurally different one.")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{v31.base.slug(goal)}__{v31.base.slug(platform)}_more_{int(time.time())}"
    # A supplied hero or product image decides who is in the frame, not the specialists.
    who = (None if any(t["role"] in ("hero", "product") for t in typed)
           else antislop.in_frame(prev.get("art_director"), prev.get("subject")))
    if not result_json.get("final_prompt"):
        raise RuntimeError("create_more returned no final prompt")
    tailor_metas = []
    final_prompt, negative_prompt, conditional = v31.finish_prompt(
        client, result_json["final_prompt"], result_json.get("negative_prompt") or "", guard, ex_terms,
        who, (prev.get("art_director"), prev.get("subject")), None if typeset_planned else locked,
        any(t["role"] == "logo" for t in typed), bool(cta_requested), emit, tailor_metas)
    out = {
        "status": "ok", "goal": goal, "platform": platform, "aspect_ratio": aspect,
        "source_id": prev.get("id"),
        "previous_archetype": result_json.get("previous_archetype"),
        "chosen_archetype": result_json.get("chosen_archetype"),
        "what_changed": result_json.get("what_changed"),
        "style_signature": result_json.get("style_signature"),
        "used_before": used,
        "menu_size": len(menu_labels),
        "exhausted": len({*(u.lower() for u in used),
                          (result_json.get("chosen_archetype") or "").lower()}
                         ) >= len(menu_labels),
        "final_prompt": final_prompt,
        "negative_prompt": negative_prompt,
        "antislop": {"active": guard["active"], "exempted": guard["exempted"], **conditional},
        "locked_text": locked, "locked_text_dropped": dropped_locked,
        "cta_requested": cta_requested,
        "reasoning_s": round(time.time() - T0, 1),
        "cost_llm": round((meta.get("cost", 0.0) if meta else 0.0)
                          + sum(m.get("cost", 0.0) for m in tailor_metas), 6),
        "thinking_tokens": meta.get("thinking_tokens", 0) if meta else 0,
    }

    # TYPESET for the new composition. The copy, fonts and colours carry over
    # from the original; only the placement follows this variation's layout.
    typeset_ts = None
    if typeset_planned:
        cw, chh = (int(x) for x in canvas_size.split("x"))
        emit({"event": "node_start", "node": "typeset"})
        try:
            carried = {"fonts": {e["role"]: e["font_key"] for e in prev_ts},
                       "copy_as_set_before": [{"role": e["role"], "text": e["text"], "color_hex": e["color_hex"]}
                                              for e in prev_ts]}
            plan, m2 = v31._call(client, v31.ty.typeset_prompt(
                goal=goal, platform=platform, W=cw, H=chh, locked=locked_s, other_facts=other_s,
                typo=carried, comp={"composition": out.get("chosen_archetype"),
                                    "placement_described": (out.get("final_prompt") or "")[:2500]},
                art=prev.get("art_director"), font_requests=[], user_font=None),
                "typeset", emit, thinking="medium", temp=0.4)
            out["cost_llm"] = round(out["cost_llm"] + m2.get("cost", 0.0), 6)
            typeset_ts = v31.ty.enforce(plan, platform=platform, W=cw, H=chh, font_requests=[], user_font=None)
            if not typeset_ts["elements"]:
                typeset_ts = None
        except Exception as e:
            emit({"event": "node_failed", "node": "typeset", "error": f"{type(e).__name__}: {str(e)[:120]}"})
            typeset_ts = None
        if not typeset_ts:
            # The prompt was told not to quote the text, and no layout image will carry it now.
            out["final_prompt"] = v31.restore_text(out["final_prompt"], locked)

    if generate_image:
        # Only the CLIENT's assets reach the renderer. The previous generated
        # graphic is withheld on purpose: an image model handed the thing it is
        # meant to diverge from returns a near-copy.
        gen_refs = [t["path"] for t in typed]
        png = out_dir / f"{stem}.png"
        emit({"event": "node_start", "node": "render"})
        t0 = time.time()
        render_prompt = out["final_prompt"]
        if typeset_ts:
            # This node's prompt was written before the text was set, so tell the
            # renderer the words it mentions ARE the text in image 1.
            try:
                render_prompt, gen_refs, layout_png = v31.typeset_render_inputs(
                    typeset_ts, out["final_prompt"],
                    typed, out_dir / f"{stem}.layout.png")
                out["layout_image"] = str(layout_png)
            except Exception as e:
                emit({"event": "node_failed", "node": "typeset",
                      "error": f"layout image: {type(e).__name__}: {str(e)[:120]}"})
                typeset_ts = None
                render_prompt = out["final_prompt"] = v31.restore_text(out["final_prompt"], locked)
        r = v31.render(client, types, render_prompt, out["negative_prompt"] or "",
                       aspect, png, gen_refs, platform=platform)
        out["render"] = r
        out["cost_image"] = round(r.get("price", 0.0), 6)
        if r["ok"]:
            out["image"] = str(png)
        emit({"event": "node_done", "node": "render",
              "seconds": round(time.time() - t0, 1), "model": r.get("rendered_by")})
        if typeset_ts and r["ok"]:
            out["text_layers"] = v31.run_text_qa(png, typeset_ts,
                                                 any(t["role"] == "font" for t in typed), emit)
            (out_dir / f"{stem}.text_layers.json").write_text(v31.ty.dumps(out["text_layers"]),
                                                              encoding="utf-8")

    out["cost_total"] = round(out["cost_llm"] + out.get("cost_image", 0), 6)
    out["wall_s"] = round(time.time() - T0, 1)
    out["events"] = events
    (out_dir / f"{stem}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None,
                    help="a *_run_*.json to take source graphics from")
    ap.add_argument("--only", nargs="*", default=["F02", "F03", "F05"])
    ap.add_argument("--out", default="create_more_out")
    a = ap.parse_args()

    path = a.run or sorted(glob.glob("blind_final/final_run_*.json"))[-1]
    recs = {r["id"]: r for r in json.load(open(path, encoding="utf-8"))["records"]
            if r.get("status") == "ok"}
    picked = [recs[i] for i in a.only if i in recs]
    if not picked:
        raise SystemExit(f"none of {a.only} found in {path}")

    out_dir = Path(a.out)
    print("=" * 74)
    print(f"CREATE MORE — {len(picked)} graphics, one new version each")
    print("  content locked; layout and style free to move")
    print("=" * 74)
    results = []
    for r in picked:
        print(f"\n{r['id']}  {r['goal']} x {r['platform']}")
        d = out_dir / r["id"]
        res = create_more(r, out_dir=d, on_event=lambda e: None)
        print(f"   was:  {res['previous_archetype']}")
        print(f"   now:  {res['chosen_archetype']}")
        print(f"   why:  {res['what_changed']}")
        print(f"   {res['wall_s']}s   ${res['cost_total']:.4f}   "
              f"{'rendered ' + str(res.get('image')) if res.get('image') else 'RENDER FAILED'}")
        res["source_image"] = (r.get("arms", {}).get("pipeline_gpt", {}) or {}).get("image")
        results.append(res)

    p = out_dir / f"create_more_{int(time.time())}.json"
    p.write_text(json.dumps({"source_run": path, "results": results},
                            indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[ok] {len(results)} variations -> {p}")


if __name__ == "__main__":
    main()
