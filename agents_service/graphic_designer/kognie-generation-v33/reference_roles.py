"""
reference_roles.py
----------------------------------------------------------------------
A reference image is not just "an image". What it is FOR changes everything
about how the pipeline must use it, and getting that wrong is worse than
having no reference at all:

    logo      must be reproduced, never redrawn. A model asked to "draw the
              logo" invents a different one every time.
    hero      IS the subject. Preserve the supplied pixels; describing what
              they contain invites the renderer to rebuild it from words.
    product   the same, but the accuracy bar is higher: an invented feature on
              a real product is a factual error, not a style choice.
    layout    ONLY its arrangement is borrowed -- the grid, the hierarchy, the
              placement of masses. Its subject, palette and content must not
              leak into a graphic about something else entirely.
    style     ONLY its visual treatment -- palette, light, texture, finish.
              Not its layout, not its subject.
    font      ONLY its lettering -- the client's own sample of the typeface they
              want. It is the sole font reference for every piece of text; its
              words, colours, layout and subject must not appear.

The failure mode each rule prevents is written next to it, because the rules
look arbitrary otherwise and get "simplified" later by someone who does not
know what they cost.
"""

from __future__ import annotations

ROLES = ("logo", "hero", "product", "layout", "style", "font")

FONT_ROLE = (
    "A FONT reference is attached: the client's own sample of the lettering they want. It "
    "governs ONLY the letterforms of the text in this graphic. Its words, colours, layout, "
    "imagery and subject are irrelevant and must not appear or influence anything else.")

# Said to every node that touches a supplied hero or product, in one place so
# the four tables cannot drift apart. F04 in the final blind test came back as a
# gold-bordered card pasted onto a separately generated scene, because the hero
# rules spoke in layout vocabulary ("framing", "placement", "crop") while the
# product rules spoke in world vocabulary ("the environment around it") -- and
# the product one integrated correctly.
INTEGRATE = (
    'The finished graphic is ONE continuous photograph. The attached image supplies that photograph, and the rest of the canvas is its own space continued outward — the same room, the same surface, the same light falling the same way. It is NEVER a rectangle sitting on a different picture: no frame, no border, no inset card, no polaroid edge, no rounded crop, no drop shadow, no second photographic scene behind it. A supplied image that reads as pasted on has failed, however good the pixels are.')

# What the SUBJECT node is told. It decides what the graphic depicts, so it
# needs to know whether a reference supplies the subject or merely dresses it.
SUBJECT_RULE = {
    "logo": (
        "A LOGO file is attached and WILL BE PLACED IN THE GRAPHIC as supplied. "
        "It is not subject matter and not the hero: do not describe it and do not "
        "let it influence what the graphic depicts. Your only obligation is to "
        "leave the design room for it."),
    "hero": (
        "A HERO image is attached and IS the subject. Do not invent a scene and "
        "do not invent a second world for it to sit inside. " + INTEGRATE + " So "
        "hero_scene describes how that ONE world CONTINUES outward from the "
        "attached image — what the same room holds just beyond its edges, at the "
        "same depth and in the same light. Never describe what it contains — "
        "naming its contents invites the renderer to rebuild it from your words "
        "instead of preserving the supplied pixels."),
    "product": (
        "A PRODUCT image is attached and IS the subject. The object must appear "
        "exactly as supplied: no invented features, finishes, ports, buttons, "
        "textures or branding. Your authentic_details may describe the WORLD "
        "around the product, never the product itself. An invented feature on a "
        "real product is a factual error, not an art-direction choice."),
    "layout": (
        "A LAYOUT reference is attached. It is a STRUCTURAL reference only. This "
        "graphic is about something completely different, so ignore its subject, "
        "its palette and its content entirely. Do not let it influence what this "
        "graphic depicts."),
    "style": (
        "A STYLE reference is attached. Borrow only its visual treatment. It does "
        "not tell you what this graphic depicts."),
    "font": FONT_ROLE,
}

ART_RULE = {
    "logo": (
        "The attached logo goes INTO this graphic exactly as supplied — same "
        "shapes, same proportions, same colours, same lettering. It is placed, "
        "never redrawn. Design a clean, quiet, tonally simple area for it to sit "
        "against so it reads at a glance without being outlined, boxed or given a "
        "drop shadow, and specify no mark of your own anywhere else."),
    "hero": (
        "The attached image is the hero and its own pixels are untouched. Your "
        "job is to make the REST of the canvas belong to the same photograph: "
        "same direction and quality of light, same colour temperature, same depth "
        "of field, same grain and the same surfaces running on past its edges. " +
        INTEGRATE + " Do not restate its contents."),
    "product": (
        "The attached product is the hero and is rendered exactly as supplied. "
        "Direct the environment, light and surfaces AROUND it so the object sits "
        "IN that world — resting on a real surface, lit by the scene's own light, "
        "casting the contact shadow that placement would cast. Do not describe, "
        "restyle, recolour or embellish the product itself, and never present it "
        "in a frame, inset card or cut-out box."),
    "layout": (
        "The attached layout reference governs ARRANGEMENT ONLY. Take nothing "
        "visual from it — not its palette, not its lighting, not its subject. "
        "Your palette and light are chosen fresh for this brief."),
    "style": (
        "The attached style reference governs the LOOK: palette, light quality, "
        "texture, finish, grade. Take nothing structural from it and none of its "
        "subject matter. Describe the treatment you are matching in words, so the "
        "renderer has it even where the reference is ambiguous."),
    "font": FONT_ROLE,
}

COMP_RULE = {
    "logo": (
        "The supplied logo is a PROMINENT element, not a footnote. Give it a "
        "deliberate position — masthead or a strong corner — at a size where it "
        "is immediately legible, with clear space around it equal to at least the "
        "height of its own mark. State its position and its size relative to the "
        "frame. It must sit within the safe area and never overlap busy imagery "
        "or other type."),
    "hero": (
        "The supplied hero ANCHORS the composition and the frame opens out from "
        "it — its own background extended to the canvas edges. Do not give it a "
        "position on a page as though it were an object; say where its subject "
        "falls in the frame and which parts of that continued world are quiet "
        "enough to carry type. " + INTEGRATE),
    "product": (
        "The supplied product stands IN the scene — on a real surface, at a "
        "believable scale, with the contact shadow that placement would cast. "
        "Say where it sits and what surrounds it. " + INTEGRATE),
    "layout": (
        "The attached layout reference is your STRUCTURE. Reproduce its "
        "arrangement: where the masses sit, the type column, the balance of "
        "occupied and empty space, the reading order. Fill that structure with "
        "THIS brief's completely different content. Do not copy its subject."),
    "style": "The style reference does not govern layout. Compose freely.",
    "font": FONT_ROLE,
}

# What the DIRECTOR must put in the compiled prompt, per role.
DIRECTOR_RULE = {
    "logo": (
        "A logo image IS ATTACHED to this request. Say plainly that the attached "
        "logo image is to be placed into the composition EXACTLY AS SUPPLIED — its "
        "own shapes, proportions, colours and lettering, unaltered — and that it "
        "must not be redrawn, restyled, recoloured, re-lettered, simplified, "
        "traced or reinterpreted, and no substitute mark invented. State its "
        "position prominently and the clear space around it. Do NOT describe what "
        "the logo looks like: describing it invites the model to draw that "
        "description instead of using the file."),
    "hero": (
        "A hero image is supplied. Refer to it only as 'the attached photograph'. "
        "Say plainly that it is reproduced exactly as supplied AND that the "
        "surrounding canvas is a seamless continuation of it — its own background "
        "extended outward with matching light, grade and grain, so the finished "
        "graphic reads as a single photograph rather than a photograph placed on "
        "a design. Then FORBID, in the prompt itself and in these words, any "
        "frame, border, inset card, polaroid edge, rounded-rectangle crop, drop "
        "shadow or separate background scene around it. Never describe its "
        "contents."),
    "product": (
        "A product image is supplied. Refer to it only as 'the attached product'. "
        "State that it appears exactly as supplied with no invented features, and "
        "that it is integrated into the scene — standing on the surface, in the "
        "scene's own light, with a real contact shadow — never framed, boxed or "
        "cut out onto a backdrop. Never describe or embellish the product "
        "itself."),
    "layout": (
        "A layout reference is supplied. State explicitly that ONLY its spatial "
        "arrangement is reused — the placement of masses, the type column, the "
        "reading order — and that its subject, palette and content are NOT to "
        "appear. The graphic depicts this brief's subject, in that arrangement."),
    "style": (
        "A style reference is supplied. State explicitly that ONLY its visual "
        "treatment is matched — palette, light, texture, finish — and that its "
        "subject and layout are NOT to appear."),
    "font": FONT_ROLE,
}


def normalise(refs) -> list[dict]:
    """Accept a path, (path, role), or {path, role} and return typed entries."""
    from pathlib import Path
    out = []
    for r in refs or []:
        if isinstance(r, dict):
            path, role = r.get("path"), (r.get("role") or "hero").lower()
        elif isinstance(r, (tuple, list)) and len(r) == 2:
            path, role = r[0], str(r[1]).lower()
        else:
            path, role = r, "hero"
        if role not in ROLES:
            role = "hero"
        p = Path(path)
        if p.exists():
            out.append({"path": p, "role": role})
    return out


def block(refs: list[dict], table: dict) -> str:
    """The instruction block for one node, covering every attached reference."""
    if not refs:
        return ("NO REFERENCE IMAGERY SUPPLIED — all imagery is generated.")
    lines = []
    for i, r in enumerate(refs, 1):
        lines.append(f"REFERENCE {i} — {r['role'].upper()}: {table[r['role']]}")
    return "\n".join(lines)


def summary(refs: list[dict]) -> str:
    if not refs:
        return "none"
    return ", ".join(f"{r['role']} ({r['path'].name})" for r in refs)
