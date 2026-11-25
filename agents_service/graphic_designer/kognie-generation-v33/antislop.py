"""SLOP GUARD for the v3.2 engine: pre-generation rules against the tells that make a
graphic read as AI slop.

No model call: `build()` is plain code (well under a millisecond), so the guard adds no
wait. It returns short instruction blocks for the TYPOGRAPHER, ART DIRECTOR, COMPOSITOR
and DIRECTOR, and a negative list that run() appends to the Director's own.

What it targets: the AESTHETIC defaults of image generators (glow and gradients, WordArt,
sparkles, filler ornaments, invented extra text, plastic skin, clutter, catalogue people),
picked from the AI-slop checklist (core v1.5) and from the tells the checker found on
Kognie graphics. The engine's per-format `defect_tags` already cover the ARTIFACTS
(garbled text, extra fingers, melted objects), so this does not repeat them.

Rules are defaults to avoid, not a style: when the brief or the chosen style asks for one
of these things ("neon rim light", "confetti", "gold foil lettering"), that rule steps
aside. Only affirmative mentions count. A style's "Avoid:" line and phrases like "no
gradients" never switch a rule off.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TYPE_WORDS = r"(?:text|type|typography|letter\w*|headline|title|font|word\w*|calligraph\w*|display|heading)"


@dataclass(frozen=True)
class Rule:
    id: str                 # short name, used in logs
    checklist: str          # AI-slop checklist items it prevents
    to: tuple               # which prompts get the instruction: typo, art, comp, director
    say: str                # the instruction to the specialists, phrased as what to do
    short: str              # the same in a line, for the DIRECTOR
    ban: tuple              # negative-prompt terms
    allow: str = ""         # regex: an affirmative mention in the brief switches the rule off
    allow_near_type: str = ""  # regex: switches it off only when said about the lettering


RULES = (
    Rule("type_effects", "G4", ("typo", "director"),
         "Type is set as flat, solid colour. No bevel, extrusion, chrome, metallic or gold-foil "
         "effect, glow, gradient fill, or heavy outline with a drop shadow on the lettering.",
         "type in flat solid colour, no 3D, metallic, glow, gradient or shadow effects",
         ("3D or bevelled text", "metallic or gold-foil text", "glowing or gradient-filled text",
          "soft or drop shadows behind text"),
         allow=r"\b(foil\w*|chrome|emboss\w*|extrud\w*|word ?art|comic|pop[\s-]?art|sticker\w*|"
               r"offset shadows?)\b",
         allow_near_type=r"\b(gold\w*|metallic|glow\w*|outline\w*|shadow\w*|3d|gradient\w*|neon)\b"),
    # The Art Director is where scene writing gets invented ("a monitor showing billing
    # dashboards", "a notebook of handwritten notes"); the renderer turns it into gibberish.
    Rule("extra_text", "G10 G9 G1", ("typo", "art", "director"),
         "The only words, numbers or symbols in the image are the given text. No extra labels, "
         "captions, stickers, pills, tags, annotations, diagrams, engraved or printed numbers, "
         "fake UI or decorative pseudo-text. Do not describe writing in the scene: screens and "
         "phones show soft abstract colour or are angled away (no dashboards, charts, code or app "
         "lists), notebooks and papers are closed or blank, equipment carries no numbers or "
         "markings, and signs, packaging and book spines are blank or turned away, except for "
         "a supplied logo where the branding rule allows it.",
         "no text beyond the given strings; screens show abstract colour or face away, notebooks "
         "closed, no charts, code, handwriting or numbers anywhere in the scene",
         ("extra text", "stray labels or captions", "decorative pseudo-text",
          "readable screens, dashboards or code", "handwritten notes", "numbers on equipment"),
         allow=r"\b(screenshots?|user interface|ui|dashboards?|mock-?ups?|handwrit\w*|"
               r"show (?:the|our|an?) (?:app|screen)|editing ui)\b"),
    Rule("sparkles", "E2", ("art", "director"),
         "No sparkles, magic dust, floating particles, bokeh orbs, lens flares or light streaks "
         "added for atmosphere.",
         "no sparkles, particles, bokeh orbs or lens flares added for mood",
         ("sparkles", "floating particles", "bokeh orbs", "lens flares"),
         allow=r"\b(sparkl\w*|glitter\w*|confetti|firework\w*|bokeh|flares?|particles?|magic\w*|"
               r"fairy lights|string lights|light streaks?)\b"),
    Rule("glow_gradient", "A5", ("art", "director"),
         "Light comes only from things that emit it: no glow, bloom, halo or neon edge on objects "
         "that do not emit light. Backgrounds, panels and shapes are flat or photographic, not "
         "decorative gradients.",
         "glow only from real light sources; flat or photographic grounds, no decorative gradients",
         ("glow on non-emitting objects", "decorative gradient backgrounds"),
         allow=r"\b(neon|glow\w*|gradient\w*|aurora|luminous|bloom|halos?|light trails?|"
               r"synthwave|cyberpunk|holograph\w*)\b"),
    Rule("atmosphere", "B1", ("art",),
         "Light the scene with what it really has, in one decisive direction. No haze, fog, "
         "smoke, god-rays or rim light on every edge added only for mood.",
         "one motivated light, no haze, god-rays or rim light everywhere",
         ("atmospheric haze", "god rays"),
         allow=r"\b(smok\w*|fog\w*|mist\w*|haz[ey]|steam\w*|incense|dhunuchi|cinematic|moody|"
               r"dramatic|god[\s-]?rays?|volumetric|rim[\s-]?light\w*|backlit|silhouett\w*)\b"),
    Rule("grade", "A1 A3", ("art",),
         "Natural white balance and controlled saturation: no yellow or sepia cast, no default "
         "teal-and-orange grade, no HDR crunch with crushed blacks.",
         "natural white balance, controlled saturation, no colour cast or HDR crunch",
         ("oversaturated HDR look", "yellow colour cast", "teal and orange grade"),
         allow=r"\b(sepia|vintage|retro|faded|warm cast|film (?:look|stock|grade)|teal[\s-]and[\s-]orange|"
               r"hdr|vibrant|saturated|pop[\s-]?art|neon|cyberpunk|duotone|risograph)\b"),
    Rule("ai_palette", "A2", ("art",),
         "Build the palette from the subject, the brand or the brief. Do not fall back on the "
         "default AI palette: purple-to-blue gradients, cyan on near-black, a pastel dreamy wash.",
         "palette from the subject or brand, not purple-blue gradients or cyan on black",
         ("purple and blue gradient palette",),
         allow=r"\b(purple|violet|lilac|lavender|magenta|cyan|neon|pastel\w*|electric blue|"
               r"synthwave|cyberpunk|#[0-9a-f]{6})\b"),
    Rule("plastic", "B3 C1", ("art",),
         "Photographic surfaces keep real texture: skin with pores and fine lines, fabric weave, "
         "worn edges, highlights that fit each material. Nothing waxy, airbrushed or plastic-glossy.",
         "real texture on skin, fabric and materials, nothing waxy or plastic-glossy",
         ("plastic sheen", "airbrushed skin"),
         allow=r"\b(3d|claymation|plasticine|matte clay|clay (?:render|model|figure|character)s?|cgi|"
               r"render\w*|plastic|toy\w*|gloss\w*|inflatable|pixar|vinyl|isometric|"
               r"illustrat\w*|vector|cartoon\w*)\b"),
    Rule("render_style", "E4", ("art",),
         "Match the rendering style to the brief. Do not default to glossy Pixar-style 3D, "
         "claymation, toy or action-figure looks, or fantasy-epic treatments.",
         "no default Pixar-style 3D, claymation or fantasy-epic look",
         ("Pixar-style 3D render",),
         allow=r"\b(3d|claymation|plasticine|clay (?:render|model|figure|character)s?|pixar|cartoon\w*|"
               r"toy\w*|animat\w*|fantasy|epic|isometric|cgi|"
               r"inflatable|render\w*)\b"),
    Rule("floating", "D5", ("art", "comp"),
         "Ground the subject: it sits in a real setting or on a deliberate surface with a contact "
         "shadow, never floating on abstract swirls, gradients or particle clouds.",
         "the subject is grounded with a contact shadow, not floating in a void",
         ("subject floating in an abstract void",),
         allow=r"\b(float\w*|levitat\w*|surreal\w*|abstract|zero[\s-]?gravity|exploded view|glow fields?)\b"),
    Rule("decor", "H3 H5 H6 I1", ("art", "director"),
         "No decorative filler or template ornaments: swooshes, blobs, waves, dot grids, ribbons, "
         "seals, shields, starbursts, 'guaranteed' badges, brush-stroke banners, emoji or "
         "clip-art icons.",
         "no filler ornaments: swooshes, blobs, ribbons, seals, starbursts, brush banners, emoji, clip-art",
         ("decorative swooshes and blobs", "ribbons, seals and badges", "starbursts", "brush-stroke banners", "emoji", "clip-art icons"),
         allow=r"\b(memphis|squiggl\w*|confetti|ribbons?|seals?|badges?|stamps?|stickers?|starbursts?|"
               r"sunbursts?|emoji\w*|icons?|brush[\s-]?strokes?|brush letter\w*|doodl\w*|ornament\w*|pop[\s-]?art|collage|torn|"
               r"tape|fan motifs?|geometric shapes|marginalia)\b"),
    Rule("clutter", "D1 D2 N2", ("comp", "director"),
         "Restraint: the subject, the text and at most one supporting graphic device, with "
         "generous negative space and one dominant focal point. Do not stack every device of "
         "the genre (banner, icon row, giant percentage, countdown, footer strip) at once.",
         "subject, text and at most one supporting device; generous space, one focal point",
         ("cluttered layout",),
         allow=r"\b(maximal\w*|busy|collage|memphis|scrapbook|layered ephemera|pattern[\s-]filled)\b"),
    Rule("people", "F1 F2 F3 F5 D4", ("art", "director"),
         "If people appear: individual faces with natural asymmetry and real expression, never "
         "identical model faces, matching poses or a row of smiles at the camera. Bodies hold "
         "relaxed, physically possible poses: joints bend the way they really do, feet and "
         "ankles sit naturally, no limb twisted or stretched for drama, no extreme freeze-frame "
         "action. Hands are simple and relaxed, or out of frame; avoid complex hand poses and "
         "interlocked fingers. Background people are soft-focus or turned away, never a "
         "detailed crowd of faces.",
         "individual faces, natural relaxed poses with joints bending as bodies do, simple hands, "
         "soft-focus background people",
         ("identical model faces", "matching poses", "complex hand gestures", "detailed crowd faces",
          "limbs or ankles bent at unnatural angles")),
    Rule("objects", "C3 E5 L2", ("director",),
         "Every object is whole and physically plausible: parts connect, mechanisms work, nothing "
         "is fused or melting. Repeated items vary naturally instead of being cloned. When the "
         "text states a number of items, show exactly that many, or do not show a countable set.",
         "objects whole and plausible, repeats varied not cloned, stated counts shown exactly",
         ("cloned identical objects",)),
    Rule("cliches", "E1 E3", ("art",),
         "Show the real subject, not a stock metaphor for it: no glowing brains, circuit traces, "
         "floating holographic UI, robot hands, rockets, lightbulbs, puzzle pieces, gears or nebulae.",
         "the real subject, never a stock metaphor (glowing brain, circuits, holograms, rockets, lightbulbs)",
         ("glowing brain", "circuit board overlay", "hologram", "robot hand", "rocket", "lightbulb", "puzzle pieces", "gears")),
    # v2 (3 Nov 2026): the hero image read as AI at a glance even with no defect, because
    # it was too perfect: lustrous gold, flawless croissants, a frozen splash, amber light.
    Rule("hyperreal", "C6 N4", ("art", "director"),
         "If the hero image is a photograph, it looks like a good ordinary one, not a flawless "
         "commercial render. Surfaces are mostly matte, with a few small highlights only where light really hits; "
         "metal and gold read as metal with soft reflections and some wear, never mirror-bright "
         "or glittering all over. Real imperfection is welcome: uneven browning, a stray crumb, a "
         "crease, dust, a scuff, items that are not identical or perfectly aligned. Show the "
         "product at an ordinary viewing distance, not a macro close-up where every detail is "
         "pin-sharp; natural depth of field and ordinary sharpness. No frozen high-speed "
         "splashes or droplets unless the brief asks for them.",
         "if photographic, an ordinary good photograph, not a flawless render: mostly matte, few small highlights, "
         "metal not mirror-bright, small real imperfections, product at normal distance not macro, "
         "no frozen splashes",
         ("hyper-realistic commercial render", "over-polished glossy surfaces", "mirror-bright metal",
          "frozen high-speed splash", "flawless identical items", "macro product close-up"),
         allow=r"\b(hyper[\s-]?real\w*|glossy|high[\s-]?shine|cgi|3d|render\w*|mirror[\s-]finish|"
               r"wet[\s-]look|splash\w*|droplets?|illustrat\w*|vector|cartoon\w*|drawn|doodl\w*|"
               r"painted|watercolou?r|risograph|halftone|collage|pop[\s-]?art|flat[\s-]illustration)\b"),
    Rule("light_temperature", "A1 B1", ("art", "director"),
         "Light is neutral and natural by default: daylight from a window or open shade, white "
         "balance around 5000-6500K. Lamps, candles or diyas in shot may glow as small points, "
         "but they do not light the scene and are not added to justify warm light: the scene "
         "stays daylight-neutral, never an amber or golden wash. No spotlight pools, rim light "
         "or studio drama.",
         "neutral daylight lights the scene; lamps, candles or diyas may glow in shot but never "
         "light it or wash it amber; no rim light",
         ("amber colour wash", "golden glow over everything", "spotlit studio drama"),
         allow=r"\b(golden[\s-]hour|sunset|sunrise|warm (?:light\w*|tones?|glow|palette|grade)|"
               r"tungsten|firelight|fireplace|cos[yz]|lanterns?|amber|sepia|retro|vintage|"
               r"dramatic|moody|night\w*)\b"),
    Rule("simple_scene", "D1 N2", ("art", "comp", "director"),
         "Keep the scene simple: the subject in a real, uncluttered setting with at most two or "
         "three props the brief motivates. Do not dress every surface (shelves, baskets, lamps, "
         "garnish, scattered petals) to make it look rich.",
         "a simple, real setting: the subject and at most two or three motivated props",
         ("over-styled set", "props on every surface"),
         allow=r"\b(maximal\w*|busy|collage|scrapbook|flat[\s-]?lay|clutter\w*|crowded|layered|"
               r"abundan\w*|market stall)\b"),
)

# With the guard on, these replace the ART DIRECTOR's LIGHT and MATERIAL guidance. The
# originals ask for "hard low sun", "a crisp rim", "real speculars", metal that "throws
# speculars", "condensation beads", "oil sheens": a recipe for the hyper-real look.
LIGHT_CRAFT = (
    "LIGHT — natural and believable, as a good photographer would find it on location. Default "
    "to neutral daylight: a window from one side or open shade, white balance around 5000-6500K. "
    "Lamps, candles or diyas in shot may glow as small points, but they do not light the scene "
    "and are never added just to justify warm light. One clear "
    "direction with soft, plausible shadows. No rim light on every edge, no spotlight pools, no "
    "glowing backlight. Flat, shadowless light is also a tell: aim for ordinary daylight with shape.")
MATERIAL_CRAFT = (
    "MATERIAL — describe surfaces as they look in an ordinary good photograph, not a commercial "
    "render: matte where the material is matte, with highlights small, few and only where light "
    "really hits. Metal and gold read as metal, with soft reflections and a little wear, never "
    "mirror-bright. Food has uneven browning and crumbs; fabric creases; wood is worn; skin has "
    "texture. Small imperfections are what make an image read as real.")
LIGHT_SCHEMA = "<direction, quality, colour temperature in K, and what real source motivates it>"

# v3 (4 Nov 2026): SUBJECT runs first and every later node treats its scene as settled, so
# this is where a scene is kept simple. In the sneaker runs it asked for "macro", "frozen
# droplets" and "rim lighting" before any guard had spoken.
SUBJECT_PLAIN = (
    "  - If this is a photograph, pick an ordinary real moment at an ordinary distance, one a "
    "photographer could shoot on an ordinary day. Not peak action, not a macro view, not a "
    "freeze-frame.\n"
    "  - hero_scene: at most two short, plain sentences: what is in the frame and where. Leave "
    "light, mood and adjectives of quality out.\n"
    "  - authentic_details: only 3 to 5 things a viewer can see at that distance, in everyday "
    "words. No engineering or spec-sheet vocabulary (foam chemistry, plate materials, anatomy "
    "terms): the generator cannot draw those and answers with hyper-detailed texture.\n"
    "  - hero_moment and platform_crop: one short sentence each.\n")
# The goal is a graphic free of slop, not a realistic one. Without this every specialist
# assumes a photograph, because their prompts talk about photographs.
SUBJECT_MEDIUM = (
    "  - The image can be any medium: a photograph, an illustration, a type-led poster, a collage, "
    "3D. Use the one the brief names. If it names none, use whichever makes this graphic strongest "
    "and say which in hero_scene. It never has to look hyper-real.\n")
# A printed bill or a dashboard invites the generator to fill it with false words and figures.
SUBJECT_NO_WRITING = (
    "  - Do not build the scene around something with readable writing (a printed bill, a page of "
    "figures, a dashboard, a screen): generators fill it with false text. Show it closed, turned "
    "away or too far off for words to form, or show something else.\n")
ART_MEDIUM = ("  - Keep the subject's medium or choose a stronger one, and name it in your scene. A "
              "graphic that is not a photograph is welcome when it suits the goal.")

# Anatomy is decided here as content, not left to a ban: the sneaker prompts never mentioned a
# leg, yet "explosive toe-off" needs one, so the generator invented it with nothing to go on.
SUBJECT_WHO = (
    "  - Decide who is in the frame and set \"in_frame\". Show a person or a body part when the "
    "brief asks for one or when it makes this graphic better. It is never required: a product "
    "can stand alone.\n"
    "  - \"none\": the scene must make sense with no body in it. An action that needs a body "
    "(mid-stride, a hand pouring) means \"part\" or \"people\" instead: a body nobody described "
    "gets invented by the generator, which is where morphed legs and ankles come from.\n"
    "  - \"part\": name the one body part, how it is cropped and the ordinary thing it is doing.\n"
    "  - \"people\": as few people as the scene needs, at an ordinary distance.\n"
    "  - Any body stays in a natural position a real person holds without effort: standing, "
    "walking, an easy jog, holding something simply.\n"
    "  - If a body is in frame, list in \"body_parts\" only the parts clearly visible.\n")
# The views that failed in the anatomy experiments (6 Nov 2026): a flat hand seen from straight
# above put the thumb in the finger row, and feet seen from straight down lost their ankle bones.
BODY_VIEWS = (
    "  - When a body is in frame, unless the brief asks otherwise use a view generators draw well: "
    "a hand with a slight natural curve and its thumb clearly apart from the fingers, not flat from "
    "straight above; feet and ankles from the side or at three-quarter view, not straight down; "
    "one or two people rather than a group.\n")
PEOPLE_FIELD = ('  "in_frame": "<none | part | people>",\n'
                '  "body_parts": ["<those clearly visible: hands, feet, legs, arms, face, mouth, ears>"],\n')
ART_WHO = ("  - The subject proposes \"in_frame\". Keep it or change it, whichever makes the better "
           "graphic, and set your own \"in_frame\". If a body is in frame, describe it plainly in "
           "your scene: which part, how it is cropped, its natural position. List the parts "
           "clearly visible in \"body_parts\".")

# Added to the final prompt in code, from the specialists' decision, so it reaches the renderer
# whatever the Director wrote. Each line states the structure the experiments got wrong. Lines
# name only the parts in this image, and a stylised figure gets one line instead.
IN_FRAME = {
    "none": ("No person or body part appears anywhere in the image.",
             ("people", "human body parts", "human hands or feet")),
    "part": ("The body in frame looks completely normal: natural bone structure and proportions, "
             "in a relaxed position.", ()),
    "people": ("People look completely normal: natural bone structure and proportions, in relaxed "
               "natural poses.", ("matching poses", "matching clothes or accessories")),
}
_BODY_BANS = ("morphed or lumpy body parts", "pale smears or ridges on skin",
              "bags or jewellery the brief did not ask for")
# In priority order: at most MAX_PARTS of them are written out, to keep the prompt short.
PARTS = {
    "hands": ("Each hand has four fingers and a shorter thumb that branches near the wrist and "
              "angles away from the fingers; knuckles vary.",
              ("extra or missing fingers", "a thumb lying in line with the fingers", "fused fingers")),
    "feet": ("Ankles narrow above the heel with a clear ankle bone on each side, and skin over the "
             "foot is smooth.",
             ("thick ankles with no ankle bones", "ridged or lumpy skin on the feet")),
    "mouth": ("Teeth are separate and even, in a natural arc; lips are smooth.",
              ("too many teeth", "teeth blurred into one band", "beads or bumps on the lips")),
    "face": ("Eyes match, with round irises and centred pupils; lips are smooth.",
             ("mismatched eyes", "off-centre pupils", "beads or bumps on the lips")),
    "ears": ("Ears have a clear rim and inner fold, even out of focus.", ("melted or shapeless ears",)),
    "legs": ("Legs taper from calf to ankle and knees bend the right way.",
             ("limbs of the wrong length or thickness",)),
    "arms": ("Arms taper to the wrist and bend only at the elbow.",
             ("limbs of the wrong length or thickness",)),
}
MAX_PARTS = 3
TOES = ("Each foot has five toes, the big toe largest.", "extra or missing toes")
_BARE_FEET = re.compile(r"\b(?:bare\s?f(?:oo|ee)t|barefoot|sandals?|flip-?flops?|toes?)\b", re.I)
CONTACT = ("Anything worn or held sits as it really would: straps go round the limb, rings circle the "
           "finger, fingers touch what they grip.")
_HELD = re.compile(r"\b(?:hold\w*|wear\w*|worn|grip\w*|pull\w*|strap\w*|rings?|watch\w*|bracelets?|"
                   r"earrings?|sandals?|boots?|shoes?|bags?)\b", re.I)
DRAWN_FIGURE = ("Figures are stylised on purpose and consistently: the right number of fingers and "
                "limbs, hands drawn simply with fingers clearly separated, joints bending the right way.")
_DRAWN_BANS = ("extra or missing fingers", "tangled fingers", "extra limbs")

# Papers, screens and signs in the scene: stated in code because the TAILOR can leave the terms out.
WRITING_LINE = "Papers, screens and signs show no readable words or numbers: only soft grey lines or plain colour."
WRITING_BAN = "readable words or numbers on papers or screens"
# Paper that carries no writing: the worked example fired on "greaseproof paper".
_PLAIN_PAPER = re.compile(r"\b(?:greaseproof|parchment|baking|tissue|wrapping|kraft|butcher|rice|wax)[ -]?papers?\b|"
                          r"\bpaper (?:bags?|cups?|napkins?|straws?|towels?|lanterns?|planes?|cranes?)\b", re.I)
_WRITING = re.compile(r"\b(?:invoices?|bills?|receipts?|documents?|spreadsheets?|reports?|papers?|"
                      r"pages?|menus?|notebooks?|newspapers?|magazines?|screens?|monitors?|laptops?|"
                      r"phones?|tablets?|dashboards?|charts?|signs?|signage|posters?|billboards?)\b", re.I)

_PART_WORDS = {
    "hands": r"(?<!robot )hands?|fingers?|thumbs?|knuckles?|wrists?|palms?",
    "feet": r"feet|foot|toes?|ankles?",
    "mouth": r"mouth|lips?|teeth|tooth|smil\w+",
    "face": r"faces?|eyes?|cheeks?|jaw\w*|chin|forehead|portrait|profile",
    "ears": r"ears?|earlobes?|earrings?",
    "legs": r"legs?|knees?|calf|calves|shins?",
    "arms": r"arms?|forearms?|elbows?",
}
_PART_RES = {k: re.compile(rf"\b(?:{v})\b", re.I) for k, v in _PART_WORDS.items()}
# Words that look like a body but are not one. Quoted strings are the graphic's own text.
_NOT_BODY = re.compile(
    r"\"[^\"\n]*\"|\u201c[^\u201d\n]*\u201d|\bhand-\w+|\bon the other hand\b|"
    r"\b(?:hour|minute|second|watch|clock)[ -]hands?\b|"
    r"\b(?:watch|clock|type|rock|cliff|dial|building)[ -]?faces?\b|\bfacing\b|\beye[- ]level\b|"
    r"\bbird'?s[- ]eye\b|\bfoot of\b|\bfoot anchor\b|\btoe[- ](?:box|cap)\b|\bprofile (?:of|view of) "
    r"(?:the|a|an) (?:shoe|bottle|building|product)\b", re.I)
_SCENE_LINE = re.compile(r"(?ims)^scene:\s*(.*?)(?=^[a-z]+:|\Z)")


def _scene(final_prompt: str) -> str:
    """The Scene line, or the whole prompt if it has none, without the graphic's own words."""
    m = _SCENE_LINE.search(final_prompt or "")
    return _NOT_BODY.sub(" ", m.group(1) if m else (final_prompt or ""))


def shows_writing(final_prompt: str, active: list) -> bool:
    return "extra_text" in active and bool(_WRITING.search(_PLAIN_PAPER.sub(" ", _scene(final_prompt))))

# v3.1: the TAILOR node, one fast call after the Director. It picks the few negatives and
# positives this scene needs, so a shoe ad is not told about glowing brains. These stay
# whatever it picks: the content controls (garbled text, invented branding, CTAs).
_ALWAYS_KEEP = re.compile(r"warped text|misspell|gibberish|malformed letter|doubled letter|nonsense "
                          r"char|garbled|logo|wordmark|trademark|brand|product label|call-to-action|"
                          r"button|shop now|extra text|stray label", re.I)
TAILOR_MAX_NEGATIVES = 8
TAILOR_MAX_POSITIVES = 3

# v3.2 (6 Nov 2026): the lists were still 47-52 terms after tailoring, because these content
# controls arrive as 20-odd near-synonyms and all of them were protected. In code, before the
# TAILOR: each group becomes one term, so the same control costs a few words.
_COMPACT = (
    ("text", re.compile(r"warped text|misspelled words|gibberish|malformed letter|doubled letter|"
                        r"nonsense char|garbled (?:text|letter)", re.I)),
    ("brand", re.compile(r"\blogos?\b|wordmark|trademark|\bbrand\w*|product labels?|manufacturer name|"
                         r"invented (?:product names|origins|specifications)|tasting notes", re.I)),
    ("cta", re.compile(r"call-to-action|calls? to action|\bbuttons?\b|pill shapes?|shop now|buy now|"
                       r"order now|learn more|sign up", re.I)),
    ("extra", re.compile(r"extra text|stray labels?|captions|pseudo-text", re.I)),
)
COMPACT_TERMS = {"text": "garbled or misspelled text", "extra": "extra text or labels",
                 "cta": "buttons or call-to-action text"}
_BRAND_TERM = {False: "invented logos or brand names or product labels",
               True: "altered or invented logos or invented label text"}
# Rule terms that only make sense for several people, or for a photograph.
_GROUP_TERM = re.compile(r"\b(?:cloned|identical|matching|crowd)\b", re.I)
_BODY_TERM = re.compile(r"\b(?:skin|body|bodies|limbs?|poses?|anatom\w*)\b", re.I)
_LIMBS = re.compile(r"\blimbs?\b", re.I)
# Per-format defect tags that the part terms above state more exactly.
_SUPERSEDED = re.compile(r"^(?:distorted )?(?:extra fingers|missing fingers|malformed hands|fused fingers|"
                         r"asymmetric eyes|distorted teeth|deformed ears)$", re.I)
_PHOTO_ONLY = re.compile(r"hyper-realistic|over-polished|mirror-bright|high-speed splash|flawless "
                         r"identical|macro product|plastic sheen|airbrushed|waxy|over-smoothed", re.I)

# Short positive reminders the TAILOR can pick from, by the rule that allows them.
POSITIVES = {
    "type_effects": "type in flat solid colour with no shadow behind it",
    "extra_text": "screens, papers and signs in the scene are blank or turned away",
    "glow_gradient": "light comes only from real light sources",
    "grade": "natural white balance and controlled saturation",
    "plastic": "real texture on skin, fabric and materials",
    "floating": "the subject sits on a real surface with a contact shadow",
    "clutter": "one focal point with generous space around it",
    "objects": "objects whole and physically plausible, repeated items varied",
    "hyperreal": "small real imperfections, highlights small and few",
    "light_temperature": "neutral natural light",
    "simple_scene": "a simple real setting with few props",
}

TAILOR = """You choose the few reminders an image generator needs for ONE prompt. Long lists overload it,
and naming something irrelevant can put it in the picture.

THE PROMPT:
\"\"\"{prompt}\"\"\"

CANDIDATE NEGATIVES (things to keep out):
{negatives}

CANDIDATE POSITIVES (short reminders to add):
{positives}

Choose at most {max_negatives} negatives: only mistakes this particular scene is likely to produce. Leave
out anything with no connection to this scene.
Choose at most {max_positives} positives: only what this scene most needs and the prompt does not already say.
Choosing fewer, or none, is right when little applies.

Return ONLY strict JSON: {{"negatives": [<numbers>], "positives": [<numbers>]}}"""


def tailor_inputs(negative_prompt: str, active: list, protect: list, drawn: bool = False,
                  writing: bool = False) -> dict:
    """Split a merged negative list into what always stays and what the TAILOR may pick from."""
    keep, free = [], []
    for term in (x.strip() for x in (negative_prompt or "").split(", ")):
        if not term:
            continue
        (keep if (_ALWAYS_KEEP.search(term) or term in protect) else free).append(term)
    keep += [t for t in protect if t not in keep]
    # A photograph's reminders are not a drawing's, and code has already said the writing line.
    skip = (("hyperreal", "plastic") if drawn else ()) + (("extra_text",) if writing else ())
    return {"keep": keep, "free": free,
            "positives": [POSITIVES[r] for r in active if r in POSITIVES and r not in skip]}


def tailor_prompt(final_prompt: str, inputs: dict) -> str:
    numbered = lambda items: "\n".join(f"  {i}. {x}" for i, x in enumerate(items, 1)) or "  (none)"  # noqa: E731
    return TAILOR.format(prompt=final_prompt, negatives=numbered(inputs["free"]),
                         positives=numbered(inputs["positives"]),
                         max_negatives=TAILOR_MAX_NEGATIVES, max_positives=TAILOR_MAX_POSITIVES)


def apply_tailor(final_prompt: str, negative_prompt: str, inputs: dict, data) -> tuple[str, str, bool]:
    """The prompt and negatives after the TAILOR's choice. The numbers can only select from the
    candidates, so it cannot invent a term. An unusable answer changes nothing."""
    if not (isinstance(data, dict) and isinstance(data.get("negatives"), list)
            and isinstance(data.get("positives"), list)):
        return final_prompt, negative_prompt, False

    def pick(numbers, items, cap):
        chosen = sorted({n for n in numbers if isinstance(n, int) and 1 <= n <= len(items)})[:cap]
        return [items[n - 1] for n in chosen]
    negatives = inputs["keep"] + pick(data["negatives"], inputs["free"], TAILOR_MAX_NEGATIVES)
    positives = pick(data["positives"], inputs["positives"], TAILOR_MAX_POSITIVES)
    if positives:
        final_prompt = f"{final_prompt}\n\nAlso: {'; '.join(positives)}."
    return final_prompt, ", ".join(negatives), True

# Words that push a render toward the hyper-real look, toned down in the Director's final
# prompt (plain code, no model call). Each replacement keeps the sentence grammatical.
HYPE = {
    "hyper-realistic": "natural", "hyperrealistic": "natural", "hyper-real": "natural",
    "ultra-realistic": "natural", "ultra-detailed": "detailed", "ultra-sharp": "sharp",
    "razor-sharp": "sharp", "flawless": "clean", "pristine": "clean", "immaculate": "neat",
    "glistening": "moist", "gleaming": "soft-sheened", "lustrous": "soft-sheened",
    "glittering": "softly reflective", "opulent": "rich", "mirror-polished": "polished",
    "high-gloss": "satin", "8k": "high-resolution", "production-quality": "finished",
    "luster": "soft sheen", "lustre": "soft sheen", "specular pings": "small highlights",
    "specular glints": "small highlights", "specular highlights": "small highlights",
    "crystalline": "fine", "hyper-detailed": "detailed",
    "rim-light": "side-light", "rim-lit": "side-lit", "rim light": "side light",
    "rim lighting": "side lighting", "glinting": "catching the light",
}
_SOFTEN_RE = re.compile(r'("[^"\n]*"|\u201c[^\u201d\n]*\u201d)|\b('
                        + "|".join(re.escape(k) for k in sorted(HYPE, key=len, reverse=True)) + r")\b", re.I)


# Appended to the final prompt in code when the hero is a photograph and the hyperreal rule
# is on, so it reaches the renderer whatever the Director wrote.
PHOTO_REGISTER = (
    "An unretouched photograph with natural colour, ordinary sharpness and mild grain; surfaces "
    "mostly matte, with the small imperfections of real objects. Not a studio product shot or a "
    "commercial render.")


# Directors often describe a photo without the word: "flash capture", "low-angle shot", "strobe".
_PHOTO_RE = re.compile(r"\b(?:photo\w*|captur\w*|shot|camera|strobe|lens|shutter)\b", re.I)
# Not plain "drawn" or "painted": quotation marks are "drawn" and walls are "painted" in photos.
_DRAWN_RE = re.compile(r"\b(?:illustrat\w*|vector|cartoon|hand-drawn|line drawing|painting of|"
                       r"watercolou?r|3d render|gouache|linocut|woodblock|risograph|paper[- ]cut|"
                       r"collage|ink drawing|type-led|typographic poster)\b", re.I)


def is_drawn(final_prompt: str) -> bool:
    """The Director described something other than a photograph."""
    return bool(_DRAWN_RE.search(final_prompt or ""))


def photo_register(final_prompt: str, active: list) -> str:
    text = final_prompt or ""
    if "hyperreal" in active and _PHOTO_RE.search(text) and not _DRAWN_RE.search(text):
        # A night or neon brief switches off light_temperature, so it must not be told "daylight".
        daylight = " Lit by available daylight." if "light_temperature" in active else ""
        return f"{final_prompt}\n\n{PHOTO_REGISTER}{daylight}"
    return final_prompt


def in_frame(*deciders) -> str | None:
    """Who is in the frame: the first of these outputs to decide (ART DIRECTOR, then SUBJECT)."""
    for decider in deciders:
        words = re.findall(r"\w+", str((decider or {}).get("in_frame") or "").lower())
        who = next((w for w in words if w in IN_FRAME), None)
        if who:
            return who
    return None


def body_parts(who: str | None, final_prompt: str, *deciders) -> list:
    """The parts visible, most at risk first: what the specialists listed plus what the Scene line
    names. Empty when no body is in frame."""
    if who not in ("part", "people"):
        return []
    named = " ".join(str(x) for d in deciders for x in ((d or {}).get("body_parts") or [])
                     if isinstance(d, dict) and isinstance(d.get("body_parts"), list))
    text = f"{named} {_scene(final_prompt)}"
    found = [p for p in PARTS if _PART_RES[p].search(text)]
    # "The woman smiled" without a listed part still has a face, and people usually have hands.
    return found or (["hands", "face"] if who == "people" else [])


def anatomy(who: str, parts: list, drawn: bool, final_prompt: str = "") -> tuple[str, list]:
    """The in-frame line and its negatives for this image."""
    line, bans = IN_FRAME[who]
    if who == "none":
        return line, list(bans)
    if drawn:
        return DRAWN_FIGURE, list(_DRAWN_BANS)
    named, scene = parts[:MAX_PARTS], _scene(final_prompt)
    lines, out = [line], list(_BODY_BANS) + list(bans)
    if "feet" in named and _BARE_FEET.search(scene):
        lines.append(TOES[0])
        out.append(TOES[1])
    for p in named:
        lines.append(PARTS[p][0])
        out += [b for b in PARTS[p][1] if b not in out]
    if who == "part" or _HELD.search(scene):
        lines.append(CONTACT)
    return " ".join(lines), out


def missing_text(final_prompt: str, locked: dict | None) -> list:
    """Locked strings the Director's prompt lost. The renderer cannot draw what it is not given.
    Checked line by line and between separators, because a details line may be stacked."""
    flat = lambda t: re.sub(r"\s+", " ", str(t).translate(_STRAIGHT)).strip().lower()  # noqa: E731
    have, lost = flat(final_prompt or ""), []
    for value in (locked or {}).values():
        for piece in re.split(r"\n+|\s+[\u00b7|]\s+", str(value)):
            if piece.strip() and flat(piece) not in have:
                lost.append(piece.strip())
    return lost


_STRAIGHT = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})


def finish(final_prompt: str, guard: dict, who: str | None, parts: list | None = None,
           locked: dict | None = None) -> tuple[str, list, list]:
    """The guard's code step on a final prompt: tone down hype words, restore any lost text, add
    the photographic register and the in-frame line. Returns the prompt, what was softened and
    any negatives. Guard off: unchanged."""
    if not guard["active"]:
        return final_prompt, [], []
    text, softened = soften(final_prompt)
    lost = missing_text(text, locked)
    if lost:
        text += "\n\nThe text also includes, exactly: " + "; ".join(f'"{x}"' for x in lost) + "."
    text = photo_register(text, guard["active"])
    writing = shows_writing(final_prompt, guard["active"])
    if writing:
        text += f"\n\n{WRITING_LINE}"
    if not who:
        return text, softened, [WRITING_BAN] if writing else []
    if parts is None:
        parts = body_parts(who, final_prompt)
    line, bans = anatomy(who, parts, is_drawn(final_prompt), final_prompt)
    return f"{text}\n\n{line}", softened, bans + ([WRITING_BAN] if writing else [])


def soften(final_prompt: str) -> tuple[str, list]:
    """Tone down hype words in a final prompt. Returns the prompt and what was changed."""
    changed = []

    def sub(m):
        if m.group(1):  # a quoted string is the user's copy: it stays exact
            return m.group(1)
        word = m.group(2)
        new = HYPE[word.lower()]
        changed.append(f"{word} -> {new}")
        return new.capitalize() if word[0].isupper() else new
    text = _SOFTEN_RE.sub(sub, final_prompt or "")
    if changed:  # "an opulent" -> "an rich": every replacement starts with a consonant
        text = _AN_RE.sub(r"\1 ", text)
        text = _TWICE_RE.sub(r"\1", text)  # "finished, production-quality" -> "finished, finished"
    return text, changed


_AN_RE = re.compile(r"\b([Aa])n (?=(?:" + "|".join(sorted({re.escape(v.split()[0]) for v in HYPE.values()})) + r")\b)")
_TWICE_RE = re.compile(r"\b(" + "|".join(sorted({re.escape(v) for v in HYPE.values()}, key=len, reverse=True))
                       + r")(?:,? +(?:and +)?\1)+\b", re.I)

_HEAD = {
    "typo": "SLOP GUARD (typography). These are the defaults that make a graphic read as AI-made; "
            "avoid them unless the brief asks for them:",
    "art": "SLOP GUARD (imagery). These are the image generator's defaults that make a graphic read "
           "as AI-made. Avoid them unless the brief or the user's chosen style asks for them; within "
           "that, be as bold and specific as the brief allows:",
    "comp": "SLOP GUARD (layout):",
    "director": "SLOP GUARD. Carry each of these into the final prompt as a positive instruction "
                "(say what the graphic does, e.g. 'type set in flat solid colour'), not as a list of "
                "banned things. The brief outranks them: if it asks for one, it stays.",
    "director_short": "SLOP GUARD. Let these shape your scene, camera and light. Do not write them "
                      "out in the prompt: the negative list carries them, and the generator works "
                      "better from a short prompt. The brief outranks them: if it asks for one, it "
                      "stays.",
    "subject": "SLOP GUARD (subject). Keep the scene simple enough for an image generator to draw "
               "well. These override the counts and lengths in the JSON below:",
}

# Phrases that mention something in order to rule it out. Removed before matching so that a
# style's "Avoid: gold-foil WordArt" or a user's "no gradients" never switches a rule off.
_NEGATED = re.compile(r"(?im)^\s*avoid\b.*$|\b(?:no|not|without|avoid\w*|never|don'?t|do not|"
                      r"nothing|zero|minus|free of|instead of)\b[^.;:\n]*")


def affirmative(text: str) -> str:
    return _NEGATED.sub(" ", text or "").lower()


def _near_type(pattern: str, text: str) -> str | None:
    """`pattern` said about the lettering: within four words of a type word."""
    gap = r"[^\w\n]+(?:\w+[^\w\n]+){0,4}?"
    near = rf"{pattern}{gap}{_TYPE_WORDS}|{_TYPE_WORDS}{gap}{pattern}"
    m = re.search(near, text)
    return m.group(0) if m else None


_HEADING = re.compile(r"[A-Z\u00c9]{2,}(?:[ -][A-Z\u00c9]{2,})*:\s*")


def clean_negatives(text: str) -> list:
    """The terms of a Director's list, without the template's headings ("ALSO BAN:") or its
    placeholder, which would otherwise be sent to the renderer as if they were terms."""
    out, seen = [], set()
    for term in re.split(r",\s*|\n+", _HEADING.sub("", text or "")):
        term = term.strip().rstrip(".")
        if term and not term.startswith("(") and term.lower() not in seen:
            seen.add(term.lower())
            out.append(term)
    return out


def merge_negatives(director_list: str, guard_terms: list) -> str:
    """The Director's comma-separated negatives plus the guard's, without repeats."""
    terms = clean_negatives(director_list)
    joined = ", ".join(terms).lower()
    return ", ".join(terms + [t for t in guard_terms if t.lower() not in joined])


def compact_negatives(terms: list, protect: list, has_logo: bool, cta_requested: bool) -> list:
    """Each group of near-synonym content controls becomes one term. A requested CTA drops its
    group. User exclusions and the anatomy terms are left exactly as given."""
    out = []
    for term in terms:
        group = None if term in protect else next((g for g, r in _COMPACT if r.search(term)), None)
        if not group:
            out.append(term)
        elif not (group == "cta" and cta_requested):
            new = _BRAND_TERM[has_logo] if group == "brand" else COMPACT_TERMS[group]
            if new not in out:
                out.append(new)
    return out


def gate_negatives(terms: list, protect: list, who: str | None, parts: list, drawn: bool) -> tuple[list, list]:
    """What this image cannot go wrong on, removed in code before any model call. Returns what
    stays and what went. `who` None means nothing is known about bodies, so they all stay."""
    kept, dropped = [], []
    for term in terms:
        named = [p for p, r in _PART_RES.items() if r.search(term)]
        if _LIMBS.search(term):
            named += ["hands", "feet", "legs", "arms"]
        go = False
        if term not in protect:
            go = drawn and bool(_PHOTO_ONLY.search(term))
            if who == "none":
                go = go or bool(named or _BODY_TERM.search(term))
            elif who in ("part", "people"):
                go = (go or "hand gesture" in term.lower()  # the brief may ask for one; PARTS covers hands
                      or (bool(parts) and bool(_SUPERSEDED.search(term)))
                      or (who == "part" and bool(_GROUP_TERM.search(term)))
                      or bool(named and parts and not set(named) & set(parts)))
        (dropped if go else kept).append(term)
    return kept, dropped


def prepare_negatives(director_list: str, guard: dict, frame_bans: list, exclusions: list,
                      who: str | None, parts: list, final_prompt: str, has_logo: bool,
                      cta_requested: bool) -> tuple[str, list, list]:
    """Everything done to the negatives in code. Returns the list, what is protected from the
    TAILOR, and what the gate removed. Guard off: only the Director's list, cleaned."""
    if not guard["active"]:
        return merge_negatives(director_list, []), list(exclusions), []
    # They are read after "Do not include": "no people" there would say the opposite.
    plain = [re.sub(r"(?i)^(?:no|without|avoid|never)\s+", "", str(t)).strip() for t in exclusions]
    protect = [t for t in plain + list(frame_bans) if t]
    terms = clean_negatives(merge_negatives(director_list, guard["negatives"]))
    terms = compact_negatives(terms, protect, has_logo, cta_requested)
    kept, dropped = gate_negatives(terms, protect, who, parts, is_drawn(final_prompt))
    return ", ".join(kept + [t for t in protect if t not in kept]), protect, dropped


def build(brief_text: str, enabled: bool = True, short_prompt: bool = True) -> dict:
    """The guard for one brief. `brief_text`: everything the user and their chosen style
    said (raw prompt, style description, visual directives). `short_prompt`: the Director
    writes the short prompt, so its rules shape choices instead of being written out."""
    empty = {"typo": "", "art": "", "comp": "", "director": "", "negatives": [],
             "active": [], "exempted": {}, "light_craft": None, "material_craft": None,
             "light_schema": None, "subject": "", "people_field": ""}
    if not enabled:
        return empty
    said = affirmative(brief_text)
    active, exempted = [], {}
    for r in RULES:
        hit = (r.allow and re.search(r.allow, said)) or None
        hit = hit.group(0) if hit else (r.allow_near_type and _near_type(r.allow_near_type, said))
        if hit:
            exempted[r.id] = hit.strip()[:60]
        else:
            active.append(r)
    # A ban term naming something the brief asks for is dropped even when its rule stays
    # ("rocket" for a rocket company, "ribbons" for a ribbon-cutting).
    negatives = []
    for r in active:
        for term in r.ban:
            named = re.search(r"\b" + re.escape(term.lower().rstrip("s")) + r"s?\b", said)
            if not named and term not in negatives:
                negatives.append(term)
    out = dict(empty, negatives=negatives, active=[r.id for r in active], exempted=exempted)
    # The ART DIRECTOR's own light and material guidance is swapped only while the matching
    # rule is on: a brief that asks for golden hour or a glossy 3D render keeps the original.
    if "light_temperature" in out["active"]:
        out.update(light_craft=LIGHT_CRAFT, light_schema=LIGHT_SCHEMA)
    if "hyperreal" in out["active"]:
        out["material_craft"] = MATERIAL_CRAFT
    # The specialists get the rules for their own discipline in full; the DIRECTOR, who
    # compiles the final prompt and can undo any of them, gets every rule in a line.
    for role in ("typo", "art", "comp", "director"):
        lines = [f"  - {r.short}" if role == "director" else f"  - {r.say}"
                 for r in active if role == "director" or role in r.to]
        if lines:
            head = _HEAD["director_short" if role == "director" and short_prompt else role]
            out[role] = head + "\n" + "\n".join(lines)
    if short_prompt:
        # A brief asking for a glossy render or a splash keeps its picture; who is in the
        # frame is decided either way.
        plain = SUBJECT_PLAIN if "hyperreal" in out["active"] else ""
        writing = SUBJECT_NO_WRITING if "extra_text" in out["active"] else ""
        out["subject"] = ("\n\n" + _HEAD["subject"] + "\n" + SUBJECT_MEDIUM + writing + plain + SUBJECT_WHO
                          + BODY_VIEWS.rstrip("\n"))
        out["people_field"] = PEOPLE_FIELD
        extra = "\n".join((ART_MEDIUM, ART_WHO, BODY_VIEWS.rstrip("\n")))
        out["art"] = f"{out['art']}\n{extra}" if out["art"] else f"{_HEAD['art']}\n{extra}"
    return out
