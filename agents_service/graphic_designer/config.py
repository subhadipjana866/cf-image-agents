"""Configuration module for the Graphic Designer agent.

Centralises all runtime configuration consumed by the graphic-designer
pipeline, including:

- **API keys** – loaded from environment variables via ``python-dotenv``.
- **Pricing tables** – per-request cost estimates for generation
  (``GEN_PRICES``) and RAG (``RAG_PRICES``) models.
- **Token limits** – maximum context-window sizes for every supported model.
- **Moderation thresholds** – category-level score cut-offs for the
  OpenAI moderation endpoint.
- **Prompt templates** – gold-standard editing prompts keyed by edit-type
  category (``editPrompts``).
- **Aspect-ratio mappings** – default and model-specific aspect-ratio
  look-up tables.
- **UI / notification strings** – user-facing messages surfaced by various
  nodes in the graph.
"""

import os
from dotenv import load_dotenv
import boto3
import settings

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
FLUX_API_KEY = os.getenv('FLUX_API_KEY')
STABILITY_API_KEY = os.getenv('STABILITY_API_KEY')
DOCUMENT_BUCKET = os.getenv("DOCUMENT_BUCKET", "kha-documents")
REGION_NAME = os.getenv("REGION_NAME", "us-west-2")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")


INDEX_PATH = "./graphic_designer/assets/faiss_index"
"""str: Filesystem path to the FAISS vector-search index."""


GEN_PRICES = {
    "gpt-5.6-terra": 0.035,
    "gpt-5.4": 0.035,
  "gpt-5": 0.0225,
  "gpt-5-mini": 0.0045,
  # Same rate as gpt-5-mini: PROVIDER_TEXT_RATES prices both at (0.25, 2.00)
  # per 1M tokens, so this isn't a guess -- it's what OpenAI actually charges
  # for gpt-5.4-mini today. Added so classify_extract's billing_id can name
  # the model that's actually on the wire instead of borrowing gpt-5-mini's.
  "gpt-5.4-mini": 0.0045,
  "gpt-5-nano": 0.0009,
  # Decisions API (generationManager's new-vs-related check). Input-only
  # pricing ($0.10 / 1M), billed at the same ~18x input markup as gpt-5-mini/nano.
  "gpt-6-luna": 0.0018,
  "gpt-4o": 0.04,
  "gpt-4o-mini": 0.0015,
  "gpt-4": 0.18,
  "gpt-4.1": 0.02, # new
  "gpt-4.1-mini" : 0.004, # new
  "gpt-4.1-nano" : 0.001, #new
  "o4-mini": 0.011, #new
  "o1" : 0.15,
  "o1-mini" : 0.03,
  'o3-mini': 0.011,
  "o3-mini-high" : 0.0176,
  "gpt-3.5-turbo": 0.004,
  "claude-3-5-sonnet-20240620": 0.036,
  "claude-3-haiku-20240307": 0.003,
  "claude-opus-4-0" : 0.18, # new
  "claude-sonnet-4-0" : 0.036, # new
  "claude-sonnet-4-5-20250929": 0.036, # new
  "claude-3-7-sonnet-latest": 0.036, # new
  "claude-haiku-4-5-20251001": 0.012, # new
  "gemini-1.5-pro": 0.0125,
  "gemini-1.5-flash": 0.0075,
  "gemini-2.0-flash": 0.001,
  "gemini-2.5-pro" : 0.023, # new
  "gemini-2.5-flash" : 0.0056, # new
  "gemini-2.5-flash-lite-preview-06-17" : 0.001, # new
  'gemini-3-pro-preview': 0.024,
  "gemini-3.1-pro-preview": 0.028,
  'gemini-3-flash-preview': 0.0056,
  # v3.3 generation pipeline's reasoning models -- billed at the same $/1K
  # rate as gemini-3-flash-preview, the host's equivalent-tier Gemini flash
  # model, since neither has a rate of its own published here yet.
  'gemini-3.8-flash': 0.0056,
  'gemini-3.7-flash': 0.0056,
  'deepseek-chat' : 0.0027, #new
  'deepseek-reasoner' : 0.0055, #new
  "open-mistral-nemo": 0.006,
  "mistral-small-latest": 0.008,
  "mistral-large-latest": 0.016,
  'llama3': 0.02,
  "dall-e-3 SD": 0.06,
  "dall-e-3 HD": 0.12,
  "dall-e-2" : 0.03,
  "sd-ultra": 0.12,
  "sd-core": 0.045,
  "sd-3.5-large-turbo": 0.06,
  "sd-3.5-medium": 0.05,
  "flux-pro-1.1": 0.06,
  "flux-pro" :  0.075,
  "flux-kontext-pro": 0.06,
  "flux-kontext-max": 0.12,
  "gpt-expand": 0.06,
  'gemini-2.0-flash-image-preview': 0.06,
  # v3.3 generation pipeline's fallback image renderer (gpt-image-2.5-flare fails or
  # is misconfigured). Same tier as nano-banana below.
  'gemini-3.1-flash-image': 0.06,
  'nano-banana': 0.06,
  'nano-banana-pro': 0.27,
  'nano-banana-2': 0.2,
  'imagen-3': 0.04,
  # Same rate gpt-image-2 billed at, which this model replaced.
  'gpt-image-2.5-flare': 0.06,
  'outpaint': 0.04,
  'search-recolor' : 0.08,
  # Masked recolor rides the same marked-edit path as search-recolor.
  'recolor': 0.08,
  'search-replace': 0.08,
  'replace': 0.08,
  'search-erase': 0.2,
  'erase': 0.06,
  'search-erase': 0.06,
  'upscale-conservative': 0.5,
  'upscale-creative' : 0.5,
  'upscale-fast' : 0.02,
  'bg-remove' : 0.04,
  'reference' : 0.1
}
"""dict[str, float]: Per-request cost (USD) for each image-generation and editing model."""

RAG_PRICES={
    "gpt-5": 0.03375,
    "gpt-5-mini": 0.00675,
    "gpt-5-nano": 0.00135,
    "gpt-4": 0.27,
    "gpt-4.1": 0.03, # new
    "gpt-4.1-mini" : 0.006, # new
    "gpt-4.1-nano" : 0.0015, #new
    "o4-mini": 0.016, #new
    "gpt-4o": 0.06,
    "gpt-4o-mini": 0.025,
    "gpt-3.5-turbo": 0.006,
    "o1": 0.25,
    "o1-mini": 0.045,
    "o3-mini": 0.016,
    "o3-mini-high": 0.00264,
    "claude-3-5-sonnet-20240620": 0.054,
    "claude-3-haiku-20240307": 0.0045,
    "claude-opus-4-0" : 0.27, # new
    "claude-sonnet-4-0" : 0.054, # new
    "claude-3-7-sonnet-latest": 0.054, # new
    "claude-sonnet-4-5-20250929": 0.054, # new
    "gemini-1.5-pro": 0.19,
    "gemini-1.5-flash": 0.012,
    "gemini-2.0-flash": 0.0015,
    "gemini-2.5-pro" : 0.035, # new
    "gemini-2.5-flash" : 0.0084, # new
    "gemini-2.5-flash-lite-preview-06-17" : 0.0015, # new
    'deepseek-chat' : 0.004, #new
    'deepseek-reasoner' : 0.0085, #new
    "open-mistral-nemo": 0.0063,
    "mistral-small-latest": 0.012,
    "mistral-large-latest": 0.024,
    'llama3': 0.03,
}
"""dict[str, float]: Per-request cost (USD) for each RAG (Retrieval-Augmented Generation) model."""

ACTUAL_RATES_PER_1K = {
    # (input, output) USD per 1,000 tokens — what the provider actually
    # charges, at the SAME unit GEN_PRICES bills the customer in (that table
    # is deliberately separate: mixing real cost and billed price is how a
    # runaway spend stays invisible). Supersedes the old PROVIDER_TEXT_RATES /
    # PROVIDER_IMAGE_RATES split (per-1M text, flat-per-image images, two
    # units) with one table at one unit covering both.
    #
    # Verified live against every provider's current pricing docs and, where a
    # model has no published per-token rate, against a real test call
    # (2026-08-06). Every entry not marked "derived" below is the provider's
    # own quoted number, unit-converted only.

    # Google Gemini text — ai.google.dev/gemini-api/docs/pricing
    "gemini-3.1-flash-lite": (0.00025, 0.00150),
    "gemini-3-flash-preview": (0.00050, 0.00300),
    "gemini-2.5-flash": (0.00030, 0.00250),
    "gemini-3.5-flash": (0.00050, 0.00300),   # unused by any call site; unverified, carried forward as-is

    # v3.3 generation pipeline's own models -- these three rows are v33's OWN
    # self-reported rates (its generation_pipeline.py RATES dict), copied here
    # so v32Generation's apiLog calls get a real actual_cost instead of an
    # UNPRICED None. gemini-3.1-flash-image is its fallback image renderer.
    "gemini-3.8-flash": (0.00050, 0.00300),
    "gemini-3.7-flash": (0.00050, 0.00300),
    "gemini-3.1-flash-image": (0.00050, 0.06000),

    # OpenAI text — developers.openai.com/api/docs/pricing
    "gpt-5-mini": (0.00025, 0.00200),
    # Decisions API: input only; output/cache are not charged.
    "gpt-6-luna": (0.00010, 0.0),
    # gpt-5.4-mini and gpt-5.6-terra were previously misfiled here at
    # gpt-5-mini's old rate / a stale guess. Corrected against OpenAI's
    # current pricing page.
    "gpt-5.4-mini": (0.00075, 0.00450),
    "gpt-5.6-terra": (0.00200, 0.01200),
    "gpt-5.4": (0.00250, 0.01500),             # the text turn inside a gptExpandImage call
    "omni-moderation-latest": (0.0, 0.0),      # confirmed free by OpenAI

    # Anthropic — platform.claude.com/docs/en/about-claude/pricing
    "claude-haiku-4-5-20251001": (0.00100, 0.00500),

    # Google Gemini image. nano-banana-2 and nano-banana-pro are genuinely
    # token-metered (Google publishes a $/1M *output*-token rate for both,
    # matching the 1K/2K per-image example figures exactly). Both are priced
    # at the 1K/2K tier — **4K requests bill ~2x this** (nano-banana-2: $0.151
    # vs $0.067; nano-banana-pro: $0.24 vs $0.134 per image), a step Google's
    # per-token rate doesn't change but the token *count* does.
    # nano-banana (gemini-2.5-flash-image) publishes only a flat $/image
    # figure ($0.039) with no per-token rate; its output side here is
    # *derived* — that flat price divided by an output-token count measured
    # from a live test call (1,290 image tokens).
    "nano-banana": (0.00030, 0.03023),          # output: derived
    "nano-banana-2": (0.00050, 0.06000),
    "nano-banana-pro": (0.00200, 0.12000),

    # OpenAI image — both genuinely token-metered. (An earlier version of this
    # table treated gpt-image-2 as flat-per-image, on the strength of
    # developers.openai.com/api/docs/guides/image-generation's claim that it
    # "uses direct per-image pricing rather than token-based calculation".
    # That page is wrong, or describes something else: OpenAI's actual
    # pricing table (platform.openai.com/docs/pricing) lists gpt-image-2
    # under "Prices per 1M tokens" with real Image/Text input+output rows,
    # and applying those rates to a live call's real token counts reproduced
    # OpenAI's own published example price to within 0.5% — not a
    # coincidence.) This row uses the *text* input rate ($5/1M) rather than the
    # image input rate ($8/1M) — output is always an image ($30/1M).
    # CAVEAT (was true, no longer): gpt-image-2 is no longer generation-only. It
    # is now the primary editor on the suggestion-execution path
    # (suggestionExecutors -> sugGptImageEditor), which DOES attach a reference
    # image, so those calls are metered at the image input rate and this row
    # under-bills them by roughly the $5->$8 gap on input tokens (~10% of an
    # edit call, since image output dominates the bill either way). One row
    # cannot be right for both shapes: raising it to 0.00800 would fix edits and
    # over-bill the much higher-volume generation path. Left at the generation
    # rate deliberately; split into per-call-type rows if edit volume grows.
    # gpt-image-1.5's row uses the *image* input/output rates instead, since
    # every edit call in this pipeline attaches an input image and image
    # tokens dominate the bill there (cross-checked against a live edit call:
    # measured effective cost ~$0.0365 vs. OpenAI's published example of
    # $0.034 for the same quality/size — consistent, measured mix ~$7.6/$30.1
    # effective vs. $8/$32 published for image tokens alone).
    # gpt-image-2.5-flare replaced gpt-image-1, 1.5 and 2 everywhere and is
    # priced the same as gpt-image-2 was.
    "gpt-image-2.5-flare": (0.00500, 0.03000),

    # Flat-fee providers with no token concept at all — Flux is pure
    # diffusion, Stability's upscaler and the Responses API's
    # image_generation tool bill per call/image. Following the convention
    # already used elsewhere in this codebase for image cost approximation
    # (calculateCost(model, 1000) — one call priced as "1,000 tokens"), each
    # flat price is simply restated as its own $/1k rate; there is no real
    # "input" vs "output" split so both sides of the tuple are equal.
    "flux-kontext-pro": (0.04, 0.04),           # BFL docs: $0.04/image (4 credits)
    "upscale-fast": (0.02, 0.02),               # Stability docs: $0.02/image (2 credits)
    # gpt-expand: no single published number exists for "gpt-5.4 + the
    # image_generation tool" as a combo — derived from one live test call
    # (2026-08-06): the text-conversation turn (2,648 in / 59 out tokens at
    # gpt-5.4's own rate above) plus the tool's own separately-reported usage
    # (1,087 in / 1,158 out tokens, priced at gpt-image-1.5's image rates, the
    # closest documented analog at the time) totaled ~$0.053.
    "gpt-expand": (0.053, 0.053),               # derived, single live sample — a rough figure, not a quote
}
"""dict[str, tuple[float, float]]: Real provider (input, output) USD per 1,000
tokens — every model id that can appear in :data:`MODEL_ROLES` or get passed to
:func:`utils.providerCost` needs an entry here, or that call logs as "unpriced".
See :data:`KOGNIE_RATES_PER_1K` for what we bill on top of this.
"""

KOGNIE_RATES_PER_1K = {model: (round(rates[0] * 2, 6), round(rates[1] * 2, 6))
                        for model, rates in ACTUAL_RATES_PER_1K.items()}
"""dict[str, tuple[float, float]]: Kognie's rate — exactly 2x
:data:`ACTUAL_RATES_PER_1K`, computed rather than hand-duplicated so the two
can never silently drift apart.

Reference only, as of this table's introduction: :data:`GEN_PRICES` is what
:func:`utils.calculateCost` actually bills against, and GEN_PRICES's existing
per-model figures are NOT a clean 2x multiple of the real rate above — the two
price books are allowed to diverge until GEN_PRICES is deliberately migrated
onto this one.
"""

FLAT_FEE_MODELS = {"flux-kontext-pro", "upscale-fast", "gpt-expand"}
"""set[str]: The subset of :data:`ACTUAL_RATES_PER_1K` that are genuinely
billed per call/image by their provider, not per token (no token concept
exists for these at all — pure diffusion or a fixed-price endpoint). Their
(input, output) rate is the same number on both sides, defined as "one call
== 1,000 token-equivalents". :func:`utils.providerCost` uses this set to know
it's safe to assume 1,000 when a call reports no usage at all, instead of
returning "unpriced" the way an un-instrumented token-metered call site should.
"""

IMAGE_CALL_PURPOSES = {
    # Every :func:`utils.apiLog` `purpose` whose call actually renders, edits or
    # segments pixels. `purpose` is the call-site name, so this is the reliable
    # half of the test: a couple of image endpoints are driven by a *text*
    # model id (``gpt-5.4`` behind ``gptExpandImage``), which the model set
    # below structurally cannot catch.
    "dalleGen", "stableDiffGen", "fluxGen", "fluxExecutor",
    "bananaImageGen", "bananaProImageGen", "nanoBanana2ImageGen",
    "bananaExecutor", "bananaProExecutor",
    "gptImageGen", "gptImageExecutor", "gptExpandImage",
    "sugBananaEditor", "sugGptImageEditor",
    "upscaleImage",
    "v33Render",
}
"""set[str]: apiLog purposes that denote real image work — see
:func:`utils.isImageApiCall`, which drives the ``show_cost`` flag on every
interrupt (:func:`utils.markImageWork`)."""

IMAGE_CALL_MODELS = {
    # The model-id half of the same test, for call sites that pass a model
    # this table knows but a purpose it doesn't (a new executor, a fallback
    # rung). Names are as they reach apiLog, which is not always the
    # GEN_PRICES key.
    "dall-e", "dall-e-2", "dall-e-3 SD", "dall-e-3 HD",
    "stable-image", "sd-ultra", "sd-core", "sd-3.5-large-turbo", "sd-3.5-medium",
    "flux", "flux-pro", "flux-pro-1.1", "flux-kontext-pro", "flux-kontext-max",
    "nano-banana", "nano-banana-pro", "nano-banana-2",
    "gemini-2.0-flash-image-preview", "imagen-3",
    "gpt-image-2.5-flare", "gpt-expand",
    "gemini-3.1-flash-image",
    "upscale-fast", "upscale-conservative", "upscale-creative",
    "outpaint", "bg-remove", "erase", "search-erase",
    "replace", "search-replace", "search-recolor",
}
"""set[str]: Image model ids as they are passed to :func:`utils.apiLog`.
Deliberately does NOT include ``lanczos-resize`` — that one is a local,
free resize, not billable image work."""

IMAGE_RESULT_INTERRUPTS = {
    # Interrupt types that ARE the delivered result of image work, so they show
    # their cost unconditionally — no marker lookup, no cost>0 test. The node
    # emitting one of these has, by definition, just handed the user a new or
    # changed image, and that is exactly the moment the charge belongs on
    # screen. Bypassing the marker also makes them immune to a pixel path that
    # was never wired into markImageWork.
    "generation_completed",
    "editing_successful",
    "expand_successful",
    "layering_executed",
    "suggestions_executed",
}
"""set[str]: Interrupt types :func:`utils.stampShowCost` always flags
``show_cost=True``. Deliberately excludes ``execution_failed`` (every executor
fell over — nothing was delivered) and the pre-execution frames like
``text_input``-based suggestion lists, which have produced no image yet."""

MODEL_ROLES = {
    "classify_extract": {
        "strategy": "alternating",
        # Order matters when strategies are disabled: pool[0] is what a
        # strategy-off run uses, and it is today's primary, so turning the
        # router on is the only thing that changes behavior.
        "pool": [
            {"provider": "google", "api_id": os.getenv("CLASSIFY_MODEL_GOOGLE", "gemini-3-flash-preview"),
             "billing_id": "gemini-3-flash-preview", "weight": 1},
            {"provider": "openai", "api_id": os.getenv("CLASSIFY_MODEL_OPENAI", "gpt-5.4-mini"),
             "billing_id": "gpt-5.4-mini", "weight": 1},
        ],
        "fallback_chain": [],
    },
    "cot_reasoning": {
        # Was "pinned" + sticky (every one of the 7 CoT stages in a run pinned
        # to whoever served stage 1). Changed to round-robin Gemini/GPT per
        # call, same mechanism as classify_extract -- each stage now resolves
        # independently, including within a single run, so a run can genuinely
        # mix Gemini- and GPT-authored stages. Anthropic stays a reactive-only
        # fallback rather than joining the rotation.
        "strategy": "alternating",
        # Defaults to flash-lite because that is what the CoT actually runs on
        # today (MODEL_PRO was never reached from the graph -- every call site
        # passed an explicit flash-lite id), and because this project's Gemini
        # key currently returns 429 RESOURCE_EXHAUSTED for every Pro model.
        # Once that quota is fixed, the mandate's Pro-tier pin is one env var:
        #     COT_MODEL=models/gemini-3.1-pro-preview
        # model_router.preflight() reports at boot which of these the key can
        # actually serve, so an unservable pin is loud instead of silent.
        "pool": [
            {"provider": "google", "api_id": os.getenv("COT_MODEL", "models/gemini-3.1-flash-lite"),
             "billing_id": "gemini-3-flash-preview", "weight": 1},
            {"provider": "openai", "api_id": os.getenv("COT_OPENAI_FALLBACK_MODEL", "gpt-5.6-terra"),
             "billing_id": "gpt-5.6-terra", "weight": 1},
        ],
        "fallback_chain": [
            {"provider": "anthropic", "api_id": os.getenv("COT_ANTHROPIC_FALLBACK_MODEL",
                                                          "claude-haiku-4-5-20251001"),
             "billing_id": "claude-haiku-4-5-20251001"},
        ],
    },
    "copywriting": {
        "strategy": "pinned",
        "pool": [
            {"provider": "google", "api_id": os.getenv("COPY_MODEL", "models/gemini-3.1-flash-lite"),
             "billing_id": "gemini-3-flash-preview", "weight": 1},
        ],
        "fallback_chain": [
            {"provider": "openai", "api_id": "gpt-5.6-terra", "billing_id": "gpt-5.6-terra"},
        ],
    },
    "image": {
        "strategy": "isolated",
        "pool": [
            {"provider": "openai", "api_id": os.getenv("IMAGE_MODEL", "gpt-image-2.5-flare"),
             "billing_id": "gpt-image-2.5-flare", "weight": 1},
        ],
        "fallback_chain": [
            {"provider": "google", "api_id": "nano-banana-2", "billing_id": "nano-banana-2"},
            {"provider": "google", "api_id": "nano-banana-pro", "billing_id": "nano-banana-pro"},
        ],
    },
}
"""dict: role -> routing policy. A *role* is what a call site asks for; which
concrete model serves it is a deployment decision that lives here and nowhere
else.

``api_id`` is what goes on the wire. ``billing_id`` is the :data:`GEN_PRICES`
key :func:`utils.calculateCost` must be handed. They are two fields rather than
one derivation because they genuinely differ today — five call sites send
``models/gemini-3.1-flash-lite`` and bill ``gemini-3-flash-preview``, and
``GEN_PRICES`` has no flash-lite key at all. Deriving one from the other would
either crash ``calculateCost`` or silently re-price every generation.
**Changing which model serves a call must never move a number the customer is
billed**; that separation is what this pair of fields buys.

``strategy`` decides how a pool member is picked:

* ``alternating`` — round-robin across the pool, and whichever member is picked,
  the others become the head of its fallback chain. For stateless, cheap calls
  (classification, extraction) where spreading load across two providers' rate
  limits is worth more than provider consistency.
* ``pinned`` — always the first pool member, with ``sticky`` meaning "and every
  call in the same run uses whatever that run first resolved". The CoT stages
  share a growing prior-stages context, so a mid-run provider switch would leave
  one model's prose as binding context for another's.
* ``isolated`` — a reserved pool that no other role borrows from. The image role
  owns the OpenAI image quota; text roles must not spend it.

``fallback_chain`` is the ordered safety net tried only after the selected pool
member fails. It is a *reactive* backstop; ``strategy`` is the *proactive*
spread, and the two are deliberately different mechanisms.
"""

PROVIDER_LIMITS = {
    # provider -> {"rpm": int|None, "tpm": int|None, "concurrency": int|None}
    # None means "no limit configured" and disables that bucket, so shipping
    # this table unfilled is a no-op rather than a throttle at a guessed number.
    #
    # TODO: fill these from each provider's own console before enabling
    # backpressure. Do NOT guess: a wrong RPM here either throttles healthy
    # traffic or fails to prevent the 429s the bucket exists to prevent.
    "google": {"rpm": None, "tpm": None, "concurrency": None},
    "openai": {"rpm": None, "tpm": None, "concurrency": None},
    "anthropic": {"rpm": None, "tpm": None, "concurrency": None},
}
"""dict: per-provider rate limits driving :mod:`graphic_designer.model_router`'s
token buckets. Every value starts as ``None`` (disabled) on purpose — these must
be read off the provider dashboards, never inferred."""

RAGBOT_PRICES = {
    # (input, output) USD per 1M tokens — each provider's published rates.
    # Gemini bills thinking tokens inside the output count.
    "gemini-2.5-flash": (0.30, 2.50),   # kept: only matters if RAGBOT_ANSWER_MODEL is overridden back to it
    "gemini-3-flash-preview": (0.50, 3.00),  # ragbot/config.py's current ANSWER_MODEL/UTILITY_MODEL default
    "claude-sonnet-4-5-20250929": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}
"""dict[str, tuple[float, float]]: Per-million-token (input, output) price for each
ragbot answer model.

Priced per token, and split input from output, rather than through
``GEN_PRICES``' flat per-1k blended rate: a ragbot answer ships the whole Kognie
corpus as a cached prefix, so at ~29k corpus tokens the flat rate overbills a
single question by an order of magnitude on either provider. The cached-prefix
discount (``CACHE_READ_MULTIPLIER``) is most of what it gets wrong, and a blended
rate cannot express that at all.
"""

CACHE_READ_MULTIPLIER = 0.1
"""float: Cached-prefix reads bill at 0.1x the model's input rate — the same
ratio on Gemini ($0.03 vs $0.30 per MTok) and Anthropic."""

CACHE_WRITE_MULTIPLIER = 1.25
"""float: Cache writes bill at 1.25x the input rate (Anthropic, 5-minute TTL).

Anthropic-only: Gemini caches long prefixes implicitly with no write premium, so
ragbot reports 0 cache-creation tokens on that path and this never applies.
"""

MAX_TOKENS={ 'gpt-5': 400000, 'gpt-5-mini': 400000, 'gpt-5-nano': 400000,
            'gpt-4o': 128000, 'gpt-4o-mini': 128000, 'gpt-4': 8192, 'gpt-4.1': 1048576, 'gpt-4.1-mini': 1048576,'gpt-4.1-nano': 1048576, 'o4-mini' : 200000, 'o1':200000, 'o1-mini':200000, 'o3-mini': 200000,'o3-mini-high': 200000, 'gpt-3.5-turbo': 16385,
            'claude-3-5-sonnet-20240620': 200000, 'claude-3-haiku-20240307': 200000, "claude-opus-4-0": 200000, "claude-sonnet-4-0": 200000, "claude-3-7-sonnet-latest": 200000, "sd-ultra": 128000, "sd-core": 128000, "sd-3.5-large-turbo": 128000, "sd-3.5-medium": 128000,
            "dall-e-3 SD": 128000, "dall-e-3 HD": 128000, "dall-e-2": 128000, "flux-pro-1.1": 128000, "flux-pro": 128000, "flux-kontext-pro": 128000, "flux-kontext-max": 128000, "gemini-2.0-flash-image-preview": 128000, 'imagen-3': 128000, 'search-recolor': 128000, 'recolor': 128000, 'search-replace': 128000, 'replace': 128000, 'erase': 128000,
            'upscale-conservative': 128000, 'upscale-creative': 128000, 'upscale-fast': 128000, 'bg-remove': 128000, 'reference': 128000,
            'gemini-1.5-pro': 1048576, 'gemini-1.5-flash': 2097152, 'gemini-2.0-flash':1048576, "gemini-2.5-pro": 1048576, "gemini-2.5-flash": 1048576, "gemini-2.5-flash-lite-preview-06-17": 1000000, 'deepseek-chat' : 68000,
  'deepseek-reasoner' : 96000, 'open-mistral-nemo': 128000, 'banana': 400000, "claude-sonnet-4-5-20250929": 200000, # new
            'mistral-small-latest': 32000, 'mistral-large-latest': 128000, 'llama3': 128000, 'gpt-image-2.5-flare': 128000, 'search-erase': 128000
        }
"""dict[str, int]: Maximum context-window token count for each supported model."""


MODERATION_THRESHOLD = {
    "sexual": 0.8,
    "sexual/minors": 0.5,
    "harassment": 0.8,
    "harassment/threatening": 0.9,
    "hate": 0.8,
    "hate/threatening": 0.9,
    "illicit": 0.9,
    "illicit/violent": 0.8,
    "self-harm": 0.7,
    "self-harm/intent": 0.8,
    "self-harm/instructions": 0.7,
    "violence": 0.9,
    "violence/graphic": 0.9
}
"""dict[str, float]: OpenAI content-moderation score thresholds by category.

A piece of content is flagged when its category score meets or exceeds
the corresponding threshold.
"""

s3_client = None
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

aspectRatioSet = {
    "general": "1:1",
    "poster": "2:3",
    "flyer": "2:3",
    "brochure front": "1:1",
    "book cover": "2:3",
    "magazine cover": "3:4",
    # "business card": "3.5:2",
    "sticker": "1:1",
    "website hero banner": "16:9",
    # "blog header": "3:1",
    "landing page image": "16:9",
    # "email header": "3:1",
    "thumbnail": "1:1",
    "product image": "1:1",
    "app splash screen": "9:16",
    "instagram post": "1:1",
    "instagram story / reel": "9:16",
    "facebook post": "1:1",
    "linkedin post": "1:1",
    "twitter / x post": "16:9",
    "youtube thumbnail": "16:9",
    "youtube shorts": "9:16",
    "profile picture": "1:1",
    "logo": "1:1",
    "icon": "1:1",
    "mascot / character art": "3:4",
    "concept art": "16:9",
    "illustration": "4:5",
    "wallpaper desktop": "16:9",
    "wallpaper mobile": "9:16",
}
"""dict[str, str]: Default aspect ratios keyed by design purpose (e.g. ``"poster"`` → ``"2:3"``)."""


PLATFORM_DIMENSIONS = {
    # Instagram
    "instagram post":         {"width": 1080, "height": 1080},  # 1:1
    "instagram portrait":     {"width": 1080, "height": 1350},  # 4:5
    "instagram story":        {"width": 1080, "height": 1920},  # 9:16
    "instagram reel":         {"width": 1080, "height": 1920},  # 9:16
    "instagram story / reel": {"width": 1080, "height": 1920},  # 9:16

    # Facebook
    "facebook post":          {"width": 1200, "height": 1200},  # 1:1 recommended
    "facebook cover":         {"width": 1640, "height": 856},
    "facebook story":         {"width": 1080, "height": 1920},
    "facebook ad":            {"width": 1200, "height": 628},

    # YouTube
    "youtube thumbnail":      {"width": 1280, "height": 720},   # 16:9
    "youtube shorts":         {"width": 1080, "height": 1920},  # 9:16

    # Twitter / X
    "twitter post":           {"width": 1600, "height": 900},   # 16:9
    "twitter / x post":       {"width": 1600, "height": 900},

    # LinkedIn
    "linkedin post":          {"width": 1200, "height": 1200},  # 1:1
    "linkedin cover":         {"width": 1584, "height": 396},

    # Pinterest
    "pinterest pin":          {"width": 1000, "height": 1500},  # 2:3

    # General web / print
    "website hero banner":    {"width": 1920, "height": 1080},
    "poster":                 {"width": 1654, "height": 2339},  # A4 @ 200 dpi
    "youtube banner":         {"width": 2560, "height": 1440},
    "email header":           {"width": 600,  "height": 200},
    "twitter cover":          {"width": 1500, "height": 500},
}
"""dict[str, dict]: Canonical pixel dimensions for popular social media and output platforms.

Each entry maps a lowercase platform/purpose name to ``{"width": int, "height": int}``.
Used by ``expandManager`` to compute the outpaint padding needed per platform.
"""

EXPAND_PLATFORM_TO_CATALOG = {
    "instagram post":         "Instagram Square Post",
    "instagram portrait":     "Instagram Portrait Post",
    "instagram story":        "Instagram Story",
    "instagram reel":         None,
    "instagram story / reel": "Instagram Story",
    "facebook post":          None,  # square; the catalog only has the landscape feed format
    "facebook cover":         None,
    "facebook story":         None,
    "facebook ad":            "Facebook Feed Landscape",
    "youtube thumbnail":      "YouTube Thumbnail",
    "youtube shorts":         None,
    "twitter post":           "X (Twitter) In-Stream Post",
    "twitter / x post":       "X (Twitter) In-Stream Post",
    "linkedin post":          "LinkedIn Feed Post",
    "linkedin cover":         None,
    "pinterest pin":          "Pinterest Pin",
    "website hero banner":    None,
    "poster":                 None,
    "youtube banner":         None,
    "email header":           None,
    "twitter cover":          None,
}
"""dict[str, str | None]: ``PLATFORM_DIMENSIONS`` key -> ``load_catalog()`` platform name.

expandManager reports it per output URL in ``result_graphics`` so main.py stores each
expanded graphic with its own platform. ``None`` means the catalog has no equivalent;
that graphic's platform stays empty and the user is asked once about it.
"""


"""str: Prompt used by ``expandManager`` to extract target platform names from a free-text user request."""


DEVICE             = settings.DEVICE
FLORENCE_MODEL     = settings.FLORENCE_MODEL
SAM_CHECKPOINT     = settings.SAM_CHECKPOINT
SAM_CONFIG         = settings.SAM_CONFIG
USE_BIREFNET       = False
BIREFNET_MODEL     = None
STABILITY_API_KEY  = settings.STABILITY_API_KEY
USE_ERASE_ENDPOINT = settings.USE_ERASE_ENDPOINT
DEPTH_MODEL        = settings.DEPTH_MODEL
USE_DEPTH_REORDER  = settings.USE_DEPTH_REORDER
INCLUDE_BACKGROUND = settings.INCLUDE_BACKGROUND
EXPAND_MASKS       = settings.EXPAND_MASKS
MAX_PER_CONCEPT    = settings.MAX_PER_CONCEPT
MANUAL_DETECTION   = settings.MANUAL_DETECTION
USE_TEXT_REMOVAL   = settings.USE_TEXT_REMOVAL
TEXT_REMOVAL_LANG  = settings.TEXT_REMOVAL_LANG
TEXT_REMOVAL_GPU   = settings.TEXT_REMOVAL_GPU


nananoBananaProSupportedAspectRatios = ["1:1","2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
"""list[str]: Aspect ratios accepted by the Nano Banana Pro model."""


editPrompts = {
    "1A": """You are an advanced image editing engine specialized in Generative Insertion and Object Substitution.
INSTRUCTIONS:
Subject:
{{#replaces}}- SUBSTITUTION: remove the existing "{{replaces}}" from the scene completely and put a new "{{target}}" in its place. The new "{{target}}" must occupy the same position, scale and perspective the "{{replaces}}" occupied, and NO part of the original "{{replaces}}" may remain visible anywhere in the image. This is a replacement, NOT an addition: the finished image contains a "{{target}}" where the "{{replaces}}" used to be, and no "{{replaces}}" at all. Reconstruct any background revealed where the two differ in size or shape.{{/replaces}}
{{^replaces}}{{#source}}- CLONE: duplicate the existing "{{source}}" and insert the copy as an additional instance. The original stays exactly as it is.{{/source}}{{^source}}- INSERT: generate and insert a new "{{target}}" into the scene. Every object already in the image stays exactly as it is.{{/source}}{{/replaces}}
Placement:
{{#attributes.placement}}- Place it exactly at this location: "{{attributes.placement}}".{{/attributes.placement}}
{{^attributes.placement}}{{^replaces}}- No location was given: place it where it sits naturally in the scene.{{/replaces}}{{#replaces}}- Placement is inherited from the object being replaced.{{/replaces}}{{/attributes.placement}}
{{#attributes.appearance}}Appearance:
- Apply the following appearance attributes: "{{attributes.appearance}}".{{/attributes.appearance}}
{{#attributes.scale}}Scale:
- Apply the following scale: "{{attributes.scale}}".{{/attributes.scale}}
Integration:
- Match lighting, shadows, and perspective to the original scene.
- Blend edges naturally.
- Do not modify unrelated objects or the global environment.

""",

    "1B": """You are an advanced image editing engine specialized in Spatial Transformation. Your task is to geometrically manipulate an object without altering its identity.
INSTRUCTIONS:
{{#target}}Target: Select the object: "{{target}}".{{/target}}
{{#action}}Action: Perform action: "{{action}}".{{/action}}
Parameters:
{{#parameters.direction}}Direction/Movement: "{{parameters.direction}}" by "{{parameters.distance}}".{{/parameters.direction}}
{{#parameters.angle}}Rotation: "{{parameters.angle}}".{{/parameters.angle}}
{{#parameters.scale_change}}Scale: "{{parameters.scale_change}}".{{/parameters.scale_change}}
Inpainting: Reconstruct the background behind the original object position using high-fidelity context.
Constraint: Do NOT change the visual attributes (color, clothes, face) of the target.

""",

    "2": """You are an advanced image editing engine specialized in Intentional Erasure. Your task is to remove an object and fill the void.
INSTRUCTIONS:
{{#target}}Target: Identify and remove: "{{target}}".{{/target}}
Inpainting: Fill the erased area using high-fidelity texture from the surrounding environment.
Constraint: The result must be seamless; no blurs, artifacts, or distortions. Do not modify adjacent objects.

""",

    "3": """You are an advanced image editing engine specialized in Attribute Modification. Your task is to change specific surface properties of an object.
INSTRUCTIONS:
{{#target}}Target: Locate the object: "{{target}}".{{/target}}
{{#attribute_type}}Modification: Change the {{attribute_type}} of the target to {{new_value}}.{{/attribute_type}}
Constraint: Preserve the object's geometry, outline, and shading.
Negative Constraint: Do NOT change the background or other object attributes.

""",

    "4": """You are an advanced image editing engine specialized in Subject Semantic Modification. Your task is to alter specific semantic features of a person or animal.
INSTRUCTIONS:
{{#target}}Target: Locate the subject: "{{target}}".{{/target}}
{{#action}}Action: Modify the subject's {{action}}.{{/action}}
{{#attributes}}Details: Apply the following changes: "{{attributes}}".{{/attributes}}
Identity: Maintain the subject's facial structure and identity unless explicitly asked to age/change gender.
Integration: Ensure clothing folds, hair, and environmental interaction remain realistic.

""",

    "5": """You are an advanced image editing engine specialized in Environment Replacement. Your task is to replace the background while preserving foreground subjects.
INSTRUCTIONS:
Segmentation: Isolate the foreground subject(s) with pixel-perfect precision.
{{#attributes}}Replacement: Replace the entire background environment with: "{{attributes}}".
Harmonization: Adjust the lighting/color bleed on the foreground subjects to match the new "{{attributes}}" ambience.{{/attributes}}
Perspective: Align the new background's horizon line with the original camera angle.

""",

    "6": """You are an advanced image generation engine specialized in Global Style Transfer. Your task is to reinterpret the image in a new artistic style.
INSTRUCTIONS:
{{#attributes}}Style: Apply the artistic style: "{{attributes}}".{{/attributes}}
Content Preservation: Keep all objects, people, and compositions recognizable.
Constraint: Do not add or remove objects; only change the rendering technique (brushstrokes, lighting, medium).

""",

    "7": """You are an advanced image editing engine specialized in Typography and Graphics. Your task is to handle text and logos within the image.
INSTRUCTIONS:
{{#action}}Action: "{{action}}" the text/graphic.{{/action}}
Content:
{{#attributes.text}}Text String: "{{attributes.text}}"{{/attributes.text}}
{{#attributes.font}}Font Style: "{{attributes.font}}"{{/attributes.font}}
{{#attributes.color}}Color: "{{attributes.color}}"{{/attributes.color}}
Geometry:
{{#attributes.size}}Size: "{{attributes.size}}"{{/attributes.size}}
{{#attributes.placement}}Placement: "{{attributes.placement}}"{{/attributes.placement}}
Integration: If adding text, follow the surface perspective (warping) and lighting. If removed, inpaint cleanly.

""",

    "8": """You are an advanced image editing engine specialized in Canvas Geometry. Your task is to adjust the frame of the image.
INSTRUCTIONS:
{{#action}}Action: Perform: "{{action}}".{{/action}}
{{#target}}Target Focus: Center composition or focus calculations on: "{{target}}".{{/target}}
[IF target IS NULL]: Center on image middle.
{{#parameters}}Parameters: Apply dimensions/aspect ratio/zoom level: "{{parameters}}".{{/parameters}}
Outpainting: If extending, generate coherent context that matches the existing scene boundaries.

""",

    "9": """You are an advanced image enhancement engine. Your task is to improve technical quality without content alteration.
INSTRUCTIONS:
{{#action}}Action: Perform: "{{action}}".{{/action}}
{{#attributes}}Focus Area: Apply enhancement specifically to: "{{attributes}}".{{/attributes}}
[IF attributes IS NULL]: Apply globally to the whole image.
Fidelity: Do NOT change facial features, shapes, or composition. Maintain strict fidelity to original data.

""",

    "10": """You are an advanced image editing engine specialized in Ambience and Mood. Your task is to relight and color-grade the scene.
INSTRUCTIONS:
{{#attributes}}Mood: Change the global atmosphere to: "{{attributes}}".{{/attributes}}
adjustments: Modify color grading, white balance, contrast, and shadows to achieve this mood.
Constraint: Do NOT change the geometry, weather (unless specified in attributes), or add new objects.

"""
}
"""dict[str, str]: Gold-standard prompt templates keyed by edit-type category.

Each template uses Mustache-style ``{{placeholders}}`` that are rendered at
runtime with the user's edit request parameters.
"""

pending_action = None
"""str | None: Globally tracked pending action (used by interrupt / resume logic)."""

pending_node = None
"""str | None: Globally tracked pending node name (used by interrupt / resume logic)."""


promptCollectorMessage = """Please tell Kognie about the image you would like to create. Provide
as many details as you can. Remember Kognie can't read you mind 😊."""
"""str: Default greeting shown by the prompt-collector node."""


bestModelNotification = """We recommend using {{best_model}} for generation based on your purpose. Let us know if you want to switch to this model?"""
"""str: Notification template shown when the best-model node suggests an alternative model."""


missingDetailsNotification = """Could you please provide the missing details. Or ask
Kognie to go ahead without the missing details."""
"""str: Notification shown when mandatory generation details are missing."""


suggestionHandlerMessage = (
    'Dear {{username}}, our Critic Agent has some suggestions to improve your design. You '
    'can apply or modify these suggestions in our Suggestion Studio.'
)
"""str: Message template inviting the user into suggestion mode."""

SUGGESTION_MORE_ASK = 'Want more suggestions to improve this graphic?'
"""str: Asked once the accepted suggestions have been executed and the stack
is empty -- offers another critique pass on the freshly edited image before
falling through to interruptBeforeLayering's normal 'what next' step."""


suggestionResponseNotification = """Hey {{username}}, are you OK with these suggestions?
Feel free to modify or reject any of them. You won't hurt our feelings! 😊"""
"""str: Follow-up template asking for feedback on generated suggestions."""


commonUnclearNotification = (
    "Sorry, I couldn't quite understand your request. Could you please clarify or provide "
    "more details about what you'd like to do with the design?"
)
"""str: Fallback message when the navigator cannot classify the user's intent."""


suggestionUnclearNotification = (
    "Sorry, I couldn't quite understand your request. Could you please clarify or provide "
    "more details about what you'd like to do with the design? Here are some suggestions "
    'to help you'
)
"""str: Unclear-intent fallback message used within suggestion mode."""


# ============================================================================
# SYSTEM PROMPTS — ROUTING & CLASSIFICATION
# ============================================================================

"""str: System prompt for Phase 1 intent classification (generation/editing/expand/suggestion/non-image-related)."""


"""str: System prompt for Phase 2 sub-classification of non-image-related queries."""


"""str: System prompt for accepting/rejecting decisions or suggestions."""


"""str: System prompt for computing state updates based on user decisions."""


"""str: System prompt for the prompt enhancer node."""


"""str: System prompt for chitchat handler."""


"""str: System prompt for question handler classifier."""


# ============================================================================
# INTERRUPT MESSAGES — USER NOTIFICATIONS
# ============================================================================
# Single home for every static user-facing interrupt text (and the templates
# for dynamic ones -- fill with str.format). Messages the LLM writes at runtime
# (chit-chat, question answers, clarification questions, critique text) are
# not here. SUGGESTION_TARGET_ASK, DESIGN_SUGGESTION_STARTED and the older
# lowercase notifications above (suggestionHandlerMessage, SUGGESTION_MORE_ASK,
# commonUnclearNotification, suggestionUnclearNotification) belong to the same
# set.

NO_IMAGE_FOR_EDIT = 'No graphic is selected for edit. Please select one.'
"""str: Message shown when user requests edit but no image is selected."""

GENERATING_STARTED = 'Generating your graphic... This may take several seconds.'
"""str: Status notification when generation begins."""

GENERATION_COMPLETED = (
    'Your graphic has been generated. Now our Critic Agent is doing a professional review '
    'of the graphic and it will some suggestions to improve your graphic. Please wait for '
    'those...'
)
"""str: Message shown after generation completes before suggestions."""

SLOP_REMOVAL_OFFER = (
    'Your graphic has been generated. Would you like us to remove any AI slop from it '
    '(the tell-tale artifacts that make a graphic look AI-made)?'
)
"""str: generation_completed question while SLOP_REMOVAL_ENABLED is on -- a
no sends the graphic on to the Critic Agent as GENERATION_COMPLETED does."""

SLOP_CREATE_MORE_ASK = 'We have removed the AI slop from your graphic. Would you like to create more?'
"""str: subgraphs.slop.slopReview -- ``slop_removed``. "create more" goes to
the navigator; anything else sends the cleaned graphic to the Critic Agent."""

SLOP_RESUME_NOTE = 'Would you like to create more?'
"""str: Appended to question/chit-chat/unclear replies given inside the slop
subgraph, so the user is put back on the question they were answering."""

NO_IMAGE_FOR_SLOP_REMOVAL = 'No graphic is selected to remove AI slop from. Please select one.'
"""str: slopManager -- slop removal asked for with no image to work on."""

EDIT_CANCELLED = "Your edit request has been cancelled. What would you like to do next?"
"""str: Message shown when user cancels an edit request."""

EDIT_SUCCESSFUL = "Your image has been edited successfully! What would you like to do next?"
"""str: Message shown after successful image edit."""

EDIT_WRONG_IMAGE_RESELECT = (
    "Sorry about that! Please select the image you want to edit on the canvas, "
    "then send your request again."
)
"""str: editWrongImageHandler -- the reply to an edit said it went to the wrong
image. The reply to this goes to the navigator like any new request."""

EXTREME_FALLBACK = (
    'Apologies 😞. We are unable to process your request at this time. Please try again '
    'from the beginning by reentering the prompt.'
)
"""str: Terminal fallback message when all operations fail."""

SUGGESTION_NEXT_STEP = "What do you want to do next?"
"""str: Message asking what to do after suggestions are completed."""

# Suggestion mode alternatives
SUGGESTION_INCOMPLETE_MESSAGE = "Apologies! Currently we can't make any suggestions in this workflow. We are working on it. But could you please provide the missing info."
"""str: Message when suggestions aren't available for current workflow."""

VALIDATION_MISSING_DETAILS = "Please provide the missing details to continue editing your image."
"""str: Validation subgraph message to continue providing edit details."""

VALIDATION_SUGGESTION_FEEDBACK = "Please provide your feedback on the suggestions to continue."
"""str: Validation subgraph message to respond to suggestions."""

# Frontend chatbox messages
INITIAL_GREETING = "Hello! I'm Kognie's AI assistant. How can I help you with your design today?"
"""str: Initial welcome message on page load."""

CONNECTION_LOST = "⚠️ Connection lost. Don't worry — you'll be reconnected automatically when you send your next message. Please note that your previous conversation history will not be available after reconnection."
"""str: Message shown when WebSocket connection is lost."""

REQUEST_PROCESSING = "Your request is being processed. Please wait for a moment..."
"""str: Loading placeholder message while processing user input."""

IMAGE_UPLOAD_FAILED = "Failed to upload images. Please try again."
"""str: Error message when image upload fails."""

CANVAS_LIMIT_REACHED = "Image generation completed, but canvas limit reached. Please remove some images to add this one."
"""str: Error message when canvas reaches image limit."""

SUGGESTION_EXECUTED = "Your image has been updated with the suggestion! ✨"
"""str: Success message when suggestion is executed."""

SUGGESTION_DECLINED = "No problem! What can we help you with next? 😊"
"""str: Message when user declines suggestion mode."""

SUGGESTION_MODE_EXITED = "You've exited suggestion mode. What can we help you with next? 😊"
"""str: Message when user exits suggestion mode."""

SUGGESTION_EDITS_COMPLETE = "Great! All your edits are complete. What would you like to do next? 🎉"
"""str: Message when all suggestion edits are complete."""

REFERENCE_LIMIT_MESSAGE = "Only a max of {max_images} images can be added as references."
"""str: Warning when trying to add too many reference images."""

SUGGESTION_MODE_ACTIVE = "You are in suggestion mode right now"
"""str: Banner message while in suggestion mode."""

SUGGESTION_MODE_DESCRIPTION = "You've entered suggestion mode where AI will give you personalized suggestions to improve your image to better suit your purpose."
"""str: Description text in suggestion mode."""

SUGGESTION_MODE_INSTRUCTION = "Please choose a suggestion to accept it, reject it or modify the suggestion to better suit you."
"""str: Instruction text in suggestion mode panel."""

SUGGESTION_MODE_COMPLETE_NOTICE = "Execution completed — exit suggestion mode"
"""str: Notice in suggestion mode when edits are complete."""


# ── Generation flow ─────────────────────────────────────────────────────────

SCOPE_SWITCH_GOAL_ASK = (
    "You selected '{design_goal}' for '{platform}', but your prompt reads like a "
    "'{detected_goal}' design{note}. Switch to that instead?"
)
"""str: scopeMismatchGate -- detected goal is a supported catalog goal. ``note``
is :data:`SCOPE_SWITCH_GOAL_NOTE` or ``""``."""

SCOPE_SWITCH_GOAL_NOTE = " on '{primary}'"
"""str: Appended to :data:`SCOPE_SWITCH_GOAL_ASK` when the prompt also names another platform."""

SCOPE_FREEFORM_GOAL_ASK = (
    "Your prompt reads like a '{detected_goal}', which isn't one of the design types we "
    "build for '{platform}'. Do you still want me to generate it straight from your "
    "description instead of the '{design_goal}' layout?"
)
"""str: scopeMismatchGate -- detected goal is off-catalog (freeform offer)."""

SCOPE_SWITCH_PLATFORM_ASK = (
    "You selected '{platform}', but your prompt asks for '{primary}'. "
    "Switch this design to '{primary}'{extra}? Everything else about the "
    "'{design_goal}' stays the same."
)
"""str: scopeMismatchGate -- prompt asks for another supported platform. ``extra``
is :data:`SCOPE_SWITCH_PLATFORM_EXTRA` or ``""``."""

SCOPE_SWITCH_PLATFORM_EXTRA = " (and then {pending})"
"""str: Appended to :data:`SCOPE_SWITCH_PLATFORM_ASK` listing the queued platforms."""

SCOPE_FREEFORM_PLATFORM_ASK = (
    "Your prompt mentions {named}, which isn't a format our Layout Agent is an expert on. "
    'I can bypass our Layout Agent and generate it straight from your description instead '
    "of using the '{design_goal}' / '{platform}' choices. Go ahead?"
)
"""str: scopeMismatchGate -- prompt names a platform with no layout (freeform offer)."""

SCOPE_ABORT_ASK = (
    "Okay, I've cancelled this generation. What would you like to do "
    "instead? You can rephrase your request for '{design_goal}' on "
    "'{platform}', or describe something different."
)
"""str: scopeAbortGate -- the user declined the scope-mismatch offer."""

MANIFEST_FIELD_FALLBACK = "Please provide: {label}."
"""str: manifestFieldsAsk -- used for a field/asset whose manifest entry has no question/reason."""

PHOTO_REQUIRED_DEFAULT = 'A real photo is needed for this design. Please upload one.'
"""str: photoRequiredGate -- used when the manifest/layout gives no reason of its own."""

LOGO_ASK = "Would you like your brand logo included in this design?"
"""str: logoDecisionGate -- the layout has a logo spot and no logo is on file. Sent with no ``reason``."""

LOGO_ASK_REASON = (
    'You asked for a logo, but no logo file was provided at project creation and none is '
    "on file for your brand. Please upload the logo and I'll place it in the proper spot."
)
"""str: logoDecisionGate / editAssetGate -- the user asked for a logo, so this asks
for the file (an asset request, not a yes/no question)."""

V32_NEEDS_INPUT_DEFAULT = (
    "Could you say more about what you'd like this design to show? Some more details "
    'would really help.'
)
"""str: v32NeedsInputAsk -- used when v33 GROUND returned no question text."""

V32_NEEDS_INPUT_REASON = "The brief doesn't have enough to design from yet."
"""str: ``reason`` sent with the v32NeedsInputAsk question."""

# ── Editing / expand ────────────────────────────────────────────────────────

EDITING_STARTED = 'Proceeding to editing workflow.'
"""str: editingManager -- ``editing_started`` spinner frame (auto-continued by main.py)."""

MANUAL_EDIT_SINGLE = 'You can set the {label} exactly with its control in the right-side menu.'
"""str: manualEditHandler -- ``manual_edit`` frame for one manual control; the
frontend opens the control named by ``manual_op``. ``{label}`` is the control's
label from manual_operations.json."""

MANUAL_EDIT_MULTIPLE = (
    'These adjustments are quicker and more precise with the controls in the '
    'right-side menu -- you can make them there in a few clicks.'
)
"""str: manualEditHandler -- ``manual_edit`` frame when one request asks for
several manual controls at once (``manual_op`` is null)."""

EXPANSION_STARTED = 'Creating designs for multiple platforms...'
"""str: expandManager -- ``expansion_started`` spinner frame (auto-continued by main.py)."""

EXPAND_PLATFORMS_ASK = (
    "Which platforms would you like to resize this image for? "
    "(e.g. Instagram Reel, Facebook Post, YouTube Thumbnail)"
)
"""str: expandManager -- the request named no platform or ratio."""

EXPAND_SUCCESSFUL = 'Your graphic has been expanded successfully! What would you like to do next?'
"""str: successfulExpandHandler -- ``expand_successful``."""

# ── Critique / suggestions ──────────────────────────────────────────────────

CRITIQUE_UPLOAD_GROUNDING_ASK = (
    'In order to review this upload the Critic Agent needs to know what it is for (goal) '
    'and where you want to post it (platform)?'
)
"""str: critiquepipeline.grounding.resolveGrounding -- critique of an uploaded image
with no goal/platform known."""

CRITICISM_FALLBACK = (
    "Sorry it's not landing. Tell me what's off — the colours, "
    "the typography, the layout, or the wording — and I'll rework it. "
    "I can also show you a few different directions to pick from."
)
"""str: criticismHandler -- used when the LLM reply is empty or the call failed."""

# ── Layering ────────────────────────────────────────────────────────────────

LAYERING_OFFER = (
    'We can turn your design into multiple layers so you can edit them individually. '
    'Would you like to proceed?'
)
"""str: interruptBeforeLayering -- ``suggestions_executed``."""

LAYERING_EXECUTED = 'Your layering request has been executed. What do you want to do next?'
"""str: common_nodes.nextStep -- ``layering_executed``."""

LAYERING_SKIPPED = "You have chosen to skip layering. What do you want to do next?"
"""str: subgraphs.layering.layeringSkipped."""

# ── Non-image ───────────────────────────────────────────────────────────────

HELPER_AGENT_ERROR = "I encountered an error while processing your question: {error}"
"""str: common_nodes.helperAgent -- the RAG agent raised."""

# ── Moved from prompts.py (still importable from there) ─────────────────────

CRITIQUE_INTENT_ASK = (
    'Before I review this — what are you planning to do with this graphic? Tell me what '
    'it needs to achieve, who it\'s for, and where it will run (e.g. "a festive sale post '
    'for our Instagram, aimed at existing customers"). The more you tell me, the more '
    'useful the critique.'
)
"""str: The ``critic_intent`` question, asked before EVERY general critique.

A design can only be judged against what it is meant to do. Without this the
lenses fell back to generic taste ("more whitespace", "stronger contrast") or,
worse, to the project's sticky catalog ``design_goal`` — which is how an
uploaded magazine cover got told to add the CTA its "product launch" brief
required. Asking costs one turn and grounds all four lenses plus the arbiter.

This is a REAL question, not a spinner ping: unlike the ``critic_started``
frame it replaced, ``main.py`` must not auto-continue it (same treatment as
``layering_started``). The answer lands in ``critique_intent`` and, once
parsed, in ``critique_goal`` / ``critique_platform``.
"""

GRAPHIC_METADATA_ASK = CRITIQUE_INTENT_ASK
"""str: The ``text_input`` / ``kind: "graphic_metadata"`` ask-once question, sent by
suggestionManager when the selected graphic's row has no goal/platform and was never
asked about. Same copy as ``CRITIQUE_INTENT_ASK`` for now; a separate name lets it
change on its own later.
"""

EDIT_BRAND_CONFLICT_WARNING = (
    'Heads up: this edit could change your design in a way that goes against your brand '
    "details. We'll still go ahead with it — if you don't like the result, you can go "
    'back to the previous version once the edit is done.'
)
"""str: The ``brand_conflict_warning`` notice, sent before an edit that contradicts the
attached brand (see :func:`nodes.editBrandWarning`).

A disclaimer, not a question: ``main.py`` auto-continues it (same treatment as
``editing_started``) and the edit runs regardless. The reverting it points at is
the ``previous_image_url`` that ``editing_successful`` already carries.
"""

EDIT_EXECUTABILITY_WARNING = (
    'Heads up: edits like this often give poor results — Kognie struggles with precise or '
    "very small changes. We're still processing your request; if you'd rather not "
    'continue, you can stop the editing process.'
)
"""str: The ``edit_executability_warning`` notice, sent before an edit the image
model is unlikely to carry out well (see :func:`nodes.editExecutabilityWarning`).

A disclaimer, not a question: ``main.py`` auto-continues it (same treatment as
``brand_conflict_warning``) and the edit runs regardless. Stopping is the normal
generation cancel -- the frame is scoped ``"generation"`` so the stop button works.
"""

LAYERING_TEXT_AUTO_NOTE = 'Text is extracted if you specify it or draw a box around it.'
"""str: Told to the user at every point layering asks what to layer.

Text extraction runs in ``layers.detection`` *before* the pipeline branches
into Gemini concept generation or the frontend-bbox path, so it happens on both
routes and the user never has to ask for it.
"""

LAYERING_OBJECT_ASK = (
    "Which elements would you like to turn into layers? Name them "
    "(e.g. \"the chair and the lamp\"), or draw boxes around them in the graphic. "
    + LAYERING_TEXT_AUTO_NOTE
)
"""str: The ``layering_started`` question.

One question, two shapes of answer -- names or boxes -- and the reply tells the
backend which path to take (see ``nodes._parse_layering_choice``). It replaced a
two-option "auto or manual?" mode prompt, which asked the user to make a
routing decision that their answer already implies, and then asked a *second*
time which objects they meant.
"""


# SUGGESTION CREATOR PROMPT SET


SUGGESTION_TARGET_ASK = (
    'Please select the graphic you want to work with. Our Critic Agent will analyze it to '
    'explore ways to make it more brilliant.'
)
"""str: Asked before an automatic critique when a design has several graphics
(the original plus every "create more"), so the review is never run against a
graphic the user did not pick."""


DESIGN_SUGGESTION_STARTED = """
Our design critic agent has come up with some suggestions to improve your image. Would you like to see them ?
"""

# ── Multi-agent COT streaming ────────────────────────────────────────────────

COT_AGENT_NAMES = {
    "generateMoreManager": ("Layout Agent", "Picking a fresh layout for another take on your design"),
    "scopeGate": ("Design Goal Analyst", "Checking your request against the project's design goal"),
    "intentClarifier": ("Design Planner Agent", "Understanding your creative brief"),
    "intentRouter": ("Brand Agent", "Reviewing the brief for brand if exists"),
    "imageAnalyzerLayoutSelector": ("Layout Agent", "Analyzing your images and picking the best layout"),
    "generation": ("Image Creator Agent", "Rendering your design"),
}
"""dict[str, tuple[str, str]]: node key -> (agent display name, COT heading) streamed
to the frontend as ``cot_progress`` frames. Edit here to rename agents in the COT log."""
