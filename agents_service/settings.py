import os
from dotenv import load_dotenv

load_dotenv()

ENV = (os.getenv("ENV", "dev"))

NEURON_MODELS_HOME = (os.getenv("NEURON_MODELS_HOME", '/home/ec2-user/efs/compiled-models'))
NEURON_CORE_COUNT = int(os.getenv("NEURON_CORE_COUNT", 4))


GEN_Q_MODEL = os.getenv("GEN_Q_MODEL", 'BeIR/query-gen-msmarco-t5-large-v1')
CLUSTER_ENCODER = os.getenv("CLUSTER_ENCODER", 'sentence-transformers/distilbert-base-nli-stsb-quora-ranking')
BI_ENCODER = os.getenv("BI_ENCODER", "sentence-transformers/nq-distilbert-base-v1")
CROSS_ENCODER = os.getenv("CROSS_ENCODER", "cross-encoder/ms-marco-MiniLM-L-6-v2")
###6/9/2023 disable contentful_classifier
#CONTENTFUL_CLASSIFIER = os.getenv("CONTENTFUL_CLASSIFIER", "./contentful_classifier")
PARAPHRASE_MODEL = os.getenv("PARAPHRASE_MODEL", "sentence-transformers/paraphrase-mpnet-base-v2")

BING_SUBSCRIPTION_KEY = os.getenv("BING_SUBSCRIPTION_KEY")

SQS_QUEUE_URL = (os.getenv("COPILOT_QUEUE_URI", "EssayPipelineQueue"))
SQS_QUEUE_REGION = (os.getenv("SQS_QUEUE_REGION", 'us-west-2'))


TAG_SELECTOR = os.getenv("TAG_SELECTOR", "[role=main] p:not(nav p, [role=navigation] p), [role=main] img:not(nav img, [role=navigation] img), [role=main] ul:not(nav ul, [role=navigation] ul), [role=main] ol:not(nav ol, [role=navigation] ol), article p:not(nav p, [role=navigation] p), article img:not(nav img, [role=navigation] img), article ul:not(nav ul, [role=navigation] ul), article ol:not(nav ol, [role=navigation] ol)")
PROBLEMATIC_URLS = set(os.getenv("PROBLEMATIC_URLS", "").split(','))
TOPIC_SEARCH_TOP = int(os.getenv("TOPIC_SEARCH_TOP", 10))
MIN_TOKENS_PER_PARAGRAPH = int(os.getenv("MIN_TOKENS_PER_PARAGRAPH", 16))
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", 512))
SELECTION_THRESHOLD = float(os.getenv("SELECTION_THRESHOLD", 0.85))
SELECTION_THRESHOLD_DELTA = float(os.getenv("SELECTION_THRESHOLD_DELTA", 0.05))
MAX_SELECTED_QUESTIONS = int(os.getenv("MAX_SELECTED_QUESTIONS", 10))
UNIT_LIMIT = int(os.getenv("UNIT_LIMIT", 3))
PEEK_THRESHOLD = float(os.getenv("PEEK_THRESHOLD", 0.15))
CONTEXT_SIZE = int(os.getenv("CONTEXT_SIZE", 3))
BING_MAX_DOCS = int(os.getenv("BING_MAX_DOCS", 10))
MAX_SUMMARY_UNITS = int(os.getenv("MAX_SUMMARY_UNITS", 3))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", 3))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", 40))
TOPIC_SIMILARITY_THRESHOLD = float(os.getenv('TOPIC_SIMILARITY_THRESHOLD', 0.25))
ANSWER_TOPIC_SIMILARITY_THRESHOLD = float(os.getenv("ANSWER_TOPIC_SIMILARITY_THRESHOLD", 0.3))
QUESTION_TOPIC_SIMILARITY_THRESHOLD = float(os.getenv("QUESTION_TOPIC_SIMILARITY_THRESHOLD", 0.35))
MIN_CLUSTER_SCORE = float(os.getenv("MIN_CLUSTER_SCORE", 0.3))
MIN_KERNEL_CROSS_SCORE = float(os.getenv("MIN_KERNEL_CROSS_SCORE", 0.0))

# DOCUMENT_BUCKET = "uploadprocessor"
DOCUMENT_BUCKET = os.getenv('DOCUMENT_BUCKET', 'kha-documents')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
DOCUMENT_BASE_URL = os.getenv('DOCUMENT_BASE_URL')
DOCUMENT_VERSION = os.getenv('DOCUMENT_VERSION', '1.4')
MAX_DOC_SECTIONS = int(os.getenv("MAX_DOC_SECTIONS", 25))
DOCUMENT_SCORE_WEIGHT = float(os.getenv("DOCUMENT_SCORE_WEIGHT", 1.2))
DOC_UNIT_LIMIT = int(os.getenv("DOC_UNIT_LIMIT", 7))
DOCUMENT_SEARCH_THRESHOLD = float(os.getenv("DOCUMENT_SEARCH_THRESHOLD", 1.2))

FILE_EXTENSIONS = [".txt", ".pdf", ".docx",".doc", ".jpg", ".jpeg", ".png", ".xlsx", ".xls", ".csv"]
LOG_BUCKET = os.getenv("LOG_BUCKET", "kognie-essay-logs")
LOG_BUCKET_URL = os.getenv("LOG_BUCKET_URL", "https://kognie-essay-logs.s3.us-west-2.amazonaws.com/")
# HTTPS URL of the layout_store/ folder in the S3 bucket holding pre-created
# layout profiles / frontend manifests / wireframes, e.g.
# "https://<bucket>.s3.<region>.amazonaws.com/layout_store". Empty = disabled
# (only the local layout_store dir and on-the-fly generation are used).
LAYOUT_STORE_BASE_URL = os.getenv("LAYOUT_STORE_BASE_URL", "")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
BING_SEARCH_URL = os.getenv("BING_SEARCH_URL","https://api.bing.microsoft.com/v7.0/search")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Layering configurations
# ============================================================
GPU_SERVER_URL = os.getenv("GPU_SERVER_URL", "http://localhost:8002")
# .strip('"\'') guards against process managers (Docker --env-file, systemd
# EnvironmentFile=) that don't strip quotes the way python-dotenv/shells do.
GPU_MODE       = os.getenv("GPU_MODE", "http").strip().strip('"\'').lower()  # "http" for local dev, "sqs" for production

# SQS queues (only used when GPU_MODE=sqs)
AWS_REGION         = os.getenv("AWS_REGION", "us-east-1")
FLORENCE_QUEUE_URL = os.getenv("FLORENCE_QUEUE_URL", "")
SAM2_QUEUE_URL     = os.getenv("SAM2_QUEUE_URL", "")
GPU_JOBS_BUCKET    = os.getenv("GPU_JOBS_BUCKET", os.getenv("DOCUMENT_BUCKET", ""))

# SAM2
SAM_CHECKPOINT = "layers/sam2.1_hiera_large.pt"
SAM_CONFIG     = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Florence-2
FLORENCE_MODEL        = "microsoft/Florence-2-large"
FLORENCE_MIN_BOX_AREA = 500
FLORENCE_MAX_BOXES    = 5


# Gemini

GEMINI_MODEL   = "gemini-3-flash-preview"

# Stability AI

USE_ERASE_ENDPOINT = True

USE_DEPTH_REORDER   = True
DEPTH_MODEL         = "depth-anything/Depth-Anything-V2-Small-hf"
# Minimum IoU between two concept boxes to consider them potentially occluding
DEPTH_IOU_THRESHOLD = 0.05
# Minimum depth score difference (after percentile) to call one "in front"
DEPTH_DIFF_THRESHOLD = 0.08

# Pipeline
MAX_PER_CONCEPT    = 1
EXPAND_MASKS       = 10
INCLUDE_BACKGROUND = True
DEVICE             = "auto"

# Manual detection
MANUAL_DETECTION   = False

# Frontend-drawn bounding boxes (manual layering)
# A drawn box goes STRAIGHT TO SAM2, as drawn. The user already told us where
# the thing is; re-deriving that with Gemini + Florence per box is two model
# calls and their latency to answer a question that was already answered.
#
# Set True to restore the recursion: each box re-run through the same automatic
# pipeline the auto path uses (Gemini concepts -> Florence grounding), scoped to
# the crop inside that box, with the objects it finds becoming the layers. The
# case for it is a box dragged loosely around several distinct objects (a mug, a
# saucer and a spoon), where SAM2 given the whole box returns one blended mask
# while N tight masks composed together cut cleanly. The case against it is that
# MANUAL_BBOX_COMPOSE_LAYERS then merges those N masks back into ONE layer per
# drawn box anyway, so for the common single-object box the split is paid for and
# then undone.
MANUAL_BBOX_RECURSIVE_DETECTION = False
# Boxes smaller than this on either side are used as-is (too small to hold
# more than one object, and too small for Gemini/Florence to read reliably).
MANUAL_BBOX_MIN_SIDE  = 32
# Crops are upscaled to at least this short side before Gemini/Florence see
# them (a 90x70 crop of a poster is unreadable at native size), never past
# MAX_SIDE on the long side.
MANUAL_BBOX_CROP_MIN_SIDE = 640
MANUAL_BBOX_CROP_MAX_SIDE = 2048
# Interpolation invents no detail: past ~4x a crop is just a bigger blur, and
# a bigger payload for Gemini.
MANUAL_BBOX_CROP_MAX_UPSCALE = 4.0
# Two boxes found inside two overlapping drawn boxes above this IoU are the
# same object seen twice -- keep the first, drop the duplicate.
MANUAL_BBOX_DEDUPE_IOU = 0.8

# Selecting text to keep as a layer
# Text is layered automatically, and by default ALL of it is. But when the user
# points at a particular text -- by drawing a box round it, or by naming it --
# that becomes the selection: only what they pointed at stays a layer, and
# every other detected text is pasted back onto the background.
# A drawn box points at a text when it holds this much of the text...
TEXT_BOX_COVER_MIN = 0.6
# ...and the text is at least this much of what the box holds. The second test
# is what stops a big box drawn around a product from claiming the caption
# sitting next to it.
TEXT_BOX_FILL_MIN = 0.25
# Shortest normalized string that may match a named text. Two characters of
# OCR noise match almost any phrase by accident.
TEXT_NAME_MIN_CHARS = 3

# Per-box OCR for manual layering (drawn boxes)
# Each drawn box is OCR'd on its own crop and classified by how much of the box
# is text. No whole-image OCR runs: comparing whole-image text rects against
# drawn boxes needs both to live in the same coordinate space and to survive
# line grouping, and it decided nothing the crop cannot answer directly.
# A box whose text area reaches this share of the box is a TEXT box -- OCR's
# reading of it becomes the text layer(s) and the pixels are erased from the
# background. Below it the box is an OBJECT box and goes to SAM2 with its text
# left in place, which is correct: an object box holds little text by
# definition, and what it does hold belongs to the object.
TEXT_BOX_AREA_MIN = float(os.getenv("TEXT_BOX_AREA_MIN", 0.5))
# Off = fall back to the old whole-image OCR + coverage/containment path.
MANUAL_BOX_OCR = os.getenv("MANUAL_BOX_OCR", "1") not in ("0", "false", "False")
# One drawn box is one layer the user expects back: after peeling, the objects
# found inside a box are composed into a single layer. Off = keep them apart.
MANUAL_BBOX_COMPOSE_LAYERS = True

# Text removal
USE_TEXT_REMOVAL   = True
TEXT_REMOVAL_LANG  = "en"
TEXT_REMOVAL_GPU   = False
TEXT_REMOVAL_THRESHOLD = float(os.getenv("TEXT_REMOVAL_THRESHOLD", 0.35))
# Merge OCR line detections that form one sentence/paragraph into a single text
# block (see layers/textextraction/pipeline/text_grouping.py). Set to 0 to
# report every raw detection separately, as before.
TEXT_GROUPING = os.getenv("TEXT_GROUPING", "1") not in ("0", "false", "False")

# LLM text layering (layers/text_layering.py). nano-banana-2 removes every text
# from the image (keeping logos, product branding and wall art), objects are
# peeled from that textless image, the texts the user picked become transparent
# image layers, and every other text is pasted back onto the final background.
# Overrides TEXT_DEMAND_DRIVEN and MANUAL_BOX_OCR while on. Set to 0 to go back
# to those paths.
LLM_TEXT_LAYERING = os.getenv("LLM_TEXT_LAYERING", "1") not in ("0", "false", "False")
# New-vs-related generation check (graphic_designer/generation_decision.py).
# Before a generation that follows an earlier one, OpenAI's Decisions API is
# asked whether the new request is about the same subject. When it is not, the
# previous design's field answers, product photo/assets and chat are flushed so
# the mandatory questions are asked again. Set to 0 to always reuse them.
GENERATION_DECISION_ENABLED = os.getenv("GENERATION_DECISION_ENABLED", "1") not in ("0", "false", "False")
GENERATION_DECISION_MODEL = os.getenv("GENERATION_DECISION_MODEL", "gpt-6-luna")
# Below this probability of "same subject" the request is treated as new.
GENERATION_DECISION_THRESHOLD = float(os.getenv("GENERATION_DECISION_THRESHOLD", 0.5))
GENERATION_DECISION_TIMEOUT = float(os.getenv("GENERATION_DECISION_TIMEOUT", 8))
# Navigator routing through the same Decisions API (graphic_designer/
# navigator_decision.py): one choice call picks the route. Any failure falls
# back to the LLM classifier. Set to 0 to always use the LLM classifier.
NAVIGATOR_DECISION_ENABLED = os.getenv("NAVIGATOR_DECISION_ENABLED", "1") not in ("0", "false", "False")
# AI slop remover branch (graphic_designer/subgraphs/slop.py). On: the
# generation_completed frame offers to remove AI slop from the new graphic, and
# the navigator routes "remove the AI slop from this" to the same subgraph.
# Off until the removal mechanism itself lands -- the graph then behaves
# exactly as it did before the branch existed.
SLOP_REMOVAL_ENABLED = os.getenv("SLOP_REMOVAL_ENABLED", "0") not in ("0", "false", "False")
# Manual edit branch (graphic_designer/manual_edit_decision.py). Every edit is
# first checked against the frontend's manual controls listed in
# graphic_designer/manual_operations.json; one that a control does exactly
# ("brightness to 46%", "increase the corner radius") is handed back to the
# frontend as `manual_op` instead of going to the image model. Off until the
# frontend handles the manual_edit frame -- every edit then goes to the AI,
# exactly as before the branch existed. An empty operations list has the same
# effect.
MANUAL_EDIT_ENABLED = os.getenv("MANUAL_EDIT_ENABLED", "0") not in ("0", "false", "False")
# Probability at or above which a Decisions API predicate counts as true for
# the manual edit rules. Higher keeps more edits with the AI.
MANUAL_EDIT_THRESHOLD = float(os.getenv("MANUAL_EDIT_THRESHOLD", 0.5))
# Wrong-image check on the reply to an edit (graphic_designer/
# edit_reply_decision.py): at or above this probability, "not this one, I meant
# another image" gets an apology and a request to select the right image.
EDIT_WRONG_IMAGE_THRESHOLD = float(os.getenv("EDIT_WRONG_IMAGE_THRESHOLD", 0.5))
# COPYWRITER step in the v3.3 engine (kognie-generation-v33/copywriter.py): it
# writes every line that goes on the graphic from what the user meant, beside
# SUBJECT, so it adds no wall time. Facts (names, numbers, dates, URLs, quotes)
# stay exact, code checks the deck and the model gets at most one repair pass.
# Set to 0 to use GROUND's extracted text as before.
COPYWRITER_ENABLED = os.getenv("COPYWRITER_ENABLED", "1") not in ("0", "false", "False")
# Flat art-style direction (graphic_designer/style_presets.py): the project's
# selected style -- one per generation -- is appended to the brief handed to
# v33 as a labelled STYLE block, so every reasoning stage sees it. Set to 0 to
# generate without style direction even when a style is selected.
STYLE_DIRECTION_ENABLED = os.getenv("STYLE_DIRECTION_ENABLED", "1") not in ("0", "false", "False")
# How unpicked texts go back onto the background: "matte" composites the
# original glyph pixels (exact); "llm" asks nano-banana-2 to redraw them from
# the original, clamped to the text regions, and falls back to "matte".
LLM_TEXT_PASTE_BACK = os.getenv("LLM_TEXT_PASTE_BACK", "matte")

# Demand-driven text extraction: only extract/erase text when it is explicitly
# requested by the user OR when it occludes a requested object. Text that is
# fully contained within an object's boundary (logos, labels, screen text) is
# treated as *part of* the object and left intact. Set to False to revert to
# the old "extract everything, paste back what's unwanted" behaviour.
TEXT_DEMAND_DRIVEN = os.getenv("TEXT_DEMAND_DRIVEN", "1") not in ("0", "false", "False")

# When classifying text relative to an object, if this fraction (or more) of
# the text's area falls inside the object's bounding box the text is considered
# EMBEDDED (a logo, label, or printed graphic that is part of the object) and
# is left in place. Below this threshold the text extends beyond the object
# boundary and is treated as OCCLUDING — it must be erased so the object edge
# can be cleanly segmented.
TEXT_EMBEDDED_THRESHOLD = float(os.getenv("TEXT_EMBEDDED_THRESHOLD", 0.85))

# Whether a text that CROSSES a selected object's boundary gets erased so SAM2
# can cut a clean edge, then pasted back onto the finished background.
#
# ON: a text lying over the object being peeled has to go, or the object's mask
# cuts through letterforms. What matters is that it comes BACK, pixel-faithful,
# onto the finished background -- see _restore_texts_on_background, which now
# falls back to a full-rectangle paste rather than ever pasting nothing, and
# logs every text it could not restore instead of failing silently.
TEXT_ERASE_OCCLUDING = os.getenv("TEXT_ERASE_OCCLUDING", "1") not in ("0", "false", "False")

# Area alone cannot tell "inside" from "straddling": a wide caption crossing an
# object's left AND right edge can still leave 90% of its own area within the
# bbox. So a text also has to stay within the object's outline to count as
# embedded -- it may overhang any one side by no more than this many pixels...
TEXT_EMBED_OVERHANG_PX = int(os.getenv("TEXT_EMBED_OVERHANG_PX", 4))
# ...or this fraction of the text's own short side, whichever is larger. OCR
# quads sit a little loose around their glyphs and an object bbox is itself
# approximate, so demanding a pixel-exact fit would call real logos occluding.
TEXT_EMBED_OVERHANG_FRAC = float(os.getenv("TEXT_EMBED_OVERHANG_FRAC", 0.15))

# =============================================================

GEN_PRICES = {
  "gpt-5": 0.0225,
  "gpt-5-mini": 0.0045,
  "gpt-5-nano": 0.0009,
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
  "gemini-1.5-pro": 0.0125,
  "gemini-1.5-flash": 0.0075,
  "gemini-2.0-flash": 0.001,
  "gemini-2.5-pro" : 0.023, # new
  "gemini-2.5-flash" : 0.0056, # new
  "gemini-2.5-flash-lite-preview-06-17" : 0.001, # new
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
  'gemini-2.0-flash-image-preview': 0.06,
  'banana': 0.06,
  'imagen-3': 0.12,
  'gpt-image-2.5-flare': 0.06,
  'search-recolor' : 0.08,
  # Masked recolor rides the same marked-edit path as search-recolor.
  'recolor': 0.08,
  'search-replace': 0.08,
  'replace': 0.08,
  'search-erase': 0.2,
  'erase': 0.1,
  'search-erase': 0.06,
  'upscale-conservative': 0.5,
  'upscale-creative' : 0.5,
  'upscale-fast' : 0.02,
  'bg-remove' : 0.04,
  'reference' : 0.1,
  'expand' : 0.04,
}

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

MAX_TOKENS={ 'gpt-5': 400000, 'gpt-5-mini': 400000, 'gpt-5-nano': 400000,
            'gpt-4o': 128000, 'gpt-4o-mini': 128000, 'gpt-4': 8192, 'gpt-4.1': 1048576, 'gpt-4.1-mini': 1048576,'gpt-4.1-nano': 1048576, 'o4-mini' : 200000, 'o1':200000, 'o1-mini':200000, 'o3-mini': 200000,'o3-mini-high': 200000, 'gpt-3.5-turbo': 16385,
            'claude-3-5-sonnet-20240620': 200000, 'claude-3-haiku-20240307': 200000, "claude-opus-4-0": 200000, "claude-sonnet-4-0": 200000, "claude-3-7-sonnet-latest": 200000, "sd-ultra": 128000, "sd-core": 128000, "sd-3.5-large-turbo": 128000, "sd-3.5-medium": 128000,
            "dall-e-3 SD": 128000, "dall-e-3 HD": 128000, "dall-e-2": 128000, "flux-pro-1.1": 128000, "flux-pro": 128000, "flux-kontext-pro": 128000, "flux-kontext-max": 128000, "gemini-2.0-flash-image-preview": 128000, 'imagen-3': 128000, 'search-recolor': 128000, 'recolor': 128000, 'search-replace': 128000, 'replace': 128000, 'erase': 128000,
            'upscale-conservative': 128000, 'upscale-creative': 128000, 'upscale-fast': 128000, 'bg-remove': 128000, 'reference': 128000, 'expand': 128000,
            'gemini-1.5-pro': 1048576, 'gemini-1.5-flash': 2097152, 'gemini-2.0-flash':1048576, "gemini-2.5-pro": 1048576, "gemini-2.5-flash": 1048576, "gemini-2.5-flash-lite-preview-06-17": 1000000, 'deepseek-chat' : 68000,
  'deepseek-reasoner' : 96000, 'open-mistral-nemo': 128000, 'banana': 400000, "claude-sonnet-4-5-20250929": 200000, # new
            'mistral-small-latest': 32000, 'mistral-large-latest': 128000, 'llama3': 128000, 'gpt-image-2.5-flare': 128000, 'search-erase': 128000
        }


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

