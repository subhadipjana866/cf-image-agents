"""Design metadata written INSIDE every image this app produces.

Each generated file carries an XMP packet describing how it was made: the
design goal, the platform, the brand, the model, and -- where the v3.3 pipeline
produced it -- the locked strings printed on the graphic and the prompt the
image model was given. :func:`read_embedded` reads it back out of the file's
bytes, so a later step (an edit, a create-more, a critique, an offline tool)
can recover a design's brief from the image alone.

Unlike S3 object metadata, this travels with the file: download it, move it
between buckets, hand it to someone, and the description is still there. It
does NOT survive most social platforms, which strip XMP on upload.

**No re-encode.** The packet is spliced into the container as a chunk -- an
``XMP `` chunk for WebP, an ``iTXt`` chunk for PNG -- so the pixels the image
model returned are the exact pixels that reach S3. Re-saving through Pillow
would cost a lossy WebP generation on every upload and would flatten the alpha
channel the layering flow depends on.

Nothing here may break a generation: every entry point catches its own errors
and returns the image unchanged.
"""
from __future__ import annotations

import json
import struct
from datetime import datetime, timezone
from io import BytesIO

# Written by every upload site: "generated" is a fresh design, "variation" a
# create-more, and the rest name the operation that produced the new file.
KINDS = ("generated", "variation", "edited", "expanded", "suggestion",
         "logo-composited")

_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_PNG_XMP_KEYWORD = b"XML:com.adobe.xmp"
_XMP_OPEN, _XMP_CLOSE = "<kognie:data>", "</kognie:data>"

_PACKET = ('<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
           '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
           '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
           '<rdf:Description rdf:about="" xmlns:kognie="https://kognie.com/ns/1.0/">'
           + _XMP_OPEN + "{payload}" + _XMP_CLOSE +
           '</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end="w"?>')

# Kinds whose file was made FROM another image, so naming a parent is true.
# A fresh "generated" has none -- state still holds the previously selected
# image, and recording that as this one's parent would invent a lineage.
_HAS_PARENT = frozenset({"variation", "edited", "expanded", "suggestion",
                         "logo-composited"})


# ---------------------------------------------------------------------------
# What we record
# ---------------------------------------------------------------------------
def build_metadata(state: dict, kind: str, model: str = "", result: dict = None) -> dict:
    """The description to embed in one image.

    :param state: the graph state the image was produced from.
    :param kind: which operation produced it -- one of :data:`KINDS`.
    :param model: the image model, when the call site knows it better than
        ``state['model']`` (each executor picks its own API model id).
    :param result: the v3.3 run result for THIS image, where the caller has
        one. It is deliberately never read off ``state['v32_prev']``: that
        field still holds the PREVIOUS design while the current one is being
        uploaded, so an edited or freeform image would come out carrying
        another graphic's headline.
    """
    state = state or {}
    result = result or {}
    ground = result.get("ground") or {}
    size = state.get("target_size") or []
    meta = {
        "kind": kind,
        "design_goal": state.get("design_goal"),
        "platform": state.get("platform_canonical") or state.get("platform"),
        "aspect_ratio": result.get("aspect_ratio") or state.get("aspect_ratio"),
        "size": f"{size[0]}x{size[1]}" if len(size) == 2 else "",
        "model": model or state.get("model"),
        "style": state.get("style"),
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "user_id": state.get("user_id"),
        "session_id": state.get("session_id"),
        "generation": state.get("generation_count"),
        "brand_id": state.get("brand_id"),
        "brand_name": state.get("brand_name"),
        "freeform": True if state.get("is_freeform") else None,
        # What the graphic actually says, and the brief behind it.
        "locked_text": result.get("locked_text") or ground.get("locked_text"),
        "other_facts": ground.get("other_stated_facts"),
        "cta_requested": result.get("cta_requested"),
        "archetype": result.get("chosen_archetype"),
        "prompt": result.get("final_prompt"),
        "brief": state.get("clarified_intent") or state.get("prompt"),
        # v33 TYPESET's editable-text metadata: fonts, copy, colours and where
        # each line sits. create_more() re-typesets a variation from it.
        "text_layers": _text_layers(result.get("text_layers")),
    }
    if kind in _HAS_PARENT:
        meta["parent_url"] = state.get("current_image_url")
    return {k: v for k, v in meta.items() if v not in (None, "", [], {})}


def _text_layers(layers):
    """``text_layers`` without v33's local file name, which means nothing once
    the file is uploaded; ``None`` when there are no typeset elements."""
    if not isinstance(layers, dict) or not layers.get("elements"):
        return None
    return {k: v for k, v in layers.items() if k != "image"}


def packet(meta: dict) -> bytes:
    """``meta`` as an XMP packet. ``b""`` when there is nothing to say."""
    try:
        if not meta:
            return b""
        text = json.dumps(meta, ensure_ascii=False, default=str)
        # The JSON sits in XML character data, so these three have to be
        # escaped or the packet is malformed and every reader rejects it.
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return _PACKET.format(payload=text).encode("utf-8")
    except Exception as e:
        print(f"[image_metadata] packet build failed ({type(e).__name__}: {e})")
        return b""


# ---------------------------------------------------------------------------
# Splicing it into the file
# ---------------------------------------------------------------------------
def _webp_chunks(data: bytes) -> list:
    """``[(fourcc, payload)]`` of a RIFF/WebP file."""
    out, pos, end = [], 12, len(data)
    while pos + 8 <= end:
        fourcc = data[pos:pos + 4]
        (size,) = struct.unpack("<I", data[pos + 4:pos + 8])
        payload = data[pos + 8:pos + 8 + size]
        if len(payload) < size:
            raise ValueError("truncated WebP chunk")
        out.append((fourcc, payload))
        pos += 8 + size + (size & 1)        # chunks are padded to even length
    return out


def _riff(chunks: list) -> bytes:
    body = b"WEBP"
    for fourcc, payload in chunks:
        body += fourcc + struct.pack("<I", len(payload)) + payload
        if len(payload) & 1:
            body += b"\x00"
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _u24(value: int) -> bytes:
    return struct.pack("<I", value)[:3]


def _webp_with_xmp(data: bytes, xmp: bytes) -> bytes:
    """A WebP carrying ``xmp``, with its image data untouched.

    A plain WebP (a single VP8/VP8L chunk) cannot hold metadata at all, so it
    is promoted to the extended format: a VP8X header stating the canvas size
    and which optional chunks follow, then the image data, then ``XMP ``.
    """
    chunks = [c for c in _webp_chunks(data) if c[0] != b"XMP "]
    has_alpha = any(c[0] == b"ALPH" for c in chunks)
    if chunks and chunks[0][0] == b"VP8X":
        head = bytearray(chunks[0][1])
        head[0] |= 0x04                      # the "this file has XMP" flag
        chunks[0] = (b"VP8X", bytes(head))
    else:
        from PIL import Image
        with Image.open(BytesIO(data)) as im:
            width, height = im.size
            has_alpha = has_alpha or im.mode in ("RGBA", "LA", "PA")
        flags = 0x04 | (0x10 if has_alpha else 0)
        chunks.insert(0, (b"VP8X", bytes([flags]) + b"\x00\x00\x00"
                          + _u24(width - 1) + _u24(height - 1)))
    chunks.append((b"XMP ", xmp))
    return _riff(chunks)


def _png_chunks(data: bytes) -> list:
    """``[(type, payload)]`` of a PNG, signature excluded."""
    out, pos = [], len(_PNG_SIG)
    while pos + 8 <= len(data):
        (size,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + size]
        if len(payload) < size:
            raise ValueError("truncated PNG chunk")
        out.append((ctype, payload))
        pos += 12 + size                      # length + type + payload + CRC
        if ctype == b"IEND":
            break
    return out


def _png_with_xmp(data: bytes, xmp: bytes) -> bytes:
    """A PNG carrying ``xmp`` as the iTXt chunk the XMP spec defines."""
    import binascii

    body = (_PNG_XMP_KEYWORD + b"\x00"       # keyword
            + b"\x00\x00"                    # uncompressed, no compression method
            + b"\x00" + b"\x00"              # no language tag, no translated keyword
            + xmp)
    # Our own packet from an earlier pass is replaced, not stacked: re-tagging
    # an edited image would otherwise grow the file on every save.
    chunks = [(t, p) for t, p in _png_chunks(data)
              if not (t == b"iTXt" and p.startswith(_PNG_XMP_KEYWORD + b"\x00"))]
    # Before the first IDAT, not after: a decoder only has to have read the
    # chunks preceding the image data by the time it hands the image over, and
    # Pillow does exactly that -- an iTXt after IDAT is missing from `info`
    # unless the caller also forces a full load().
    at = next((i for i, (t, _) in enumerate(chunks) if t == b"IDAT"), len(chunks) - 1)
    chunks.insert(at, (b"iTXt", body))
    out = bytearray(_PNG_SIG)
    for ctype, payload in chunks:
        out += (struct.pack(">I", len(payload)) + ctype + payload
                + struct.pack(">I", binascii.crc32(ctype + payload) & 0xFFFFFFFF))
    return bytes(out)


def embed(image_bytes: bytes, meta: dict) -> bytes:
    """``image_bytes`` with ``meta`` spliced in as XMP.

    Returns the input unchanged for an empty description, an unsupported
    container, or any failure -- a metadata problem must never cost the user
    their image.
    """
    try:
        xmp = packet(meta)
        if not xmp:
            return image_bytes
        if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
            return _webp_with_xmp(image_bytes, xmp)
        if image_bytes[:8] == _PNG_SIG:
            return _png_with_xmp(image_bytes, xmp)
        print(f"[image_metadata] not a WebP or PNG ({image_bytes[:8]!r}) -- left untagged")
        return image_bytes
    except Exception as e:
        print(f"[image_metadata] embed failed ({type(e).__name__}: {e}) -- image unchanged")
        return image_bytes


def tagged(image, state: dict, kind: str, model: str = "", result: dict = None) -> BytesIO:
    """Upload-ready buffer for ``image``, with its design metadata inside.

    Takes raw bytes or a ``BytesIO`` (what the upload sites already hold) and
    always returns a ``BytesIO`` positioned at 0, so it drops straight into
    ``s3_client.upload_fileobj``.
    """
    try:
        if isinstance(image, BytesIO):
            image.seek(0)
            data = image.read()
        else:
            data = bytes(image)
        return BytesIO(embed(data, build_metadata(state, kind, model, result)))
    except Exception as e:
        print(f"[image_metadata] tagging failed ({type(e).__name__}: {e}) -- uploading as-is")
        if isinstance(image, BytesIO):
            image.seek(0)
            return image
        return BytesIO(image)


# ---------------------------------------------------------------------------
# Reading it back
# ---------------------------------------------------------------------------
def read_embedded(image_bytes: bytes) -> dict:
    """The description :func:`embed` wrote into an image, or ``{}``.

    Takes the image's bytes, so the caller is expected to be a step that has
    the file in hand already -- an edit, a composite, a create-more reference.
    """
    try:
        from PIL import Image

        with Image.open(BytesIO(image_bytes)) as im:
            raw = im.info.get("xmp") or im.info.get(_PNG_XMP_KEYWORD.decode()) or ""
        text = raw.decode("utf-8", "ignore") if isinstance(raw, bytes) else str(raw)
        start, end = text.find(_XMP_OPEN), text.find(_XMP_CLOSE)
        if start < 0 or end < 0:
            return {}
        payload = text[start + len(_XMP_OPEN):end]
        payload = payload.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
        return json.loads(payload)
    except Exception as e:
        print(f"[image_metadata] read failed ({type(e).__name__}: {e})")
        return {}


def carried(meta: dict) -> dict:
    """A v3.3-result-shaped view of an embedded description, for an upload
    that changes pixels but not one word of the design (a pasted logo).

    Passed to :func:`tagged` as ``result``, it carries the source file's
    locked strings and brief over to the new file -- without it the copy deck
    the critique checks a generated graphic against is lost the moment a logo
    is composited onto it. Never use it for an edit: an edit can change the
    wording, and the old deck would then describe text that is gone.
    """
    meta = meta or {}
    return {k: v for k, v in {
        "locked_text": meta.get("locked_text"),
        "ground": {"other_stated_facts": meta.get("other_facts")} if meta.get("other_facts") else None,
        "cta_requested": meta.get("cta_requested"),
        "chosen_archetype": meta.get("archetype"),
        "final_prompt": meta.get("prompt"),
        "aspect_ratio": meta.get("aspect_ratio"),
        "text_layers": meta.get("text_layers"),
    }.items() if v not in (None, "", [], {})}


def read_url(url: str, timeout: int = 30) -> dict:
    """:func:`read_embedded` for an image URL. Downloads the file to do it."""
    try:
        import requests

        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return read_embedded(resp.content)
    except Exception as e:
        print(f"[image_metadata] fetch failed for {url!r} ({type(e).__name__}: {e})")
        return {}
