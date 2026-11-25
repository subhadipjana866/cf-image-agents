"""Image I/O for agent tools: fetch any source, save a result, tag it, upload it.

Uploads follow the recipe every graph node uses (``tagged`` XMP + public S3
object). Without AWS credentials the file is written under
``graphic_designer/agents/output`` instead, so tools still work locally.
"""
from __future__ import annotations

import base64
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image

from ..config import DOCUMENT_BUCKET, REGION_NAME, s3_client
from ..image_metadata import tagged
from ..utils import makeApiRequest

LOCAL_OUT = Path(__file__).parent / "output"


def load_bytes(src: str) -> bytes:
    """Bytes of a data URI, http(s) URL, file:// URL or local path."""
    if src.startswith("data:"):
        return base64.b64decode(src.split(",", 1)[1])
    if src.startswith(("http://", "https://")):
        return makeApiRequest("GET", src, timeout=(5, 60)).content
    path = Path(src[7:] if src.startswith("file://") else src)
    return path.read_bytes()


def load_image(src: str) -> Image.Image:
    img = Image.open(BytesIO(load_bytes(src)))
    img.load()
    return img


def to_data_uri(src_or_bytes: str | bytes) -> str:
    """A data URI for any source; the edit and generation nodes only take these."""
    if isinstance(src_or_bytes, str) and src_or_bytes.startswith("data:"):
        return src_or_bytes
    data = load_bytes(src_or_bytes) if isinstance(src_or_bytes, str) else src_or_bytes
    fmt = (Image.open(BytesIO(data)).format or "PNG").lower()
    return f"data:image/{'jpeg' if fmt == 'jpg' else fmt};base64,{base64.b64encode(data).decode()}"


def has_transparency(img: Image.Image) -> bool:
    if img.mode not in ("RGBA", "LA", "PA") and "transparency" not in img.info:
        return False
    alpha = img.convert("RGBA").getchannel("A")
    return alpha.getextrema()[0] < 255


def encode(img: Image.Image, fmt: Optional[str] = None) -> tuple[bytes, str]:
    """(bytes, extension). Transparent images stay PNG unless a format is forced."""
    fmt = (fmt or ("png" if has_transparency(img) else "webp")).lower()
    buf = BytesIO()
    if fmt in ("jpg", "jpeg"):
        img.convert("RGB").save(buf, "JPEG", quality=95)
        return buf.getvalue(), "jpg"
    if fmt == "png":
        img.save(buf, "PNG")
        return buf.getvalue(), "png"
    img.convert("RGBA" if has_transparency(img) else "RGB").save(buf, "WEBP", quality=95)
    return buf.getvalue(), "webp"


def upload(data: bytes, ext: str, *, user_id: str, session_id: str, meta_state: dict,
           kind: str, model: str = "", result: Optional[dict] = None) -> str:
    """Tag and upload; returns a public URL (or a file:// URL when S3 is not set up)."""
    body = tagged(BytesIO(data), meta_state, kind, model, result)
    name = f"agent_{kind}_{time.time():.6f}.{ext}"
    content_type = {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}[ext]
    if s3_client is not None:
        key = f"{user_id}/{session_id}/{name}"
        s3_client.upload_fileobj(body, DOCUMENT_BUCKET, key,
                                 ExtraArgs={"ContentType": content_type, "ACL": "public-read"})
        return f"https://{DOCUMENT_BUCKET}.s3.{REGION_NAME}.amazonaws.com/{key}"
    LOCAL_OUT.mkdir(exist_ok=True)
    path = LOCAL_OUT / name
    path.write_bytes(body.getvalue())
    return path.resolve().as_uri()
