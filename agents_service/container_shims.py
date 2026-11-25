"""Make the copied Kognie agent code run inside a small, stateless container.

Import this module before anything from ``graphic_designer``. It changes no copied
file; it only adjusts the environment around them:

1. Stub modules for code paths these three agents never reach but that are imported
   at module load (the RAG research pipeline, OCR layer detection, FAISS). Their real
   dependencies pull in torch and transformers, which would make the image gigabytes.
2. An in-memory object store in place of AWS S3. The nodes upload every result with
   ``s3_client.upload_fileobj`` and then build an S3 URL by hand. Here uploads land in
   ``STORE`` under the same key, and any HTTP GET of such a URL is answered from
   ``STORE``. The service returns the bytes to the Worker, which writes them to R2.
"""
from __future__ import annotations

import io
import os
import sys
import threading
import types

# ------------------------------------------------------------------ 1. stubs
class _Anything(types.ModuleType):
    """A module whose every public attribute is a placeholder class."""

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        placeholder = type(name, (), {"__init__": lambda self, *a, **k: None})
        setattr(self, name, placeholder)
        return placeholder


def _stub(name: str, package: bool = False) -> None:
    if name in sys.modules:
        return
    mod = _Anything(name)
    mod.__file__ = f"<container stub {name}>"
    if package:
        mod.__path__ = []
    sys.modules[name] = mod


for _name in ("research_pipeline",):
    _stub(_name)
for _name in ("layers", "langchain_community", "langchain_community.vectorstores"):
    _stub(_name, package=True)
for _name in ("layers.detection", "layers.segmentation"):
    _stub(_name)

# ------------------------------------------------------------------ 2. object store
BUCKET = "kognie-container-store"
REGION = "local"
URL_PREFIX = f"https://{BUCKET}.s3.{REGION}.amazonaws.com/"
os.environ["DOCUMENT_BUCKET"] = BUCKET
os.environ["AWS_REGION"] = os.environ["REGION_NAME"] = REGION

STORE: dict[str, tuple[bytes, str]] = {}      # key -> (bytes, content type)
_lock = threading.Lock()


class LocalS3:
    """The two boto3 S3 calls the copied code makes."""

    def upload_fileobj(self, fileobj, bucket, key, ExtraArgs=None):  # noqa: N803 - boto3 name
        data = fileobj.read() if hasattr(fileobj, "read") else bytes(fileobj)
        ctype = (ExtraArgs or {}).get("ContentType", "application/octet-stream")
        with _lock:
            STORE[key] = (data, ctype)

    def put_object(self, Bucket, Key, Body, ContentType="application/octet-stream", **_):  # noqa: N803
        data = Body.read() if hasattr(Body, "read") else bytes(Body)
        with _lock:
            STORE[Key] = (data, ContentType)


def stored(url: str) -> tuple[bytes, str] | None:
    """Bytes for a URL the nodes built for an upload, if it is one of ours."""
    if not url.startswith(URL_PREFIX):
        return None
    with _lock:
        return STORE.get(url[len(URL_PREFIX):])


def forget(url: str) -> None:
    """Drop a stored upload once its bytes have been handed back."""
    if url.startswith(URL_PREFIX):
        with _lock:
            STORE.pop(url[len(URL_PREFIX):], None)


def install_store() -> None:
    """Point the copied config at LocalS3 and serve its URLs to ``requests``."""
    import requests
    from requests.adapters import BaseAdapter
    from requests.models import Response

    class _StoreAdapter(BaseAdapter):
        def send(self, request, **_):
            resp = Response()
            hit = stored(request.url)
            resp.status_code = 200 if hit else 404
            resp.headers["Content-Type"] = hit[1] if hit else "text/plain"
            resp.raw = io.BytesIO(hit[0] if hit else b"not found")
            resp.url = request.url
            resp.request = request
            return resp

        def close(self):
            pass

    original_init = requests.Session.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.mount(URL_PREFIX, _StoreAdapter())

    requests.Session.__init__ = patched_init

    from graphic_designer import config
    config.s3_client = LocalS3()
    config.DOCUMENT_BUCKET = BUCKET
    config.REGION_NAME = REGION
