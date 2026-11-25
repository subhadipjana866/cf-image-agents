"""Versioned asset store.

Versions are immutable. Graph state and prompts carry references only; the
pixels live on disk under `root`. This implementation keeps metadata in
memory. For production, back the metadata with a database table and the
files with blob storage; the interface can stay the same.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from .schemas import AssetRef


@dataclass
class Version:
    ref: AssetRef
    path: Path
    parent: Optional[AssetRef]
    made_by: str                 # an agent id, or "user"
    tool: Optional[str]
    instruction: str
    method: str                  # "deterministic" | "generative" | "manual" | "upload"
    reports: dict[str, Any] = field(default_factory=dict)
    text_blocks: list[dict] = field(default_factory=list)   # each marked specified/detected
    layers: list[dict] = field(default_factory=list)
    layout_spec: Optional[dict] = None


@dataclass
class Asset:
    asset_id: str
    session_id: str
    description: str
    versions: list[Version] = field(default_factory=list)


class AssetStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._assets: dict[str, Asset] = {}
        self._counter = 0

    # ------------------------------------------------------------------ writing
    def _save(self, asset_id: str, version: int, image: Image.Image | str | Path) -> Path:
        path = self.root / f"{asset_id}_v{version}.png"
        if isinstance(image, Image.Image):
            image.save(path)
        else:
            src = Path(image)
            if src.suffix.lower() == ".png":
                shutil.copyfile(src, path)
            else:
                Image.open(src).save(path)
        return path

    def new_asset(
        self,
        session_id: str,
        image: Image.Image | str | Path,
        *,
        description: str,
        made_by: str,
        instruction: str = "",
        method: str = "generative",
        layout_spec: Optional[dict] = None,
    ) -> AssetRef:
        self._counter += 1
        asset_id = f"a{self._counter}"
        ref = AssetRef(asset_id=asset_id, version=1)
        asset = Asset(asset_id=asset_id, session_id=session_id, description=description)
        asset.versions.append(
            Version(ref, self._save(asset_id, 1, image), None, made_by, None, instruction,
                    method, layout_spec=layout_spec)
        )
        self._assets[asset_id] = asset
        return ref

    def new_version(
        self,
        parent: AssetRef,
        image: Image.Image | str | Path,
        *,
        made_by: str,
        tool: Optional[str] = None,
        instruction: str = "",
        method: str = "deterministic",
    ) -> AssetRef:
        asset = self._assets[parent.asset_id]
        number = len(asset.versions) + 1
        ref = AssetRef(asset_id=asset.asset_id, version=number)
        prev = self.get(parent)
        asset.versions.append(
            Version(ref, self._save(asset.asset_id, number, image), parent, made_by, tool,
                    instruction, method, layout_spec=prev.layout_spec)
        )
        return ref

    def attach_report(self, ref: AssetRef, kind: str, data: Any) -> None:
        self.get(ref).reports[kind] = data

    def set_text_blocks(self, ref: AssetRef, blocks: list[dict], *, source: str) -> None:
        """`source` is "specified" (rendered from a layout) or "detected"."""
        self.get(ref).text_blocks = [dict(b, source=source) for b in blocks]

    def add_layer(self, ref: AssetRef, layer: dict) -> None:
        self.get(ref).layers.append(layer)

    # ------------------------------------------------------------------ reading
    def get(self, ref: AssetRef) -> Version:
        return self._assets[ref.asset_id].versions[ref.version - 1]

    def exists(self, ref: AssetRef) -> bool:
        asset = self._assets.get(ref.asset_id)
        return bool(asset) and 1 <= ref.version <= len(asset.versions)

    def latest(self, asset_id: str) -> AssetRef:
        return self._assets[asset_id].versions[-1].ref

    def open(self, ref: AssetRef) -> Image.Image:
        return Image.open(self.get(ref).path).convert("RGBA")

    def describe(self, session_id: str, focus: Optional[AssetRef] = None) -> list[dict]:
        """The assets block for prompts: metadata only, latest version per asset."""
        out = []
        for asset in self._assets.values():
            if asset.session_id != session_id:
                continue
            v = asset.versions[-1]
            out.append(
                {
                    "asset_id": asset.asset_id,
                    "latest_version": v.ref.version,
                    "description": asset.description,
                    "produced_by": v.made_by,
                    "method": v.method,
                    "in_focus": bool(focus and focus.asset_id == asset.asset_id),
                    "text_blocks": [
                        {"id": b.get("id"), "text": b.get("text"), "role": b.get("role")}
                        for b in v.text_blocks
                    ],
                    "layers": [layer.get("element") for layer in v.layers],
                }
            )
        return out

    def analysis(self, ref: AssetRef) -> dict:
        """What is already known about a version, for the readiness check."""
        v = self.get(ref)
        return {
            "text_blocks": v.text_blocks,
            "layers": v.layers,
            "layout_spec": v.layout_spec,
            "reports": list(v.reports),
        }
