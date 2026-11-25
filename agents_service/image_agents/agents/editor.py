"""Image Editor: one agent with a set of edit tools.

A planning call chooses the tool calls; the tools do the work. Several chained
tasks can be planned in one call (`run_batch`). Deterministic
tools are implemented here with Pillow. Generative tools go through a
`GenerativeBackend` that you supply (an inpainting or image-editing model).
Each tool call creates a new asset version; the agent reports the last one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageEnhance, ImageOps

from ..schemas import Outcome, TaskEnvelope
from ..steps import render
from .base import AgentContext


class GenerativeBackend(Protocol):
    def edit(self, image: Image.Image, instruction: str,
             box: Optional[list[int]] = None) -> Image.Image: ...


# ------------------------------------------------------------- deterministic tools
def _crop(img: Image.Image, box: Optional[list[int]] = None,
          aspect_ratio: Optional[str] = None) -> Image.Image:
    if box:
        return img.crop(tuple(box))
    if not aspect_ratio:
        raise ValueError("crop needs a box or an aspect_ratio")
    w_part, h_part = (float(x) for x in aspect_ratio.split(":"))
    w, h = img.size
    target = w_part / h_part
    if w / h > target:                      # too wide: trim the sides
        new_w = round(h * target)
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))
    new_h = round(w / target)
    top = (h - new_h) // 2
    return img.crop((0, top, w, top + new_h))


def _resize(img: Image.Image, width: int, height: int) -> Image.Image:
    return img.resize((int(width), int(height)), Image.LANCZOS)


def _round_corners(img: Image.Image, radius: int) -> Image.Image:
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, *img.size), radius=int(radius), fill=255)
    out = img.convert("RGBA")
    out.putalpha(ImageChops.multiply(out.getchannel("A"), mask))
    return out


def _adjust(img: Image.Image, brightness: float = 1.0, contrast: float = 1.0,
            saturation: float = 1.0, warmth: float = 0.0) -> Image.Image:
    """Factors of 1.0 leave the image unchanged. `warmth` runs from -1 to 1."""
    alpha = img.getchannel("A") if img.mode == "RGBA" else None
    rgb = img.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
    rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    rgb = ImageEnhance.Color(rgb).enhance(saturation)
    if warmth:
        r, g, b = rgb.split()
        r = r.point(lambda v: min(255, round(v * (1 + 0.15 * warmth))))
        b = b.point(lambda v: min(255, round(v * (1 - 0.15 * warmth))))
        rgb = Image.merge("RGB", (r, g, b))
    if alpha is not None:
        rgb.putalpha(alpha)
    return rgb


def _grayscale(img: Image.Image) -> Image.Image:
    alpha = img.getchannel("A") if img.mode == "RGBA" else None
    out = ImageOps.grayscale(img.convert("RGB")).convert("RGB")
    if alpha is not None:
        out.putalpha(alpha)
    return out


def _recolor_region(img: Image.Image, box: list[int], color: str) -> Image.Image:
    """Recolour a rectangular region in place, keeping its light and shade."""
    out = img.convert("RGBA")
    region = out.crop(tuple(box))
    tinted = ImageOps.colorize(ImageOps.grayscale(region.convert("RGB")),
                               black="black", white="white",
                               mid=ImageColor.getrgb(color)).convert("RGBA")
    tinted.putalpha(region.getchannel("A"))
    out.paste(tinted, tuple(box[:2]))
    return out


@dataclass
class Tool:
    name: str
    description: str
    args: dict[str, str]
    method: str                          # "deterministic" | "generative"
    fn: Optional[Callable[..., Image.Image]] = None


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool("crop", "Crop the canvas to a pixel box or to an aspect ratio, centred.",
             {"box": "[left, top, right, bottom] in pixels, optional",
              "aspect_ratio": "'W:H', optional"}, "deterministic", _crop),
        Tool("resize", "Resize the canvas.", {"width": "pixels", "height": "pixels"},
             "deterministic", _resize),
        Tool("round_corners", "Round the corners of the image.", {"radius": "pixels"},
             "deterministic", _round_corners),
        Tool("adjust", "Adjust tone and colour of the whole image.",
             {"brightness": "factor, 1.0 = unchanged", "contrast": "factor",
              "saturation": "factor", "warmth": "-1 (cooler) to 1 (warmer)"},
             "deterministic", _adjust),
        Tool("grayscale", "Convert the image to black and white.", {}, "deterministic",
             _grayscale),
        Tool("recolor_region", "Recolour a rectangular region in place.",
             {"box": "[left, top, right, bottom] in pixels", "color": "CSS colour name or hex"},
             "deterministic", _recolor_region),
        Tool("generative_edit",
             "Change image content: remove, add or replace an object, change the "
             "background, edit text that exists only as pixels.",
             {"instruction": "what to change", "box": "region to confine the change, optional"},
             "generative"),
    ]
}


class ImageEditorAgent:
    agent_id = "image-editor"

    def __init__(self, generative_backend: Optional[GenerativeBackend] = None):
        self._backend = generative_backend

    def _apply(self, tool: Tool, img: Image.Image, args: dict[str, Any]) -> Image.Image:
        if tool.method == "generative":
            if self._backend is None:
                raise RuntimeError("no generative backend is configured for the Image Editor")
            return self._backend.edit(img, args.get("instruction", ""), args.get("box"))
        return tool.fn(img, **args)

    def run(self, env: TaskEnvelope, ctx: AgentContext) -> Outcome:
        return self.run_batch([env], ctx)[0]

    def run_batch(self, envs: list[TaskEnvelope], ctx: AgentContext) -> list[Outcome]:
        """Plan several chained tasks in one model call, then carry them out in order.

        Each task works on the result of the one before. Returns one outcome
        per task carried out, in order, stopping after the first that is not
        completed; later tasks are left for another step. Raises if the plan
        does not cover the first task, so the caller can fall back to
        planning one task at a time.
        """
        head = envs[0]
        if not head.target_assets:
            return [Outcome(kind="failed", reason="no design was given to change")]
        current = head.target_assets[0]
        version = ctx.store.get(current)
        img = ctx.store.open(current)

        raw = ctx.llm.complete_json(
            "editor_plan",
            render(
                "editor_plan",
                tools=[{"name": t.name, "description": t.description, "args": t.args,
                        "method": t.method} for t in TOOLS.values()],
                asset={"asset_id": current.asset_id, "version": current.version,
                       "size": list(img.size), "text_blocks": version.text_blocks,
                       "layers": version.layers, "layout_spec": version.layout_spec},
                tasks=[{"id": e.task_id, "goal": e.goal, "constraints": e.constraints,
                        "assumptions": e.assumptions, "inputs": e.inputs,
                        "task_record": e.context, "resume": e.resume} for e in envs],
            ),
        )
        plans = raw.get("plans")
        if plans is None and "action" in raw and len(envs) == 1:
            plans = [raw]                        # bare single-task shape
        plans = [p for p in plans or [] if isinstance(p, dict)]
        if len(envs) == 1 and len(plans) == 1:
            plans[0]["task_id"] = head.task_id
        by_task = {p.get("task_id"): p for p in plans}
        if head.task_id not in by_task:
            raise ValueError("the edit plan does not cover the first task")

        outcomes: list[Outcome] = []
        for env in envs:
            plan = by_task.get(env.task_id)
            if plan is None:                     # not planned: leave it for a later step
                break
            outcome, current, img = self._carry_out(env, plan, current, img, ctx)
            outcomes.append(outcome)
            if outcome.kind != "completed":
                break
        return outcomes

    def _carry_out(self, env: TaskEnvelope, plan: dict, current, img: Image.Image,
                   ctx: AgentContext):
        """Apply one task's plan. Returns (outcome, latest version, latest image)."""
        action = plan.get("action")
        if action == "input_required":
            return Outcome(kind="input_required", request=plan.get("request", ""),
                           options=plan.get("options") or [],
                           agent_state=env.agent_state), current, img
        if action == "assistance_required":
            return Outcome(kind="assistance_required", need=plan.get("need", ""),
                           acceptance=plan.get("acceptance", ""),
                           need_inputs={"target_assets": [current.model_dump()]},
                           agent_state=env.agent_state), current, img
        if action != "run_tools":
            return Outcome(kind="failed", reason=plan.get("reason")
                           or "the editor found no way to do this"), current, img

        start, changes, generative = current, [], False
        for call in plan.get("calls") or []:
            tool = TOOLS.get(call.get("tool"))
            if tool is None:
                return Outcome(kind="failed",
                               reason=f"unknown edit tool: {call.get('tool')}"), start, img
            try:
                img = self._apply(tool, img, call.get("args") or {})
            except Exception as err:  # a bad argument or a backend failure
                return Outcome(kind="failed", reason=f"{tool.name} failed: {err}",
                               retryable=tool.method == "generative"), start, ctx.store.open(start)
            current = ctx.store.new_version(current, img, made_by=self.agent_id, tool=tool.name,
                                            instruction=env.goal, method=tool.method)
            changes.append({"tool": tool.name, "args": call.get("args") or {}})
            generative = generative or tool.method == "generative"
        if not changes:
            return Outcome(kind="failed", reason="the editor planned no changes"), start, img

        chosen = plan.get("chosen_values") or {}
        summary = env.goal.rstrip(".") + ": done"
        if chosen:
            summary += " (" + ", ".join(f"{k}: {v}" for k, v in chosen.items()) + ")"
        return Outcome(kind="completed", assets=[current], generative=generative,
                       result={"changes": changes, "chosen_values": chosen},
                       summary=summary + "."), current, img
