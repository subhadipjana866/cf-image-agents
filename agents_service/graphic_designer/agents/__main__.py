"""Try the generation and edit agents from the shell.

    # generation: the brief is what the input-collector would hand over
    python -m graphic_designer.agents generate --prompt "Diwali sale for Kavya Sarees" \\
        --goal "Festive / Seasonal Sale" --platform "Instagram Portrait Post" \\
        --field discount="30% off" --field dates="Oct 20-25" \\
        "Create this design." "Give me 2 more variations."

    # editing: any image URL or local path, then one or more user turns
    python -m graphic_designer.agents edit --image poster.webp \\
        "Make the headline bigger, background navy, round the corners, then square for Instagram" \\
        "undo the last change"

    # no image yet: the agent asks the user to select one; a turn "image:<url>" is a
    # frontend reply carrying only the selected image ("image:<url>|text" adds text)
    python -m graphic_designer.agents edit "remove the banana" "image:tests/test.jpeg"

    # interactive: --chat instead of turns; "/select <url> [text]" sends a selection
    python -m graphic_designer.agents edit --image poster.webp --chat

    # full path: navigation (interpret + select, printed step by step), then the agent
    python -m graphic_designer.agents chat --chat
    python -m graphic_designer.agents chat --image tests/test.jpeg "remove the banana and give me 2 more"

Each turn prints the tool calls in order, the agent's reply, the versions and cost.
Without AWS keys in .env, images are written to graphic_designer/agents/output/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphic_designer.agents import (Brief, DesignSession, build_edit_agent,  # noqa: E402
                                     build_generation_agent)
from graphic_designer.agents.build import DEFAULT_MODEL  # noqa: E402


def _print_turn(new_messages: list, session: DesignSession, cost_before: float) -> None:
    step = 0
    for m in new_messages:
        for call in getattr(m, "tool_calls", None) or []:
            step += 1
            args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
            print(f"  {step}. {call['name']}({args})")
        if m.type == "tool":
            first = str(m.content).splitlines()[0] if m.content else ""
            print(f"       -> {first[:220]}")
    final = new_messages[-1]
    print(f"\nagent: {final.content}\n")
    if session.current:
        print(f"  latest: version {session.current.number}  {session.current.url}")
    print(f"  cost this turn: ${session.cost - cost_before:.4f}   session total: ${session.cost:.4f}")


def _as_url(session: DesignSession, image: str) -> str:
    """Upload a local file once, so every tool and node gets a normal URL."""
    if image.startswith(("http://", "https://")):
        return image
    from graphic_designer.agents.images import encode, load_image, upload
    data, ext = encode(load_image(image))
    return upload(data, ext, user_id=session.user_id, session_id=session.session_id,
                  meta_state={}, kind="edited")


def _run(agent, session: DesignSession, turns: list[str], chat: bool) -> None:
    from graphic_designer.agents.turns import run_turn
    history: list = []

    def one(text: str, selected: str | None = None) -> None:
        nonlocal history
        cost_before = session.cost
        print(f"\nyou: {text or ''}" + (f"   [selected image: {selected}]" if selected else ""))
        result = run_turn(agent, session, history, text=text,
                          selected_image=_as_url(session, selected) if selected else None)
        history = result.history
        if not result.new_messages:
            print("  (nothing to do)")
            return
        _print_turn(result.new_messages, session, cost_before)
        if result.needs_image:
            print("  >> frontend: show the image picker (select_image)")

    def parse(turn: str) -> tuple[str, str | None]:
        if turn.startswith("image:"):
            image, _, text = turn[6:].partition("|")
            return text, image
        if turn.startswith("/select "):
            image, _, text = turn[8:].strip().partition(" ")
            return text, image
        return turn, None

    for turn in turns:
        one(*parse(turn))
    if chat:
        print("\nChat mode: empty line or Ctrl-D to stop.")
        while True:
            try:
                text = input("you> ").strip()
            except EOFError:
                break
            if not text:
                break
            one(*parse(text))


def _parse(turn: str) -> tuple[str, str | None]:
    if turn.startswith("image:"):
        image, _, text = turn[6:].partition("|")
        return text, image
    if turn.startswith("/select "):
        image, _, text = turn[8:].strip().partition(" ")
        return text, image
    return turn, None


def _chat_loop(chat, session: DesignSession, turns: list[str], interactive: bool) -> None:
    def one(raw: str) -> None:
        text, image = _parse(raw)
        chat.turn(text, _as_url(session, image) if image else None)

    for turn in turns:
        one(turn)
    if interactive or not turns:
        print("Chat: type a message. '/select <image> [text]' sends an image selection. "
              "Empty line or Ctrl-D stops.")
        while True:
            try:
                raw = input("you> ").strip()
            except EOFError:
                break
            if not raw:
                break
            one(raw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="run the generation agent on a brief")
    g.add_argument("--prompt", required=True, help="the user's request")
    g.add_argument("--goal", help="catalog design goal; omit for freeform")
    g.add_argument("--platform", help="catalog platform")
    g.add_argument("--aspect", default=None, help="aspect ratio for freeform, e.g. 4:5")
    g.add_argument("--field", action="append", default=[], help="field answer, key=value")
    g.add_argument("--photo", help="product photo URL or path")
    g.add_argument("--photo-mode", default="reference", choices=["as_is", "reference"])
    g.add_argument("--logo", help="logo URL or path")
    g.add_argument("--style", help="style preset id or free text")
    g.add_argument("--num", type=int, default=1, help="images per freeform render")

    e = sub.add_parser("edit", help="run the edit agent on an image")
    e.add_argument("--image", help="image URL or local path to start from (optional)")
    e.add_argument("--logo", help="logo URL or path, for place_logo")

    c = sub.add_parser("chat", help="navigation first, then the generation or edit agent")
    c.add_argument("--image", help="image URL or local path already in the session (optional)")
    c.add_argument("--nav-model", default=DEFAULT_MODEL, help="model for interpret + select")
    c.add_argument("--prompt", help="preset brief for the first new design (as the collector would)")
    c.add_argument("--goal")
    c.add_argument("--platform")
    c.add_argument("--aspect")
    c.add_argument("--field", action="append", default=[], help="field answer, key=value")
    c.add_argument("--logo", help="logo URL or path")
    c.add_argument("--style")

    for p in (g, e, c):
        p.add_argument("--chat", action="store_true", help="keep chatting after the given turns")
        p.add_argument("turns", nargs="*", help="user messages, in order")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    session = DesignSession()
    if args.cmd == "chat":
        from graphic_designer.agents.chat import Chat
        if args.image:
            session.start_with_image(_as_url(session, args.image))
        preset = None
        if args.prompt:
            preset = Brief(prompt=args.prompt, design_goal=args.goal, platform=args.platform,
                           aspect_ratio=args.aspect,
                           field_answers=dict(f.split("=", 1) for f in args.field),
                           logo_url=args.logo, logo_mode="provided" if args.logo else "none",
                           style=args.style)
        elif args.logo:
            session.brief = Brief(prompt="", logo_url=args.logo, logo_mode="provided")
        chat = Chat(session, model=args.model, nav_model=args.nav_model, preset_brief=preset)
        _chat_loop(chat, session, args.turns, args.chat)
        return 0
    if args.cmd == "generate":
        fields = dict(f.split("=", 1) for f in args.field)
        session.brief = Brief(
            prompt=args.prompt, design_goal=args.goal, platform=args.platform,
            aspect_ratio=args.aspect, field_answers=fields, photo_url=args.photo,
            photo_mode=args.photo_mode if args.photo else None, logo_url=args.logo,
            logo_mode="provided" if args.logo else "none", style=args.style, num_images=args.num)
        agent = build_generation_agent(args.model)
        turns = args.turns or ([] if args.chat else ["Create this design."])
    else:
        if args.image:
            session.start_with_image(_as_url(session, args.image))
        if args.logo:
            session.brief = Brief(prompt="", logo_url=args.logo, logo_mode="provided")
        agent = build_edit_agent(args.model)
        turns = args.turns
        if not turns and not args.chat:
            ap.error("give at least one edit request, or --chat")
    _run(agent, session, turns, args.chat)
    return 0


if __name__ == "__main__":
    sys.exit(main())
