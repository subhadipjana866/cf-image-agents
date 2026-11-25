"""Test central navigation: does the orchestrator pick the right agent from the cards?

    python -m agent_navigation                       # every case in cases.yaml
    python -m agent_navigation --only edit-,layer-   # cases whose id starts with these
    python -m agent_navigation --say "make it pop"   # one input, full routing printed
    python -m agent_navigation --chat                # type inputs, keeps history
    python -m agent_navigation --model openai:gpt-5.4 --select-model openai:gpt-5.4-mini
    python -m agent_navigation --disable slop-terminator   # as with SLOP_REMOVAL_ENABLED=0

Each input costs two model calls (interpret, then one batched select), plus a
retry call only if a batched reply leaves a task out. Results go to
agent_navigation/results/<timestamp>.md and .json.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_navigation.llm import TrackedChatLLM  # noqa: E402
from agent_navigation.navigator import DEFAULT_ASSETS, Navigator, Route  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "openai:gpt-5.4"   # gpt-5.4-mini ignored not_for lines in testing

_PERSON = copy.deepcopy(DEFAULT_ASSETS[0]) | {
    "description": "AI-generated lifestyle photo ad: a woman holding a coffee cup in both hands, "
                   "headline at the top",
}
_EDITED = copy.deepcopy(DEFAULT_ASSETS[0]) | {
    "latest_version": 2, "produced_by": "image-editor", "method": "generative",
    "description": "Instagram post for a coffee shop grand opening; v2 changed the background "
                   "from cream to navy",
}
ASSET_PRESETS = {"default": DEFAULT_ASSETS, "none": [], "person": [_PERSON], "edited": [_EDITED]}


def passed(case: dict, got: list[str]) -> bool:
    if "expect_any" in case:
        return any(got == list(option) for option in case["expect_any"])
    if "expect_all" in case:
        return bool(got) and all(agent == case["expect_all"] for agent in got)
    expected = list(case["expect"])
    return sorted(got) == sorted(expected) if case.get("unordered") else got == expected


def expected_text(case: dict) -> str:
    if "expect_any" in case:
        return " | ".join(", ".join(o) for o in case["expect_any"])
    if "expect_all" in case:
        return f"all {case['expect_all']}"
    return ", ".join(case["expect"]) + (" (any order)" if case.get("unordered") else "")


def run_case(nav: Navigator, case: dict) -> dict:
    try:
        route = nav.route(case["input"], history=case.get("history"),
                          assets=ASSET_PRESETS[case.get("assets", "default")],
                          pending_task=case.get("pending_task"))
        got = route.agents
        return {"case": case, "route": route.to_dict(), "got": got, "ok": passed(case, got)}
    except Exception as err:  # keep the run going; report the case as failed
        return {"case": case, "route": None, "got": [f"error: {err}"], "ok": False}


def describe(route: Route) -> str:
    lines = [f"  input_type: {route.input_type}"]
    if route.clarification:
        lines.append(f"  clarify: {route.clarification['request']}  "
                     f"options={route.clarification.get('options')}")
    if route.pending_input:
        lines.append(f"  answers pending task {route.pending_input['task_id']} "
                     f"({route.pending_input['agent_id']}): {route.pending_input['input']}")
    for t in route.tasks:
        extra = f" depends_on={t.depends_on}" if t.depends_on else ""
        extra += f" supports={t.supports_task_id}" if t.supports_task_id else ""
        lines.append(f"  {t.id} [{t.performed_by}, {t.desired_result}] {t.goal}{extra}")
        target = t.agent_id or "-"
        lines.append(f"     -> {target} ({t.decision}, {t.confidence}) {t.analysis}")
        if t.missing_inputs:
            lines.append(f"     missing: {t.missing_inputs}")
        if t.capability_gap:
            lines.append(f"     gap: {t.capability_gap}")
    return "\n".join(lines)


def write_report(results: list[dict], usage: dict, model_desc: str) -> Path:
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ok = sum(r["ok"] for r in results)
    md = [f"# Navigation run {stamp}", "",
          f"Models: {model_desc}. Passed {ok}/{len(results)}. "
          f"Calls {usage['calls']}, tokens in {usage['input_tokens']}, "
          f"out {usage['output_tokens']}, model time {usage['seconds']} s.", "",
          "| Case | Input | Expected | Got | |", "| --- | --- | --- | --- | --- |"]
    for r in results:
        c = r["case"]
        md.append(f"| {c['id']} | {c['input']} | {expected_text(c)} | {', '.join(r['got'])} | "
                  f"{'pass' if r['ok'] else '**FAIL**'} |")
    failures = [r for r in results if not r["ok"] and r["route"]]
    if failures:
        md += ["", "## Failures", ""]
        for r in failures:
            md += [f"### {r['case']['id']}", "", "```", json.dumps(r["route"], indent=2), "```", ""]
    (out_dir / f"{stamp}.md").write_text("\n".join(md))
    (out_dir / f"{stamp}.json").write_text(json.dumps(
        {"models": model_desc, "usage": usage, "results": results}, indent=2, default=str))
    return out_dir / f"{stamp}.md"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--cases", default=str(HERE / "cases.yaml"))
    ap.add_argument("--only", help="comma-separated case-id prefixes")
    ap.add_argument("--say", help="route one input and print the full result")
    ap.add_argument("--chat", action="store_true", help="interactive; keeps conversation history")
    ap.add_argument("--assets", default="default", choices=sorted(ASSET_PRESETS),
                    help="asset preset for --say and --chat")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="langchain model string for both steps")
    ap.add_argument("--select-model", help="different model for the select step")
    ap.add_argument("--disable", default="", help="comma-separated agent ids to switch off")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    models = {"default": args.model}
    if args.select_model:
        models["select"] = args.select_model
    model_desc = ", ".join(f"{k}={v}" for k, v in models.items())
    llm = TrackedChatLLM(models)
    nav = Navigator(llm, disabled=tuple(a for a in args.disable.split(",") if a))
    usage = llm.usage.totals

    if args.say:
        route = nav.route(args.say, assets=ASSET_PRESETS[args.assets])
        print(describe(route))
        print(f"  agents: {route.agents}   usage: {usage()}")
        return 0

    if args.chat:
        history: list[dict] = []
        print(f"Navigation chat ({model_desc}). Empty line or Ctrl-D to stop.")
        while True:
            try:
                text = input("you> ").strip()
            except EOFError:
                break
            if not text:
                break
            route = nav.route(text, history=history, assets=ASSET_PRESETS[args.assets])
            print(describe(route))
            print(f"  agents: {route.agents}")
            history += [{"role": "user", "content": text},
                        {"role": "assistant", "content": "routed to " + ", ".join(route.agents)}]
        print(f"usage: {usage()}")
        return 0

    cases = yaml.safe_load(Path(args.cases).read_text())
    if args.only:
        prefixes = tuple(p.strip() for p in args.only.split(",") if p.strip())
        cases = [c for c in cases if c["id"].startswith(prefixes)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda c: run_case(nav, c), cases))

    for r in results:
        mark = "PASS" if r["ok"] else "FAIL"
        print(f"{mark}  {r['case']['id']:<32} expected {expected_text(r['case']):<40} "
              f"got {', '.join(r['got'])}")
    ok = sum(r["ok"] for r in results)
    totals = usage()
    report = write_report(results, totals, model_desc)
    print(f"\n{ok}/{len(results)} passed. {totals}. Report: {report}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
