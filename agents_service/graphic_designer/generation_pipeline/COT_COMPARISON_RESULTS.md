# CoT Architecture Comparison: Single Pro Stream vs. Multi-Call Flash

**Status: experimental finding, NOT yet wired in as the default. Decision pending.**

Tests whether splitting the long CoT reasoning call into 7 smaller, sequential
Gemini Flash-Lite calls (one per stage) beats the current approach — one long
Gemini Pro call, streamed, with sections detected via token-buffer tag-scanning.

Both approaches were run on **identical upstream context** (same clarifier +
router output, run once per case) so any difference is attributable to the
CoT mechanism itself, not upstream stochasticity.

## Results

| Case | Current (1× Pro, streamed) | Multi-call (7× Flash-Lite) | Speed | Cost |
|---|---|---|---|---|
| Hiring / LinkedIn | 25.3s, $0.0177 | 17.9s, $0.0033 | **29% faster** | **5.3× cheaper** |
| Grand Opening / Story | 23.8s, $0.0176 | 17.2s, $0.0033 | **28% faster** | **5.3× cheaper** |
| Menu Promo / with photo | 29.8s, $0.0193 | 19.1s, $0.0040 | **36% faster** | **4.8× cheaper** |

Counter-intuitive but confirmed: multi-call uses ~4-5× **more** total tokens
(shared context gets re-sent on every one of the 7 calls), yet is still far
cheaper, because Flash-Lite's per-token price is ~16× lower than Pro's. It's
also faster despite 7 sequential network round-trips — each Flash-Lite call
returns in 1.2-4.7s.

## Reliability

- **Event boundaries are deterministic** with multi-call (call returned = event
  fired) vs. heuristic with the current approach (regex-scans a live token
  buffer for section tags, with a 300-char "SUMMARY window" heuristic).
- **Two real bugs were found and fixed during testing** (both in the
  experimental harness, not the core idea):
  1. Multi-call initially invented a literal pixel dimension (`"100x60px"`)
     for a logo placeholder — every template bans this. Fixed by strengthening
     the per-stage negative-constraint wording; re-run confirmed zero pixel
     violations afterward.
  2. Multi-call initially ignored an attached photo entirely (only the
     no-photo template wording had been ported to the per-stage instructions).
     Fixed by adding photo-aware clauses per stage; re-run confirmed correct
     "supplied photo as hero, background extended" behavior afterward.

## Ambiguity / fabrication resolution

Same prompt that earlier fabricated a fake "$12" price (no price given):

| | Current (Pro) | Multi-call (Flash) |
|---|---|---|
| No price given | Invented **"ROOFTOP EXCLUSIVE"** filler | *"I will use 'FRESH, FRUITY, ICE COLD'... without inventing a fake price"* |

**Honest caveat:** this difference is mostly due to a new anti-fabrication
instruction added specifically while building the multi-call variant ("do not
invent specific factual claims") — not an inherent property of splitting into
calls. **This instruction should be back-ported into the current single-call
template regardless of which architecture is chosen** — it's a free win either
way.

## Open items before fully committing

1. **Confound not yet isolated**: this test changed two variables at once
   (split into calls + switch Pro→Flash-Lite). A "single-call-but-Flash"
   variant would show how much of the win is the model tier vs. the split.
2. **Per-stage retry logic** not yet built — 7 sequential calls means 7
   chances for a transient API error. This is an *advantage* once built
   (retry just the failed stage), but isn't implemented yet.

## Recommendation

Move to the multi-call / Flash-Lite approach — faster, ~5× cheaper, cleaner
event boundaries, no quality regression once the two bugs above were fixed.
Not yet switched as the default pending sign-off.

## Files

- `v5_4_cot_multicall.py` — the multi-call implementation (`run_multicall_fusion()`)
- `cot_comparison_experiment.py` — the comparison harness (needs a sample photo
  at `assets/hero_portrait_cocktails.png` for the with-photo case — not bundled
  in this package to keep it lean)
- Full per-case outputs (brief_review + final_prompt text for both approaches):
  `results/cot_comparison/<case_name>/` in the original repo
