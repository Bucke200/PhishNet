# Architecture — Phase 6 serving path

Every annotation is a measured value copied from a report; sources are in
the table below. Nothing here is estimated.

## Serving path

```text
browser tab
   │  POST /predict {url}
   ▼
phishnet.serving.app ── /health: pinned model + column hashes, thresholds,
   │                            Tier-2 mode
   │
   ├─ shortener host? ── yes ─► resolve redirects
   │                             GET, abort before body, ≤ 5 hops, 2 s
   │                             │ unresolved ─► "can't assess" (no score)
   │                             ▼ final URL
   ▼
Tier 1  row (a) LightGBM fast path          p50 0.45 ms in-process
   │    canonicalize → features → row →          ~7 ms HTTP in container
   │    booster_.predict  (no pandas, no wrapper)
   │
   ├─ score < 0.6493  ────────────► allow
   ├─ score ≥ 0.9269  ────────────► alert
   ▼ in band [0.6493, 0.9269)
Tier 2  (in-band rows only; out-of-band never calls it)
   ├─ detector hit ───────────────► alert (escalate, phase6-A)
   └─ otherwise:
        sealed ── replay Phase 5 verdicts      offline demo default
        live   ── fetcher (Playwright) → LLM  p50 1,308 ms; $0.367 / 1k
                   │ schema/refusal/timeout/      escalated
                   │ unfetchable ─► alert (fail closed)
                   ▼
        phishing → alert        benign / suspicious → allow
```

## Annotations and sources

| node | value | source |
|---|---|---|
| Tier-1 latency | 0.45 ms p50 in-process; ~7 ms HTTP in container | `reports/phase6.json` |
| band edges | `lower_edge` 0.6493, `t_alert` 0.9269 | `reports/phase4.json` |
| Tier-1 identity | max abs diff 0.0 vs the Phase 3 headline scorer | `reports/phase6.json` |
| LLM latency | p50 1,308 ms / p90 2,069 ms | `reports/phase4.md` |
| LLM cost | $0.367 per 1,000 escalated rows (Groq listed, 2026-09-18) | `reports/phase4.md` |
| detector / fail-closed | `phase6-A`; Retain prohibited | `reports/phase6.md`, `docs/production-gaps.md` §5 |
| shortener | ≤ 5 hops, 2 s, no body read | `docs/phase6-preregistration.md` (`phase6-D`) |
| out-of-band | Tier 2 never runs | `reports/phase6.md` |

## Frozen inputs

- model `ablation_lexical_gbm_model.pkl`, SHA256 `7b765bfc…4024f`;
- columns `ablation_lexical_feature_columns.pkl`, SHA256 `39d0e665…d79e`
  (79 columns);
- thresholds `t_alert` / `t_1pct` / `lower_edge` read from
  `reports/phase4.json` and verified at startup.

## Images

- Tier-1 serving image: `backend/Dockerfile` (no browser; fetches only the
  two row (a) artifacts).
- Live Tier-2 fetcher image: `backend/fetcher/Dockerfile` (Playwright), a
  separate service so the latency number is not measured beside a browser.
