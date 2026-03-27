# CLAUDE.md — Badminton Tier-1 Master Rules
## MANDATORY READING at every session start. These rules are non-negotiable and apply to every file, every line, every decision.

Last updated: 2026-03-23 | Version: Badminton-Tier1-V1 | Target score: 96+/100

---

## 1. ENFORCEMENT PROTOCOL — 5 Core Commandments

1. **NEVER take shortcuts** — no placeholder code, no stub implementations, no "TODO: implement later"
2. **NEVER simplify** — full-power models only; no lite/degraded versions
3. **NEVER use fallbacks** — if data is missing, raise a domain exception, never return a default probability
4. **NEVER claim without proof** — Glob → Read → count LOC → list methods → THEN make claims
5. **NEVER lie** — honest answers always, even if unwanted

---

## 2. ZERO REGRESSION POLICY (#1 Priority)

Before touching ANY file:
1. Identify ALL dependents of the file
2. Plan surgical minimum-touch fix
3. After every change: `python scripts/lock_regression_state.py verify`

**FORBIDDEN:**
- Refactoring while fixing a bug
- Renaming symbols that are in use
- Changing API response structures without full audit
- Removing "unused" imports without Grep verification

---

## 3. REGRESSION LOCK

After every session:
- Add new files to `scripts/lock_regression_state.py` FIXED_BUGS list
- Run `python scripts/lock_regression_state.py lock`
- Commit `lock_state.json`
- NEVER skip this step

---

## 4. AUTONOMOUS EXECUTION MODE

When user says "GO AUTONOMOUS": full access, no questions, best performance.

**Sequential Task Protocol:**
```
Execute → Check → Test → Debug → Re-Test → Validate → Save → Next Phase
```

**Truthfulness Protocol:** Honest answers always, even if the user won't like it.

**Proactive Improvements:** Always surface optimisations discovered during work.

---

## 5. CONTEXT LIMIT CONTINUATION PROTOCOL

Resume at the EXACT point of stopping — never restart, skip, or summarise past work.

Before starting the next phase, answer:
- "Did you check EVERY file?"
- "Did you skip anything?"
- "Is the current phase 100% complete?"

Session continuation checklist:
1. Read CLAUDE.md (this file)
2. Read PLATFORM_STATE.md
3. Identify exact stopping point
4. List all remaining tasks
5. Continue from that point

---

## 6. MANDATORY AUDIT PROTOCOL

Before claiming anything exists or is missing:
1. Run Glob to find files
2. Read the file
3. Count lines of code
4. List all methods/classes
5. THEN make your claim

**FORBIDDEN:** Claiming something is missing without Glob proof.

---

## 7–12. TRUTHFUL REPORTING RULES

| Rule | Description |
|---|---|
| Rule 7 | Never claim without verification |
| Rule 8 | Verify before writing status reports |
| Rule 9 | Evidence-based assessments only |
| Rule 10 | Session summaries can be wrong — always re-verify |
| Rule 11 | Inventory audit: Glob all modules, read services, report with proof |
| Rule 12 | Truthfulness over speed — slow and correct beats fast and wrong |

---

## 8. ABSOLUTE RULES

| Rule | Restriction |
|---|---|
| No Mock Data | Forbidden in production; use `raise NotImplementedError` |
| No Hardcoded Probabilities | Must come from RWP calculator → Markov engine |
| No Hardcoded Odds | Must come from derivative engine |
| Full Power Models Only | V1 plan (66 features, 8 ELO pools) — no lite/degraded versions |
| Data Validation | Results must be reproducible against curated tier-1 data ±0.01% |
| Bulletproof Backups | Zero data loss, <30s crash recovery |

---

## 9. FAKE DATA = REGRESSION (Zero Tolerance)

The following patterns in business logic are **immediate regressions**:
```python
return 0.5          # FORBIDDEN
return 15000.0      # FORBIDDEN
return 1.85         # FORBIDDEN — hardcoded odds
```

Only acceptable alternatives:
- Query real data source
- `raise NotImplementedError`
- `raise RuntimeError("data unavailable: <reason>")`
- `return None` with structured error log

**Detection pattern:**
```bash
grep -rn "return 0\.[0-9]\|return [0-9]\+\.[0-9]\+" badminton/
```

---

## 10. ML MODEL INTEGRITY RULES

| Rule | Requirement |
|---|---|
| Rule 13 | No label leakage — random pair swap, ~50% class balance (P1 win rate in [0.45, 0.55]) |
| Rule 14 | No data leakage — strict temporal ordering, features computed BEFORE ELO updates |
| Rule 15 | Continuous leakage monitoring at every pipeline stage |
| Rule 16 | Full power model deployment — all 66 features, 3-regime framework (R0/R1/R2) |
| Rule 17 | No synthetic/mock data in models — errors raise exceptions |

---

## 11. SOLID PRINCIPLES

- **SRP**: One reason to change per class
- **OCP**: Open for extension (interfaces/protocols), closed for modification
- **LSP**: Subclasses substitutable for base class
- **ISP**: Small focused interfaces — no fat interfaces
- **DIP**: Depend on abstractions, not concretions

---

## 12. BADMINTON FOLDER STRUCTURE RULES

```
badminton/
├── core/               # Atomic pricing unit (RWP), Markov engine, scoring rules
├── ml/                 # Feature engineering, ELO, training, inference
├── markets/            # Pre-match, live, derivatives (97 markets), SGP, outrights
├── feed/               # Optic Odds, Flashscore, BWF rankings, entity mapping
├── settlement/         # Grading service, void rules, score validation
├── agents/             # VaultAgent orchestrators and supervisors
├── api/                # Routes only — no business logic
├── scripts/            # Backfill, train, calibrate, regression lock
├── config/             # Sport constants, tournament tiers, feed config
└── tests/              # unit/, integration/, e2e/
```

**Import rules:**
- `core/` is the foundation — imported by everyone
- `ml/` imports from `core/` and `config/` only
- `markets/` imports from `core/`, `ml/`, `config/`
- `feed/` imports from `core/`, `config/` only
- `settlement/` imports from `core/`, `config/`
- `agents/` imports from all domains via interfaces
- `api/` imports from `agents/` only (no direct domain imports)

---

## 13. CODE QUALITY STANDARDS

- Type hints on ALL parameters, return types, and class attributes
- Custom domain exceptions with meaningful messages (not bare `Exception`)
- Structured logging via `structlog` — no `print()` statements
- Test coverage > 80%
- No bare `except:` — always catch specific exception types

---

## 14. AGENT DEVELOPMENT (VaultAgent Pattern)

```
BadmintonOrchestratorAgent
├── PreMatchSupervisorAgent
├── LiveSupervisorAgent
├── OutrightSupervisorAgent
├── SGPSupervisorAgent
└── MonitoringSupervisorAgent
```

**Latency targets:**
- p50 < 50ms (per-rally reprice)
- p95 < 100ms
- p99 < 200ms

---

## 15. API DEVELOPMENT

- Routes: `api/routes.py`
- Standard response format:
```json
{
  "success": true,
  "data": {...},
  "meta": {
    "request_id": "uuid",
    "timestamp": "ISO8601",
    "discipline": "MS",
    "tournament_tier": "SUPER_1000"
  }
}
```
- All endpoints must have Pydantic request/response models
- All services expose `/health`, `/health/ready`, `/health/live`

---

## 16. GIT CONVENTIONS

- Branches: `feature/{ticket}-{description}` / `bugfix/{ticket}-{description}`
- Commits: `type(scope): description`
  - e.g., `feat(markov): add doubles service-court state machine`
  - e.g., `fix(rwp): correct golden-point probability at 29-29`
- No committed secrets — use `.env.example`
- Model `.pkl` files must be explicitly `git add`-ed and pushed — never assume tracked

---

## 17. DEPLOYMENT

- Service name: `xg3-badminton` (port 8009)
- No secrets committed — `.env.example` only
- All services expose `/health`, `/health/ready`, `/health/live`
- `.pkl` Railway Trap: `.pkl` files must be explicitly `git add`-ed + pushed. Training = Push (inseparable steps)

---

## 18. BADMINTON-SPECIFIC CONSTANTS (V1 Plan)

| Parameter | Value | Source |
|---|---|---|
| Disciplines | MS, WS, MD, WD, XD | BWF |
| Games to win match | 2 (Best of 3) | BWF Laws of Badminton |
| Points to win game | 21 | BWF |
| Deuce rule | Both sides must reach 20-20 first | BWF |
| Golden point | At 29-29, next point wins (30 wins) | BWF |
| Serving rule | WINNER of rally serves next | BWF |
| Serves first in new game | WINNER of previous game | BWF (C-04 correction) |
| S1000 main draw size | 32 players | BWF (C-02 correction) |
| S750 main draw size | 32 players | BWF (C-02 correction) |
| S500 main draw size | 32 players | BWF |
| S300 main draw size | 32 players | BWF |
| Thomas Cup rubbers | First to 3 (5 max: MS1/MD1/MS2/MD2/MS3) | BWF (C-15 correction) |
| ELO pools | 8 (MS, WS, MD_pair, WD_pair, XD_pair, MD_individual, WD_individual, XD_individual) | V1 Plan |
| ML features | 66 across 9 groups | V1 Plan |
| MC simulations | 100,000 per match | V1 Plan |
| Market families | 15 | V1 Plan |
| Total markets | 97 | V1 Plan |
| QA gates | H1–H11 | V1 Plan |
| Tests target | 150+ | V1 Plan |
| RWP baseline MS | 0.515 (slight server advantage) | Literature |
| RWP baseline WS | 0.512 | Literature |
| RWP baseline Doubles | 0.508 | Literature |
| Brier threshold | 0.24 | V1 Plan |
| AUC threshold | 0.65 | V1 Plan |
| ECE threshold | 0.05 | V1 Plan |
| Live ghost trigger | 30 seconds | ADR-018 |
| Live suspend trigger | 180 seconds | ADR-018 |
| Outright reprice interval | 60 seconds | V1 Plan |

---

## 19. MANDATORY AGENT ROSTER

| Tier | Agents | Trigger |
|---|---|---|
| T1 | backend-engineer, frontend-engineer | EVERY backend/frontend change |
| T2 | code-specialist, architect-reviewer, QA engineer, QA automation | Every non-trivial task |
| T3 | ml-specialist, data-scientist, quant-analyst, trading-risk-reviewer, data-engineer | Model/data/odds work |
| T3B | MLOps Engineer | Any model training/evaluation/deployment |
| T4 | DevOps, Infrastructure, DB Architect, DB Developer, API Developer | Infrastructure changes |
| T6 | Sport Expert, Sport Trader, Odds Compiler, Liability Manager, Head of Sportsbook | Domain-specific badminton trading |

**Mandatory deployment rules:**
- T1 always-active (zero exceptions)
- QA after every implementation
- trading-risk-reviewer on ANY pricing code

---

## 20. AUDITOR CORRECTIONS — ALL 27 MUST BE RESPECTED

All corrections from ChatGPT (C-01..C-16) and Grok (G-01..G-11) are binding:

**CRITICAL:**
- C-01: Tournament registry is season-versioned (configurable, not hardcoded enum)
- C-02: Main draw sizes are 32 for S1000/S750/S500/S300 (NOT 64)
- C-03: Ranking points table is versioned (post-April 2024 structure)
- C-04: Server in new game = WINNER of previous game (not loser)
- C-13: BWF rules confirmed: Bo3/21/deuce-20-20/golden-29-29

**HIGH:**
- C-05: Explicit P0/P1/P2/P3 data tier separation in code
- C-07: `p_win_match` games_won_b==1 branch logic bug fixed
- C-08: Doubles service-court state tracked (XD/MD/WD rotation)
- C-09: 3 modeling targets: P(win), P(2-0|win), P(deuce)
- C-10: Pair chemistry: recency decay + partner-switch transfer
- C-12: Market depth: alt game handicaps, comeback markets, race ladders
- C-14: Sportsbook operator controls: stale state suspension, max liability
- C-16: Entity resolution is a first-class subsystem

**MUST IMPLEMENT (Grok):**
- G-01: Injury/withdrawal NLP signal from news corpus
- G-02: Venue-specific form (hall humidity, court surface type)
- G-03: Advanced live micro-markets (next rally length, smash winner)
- G-04: Team event rubber-specific fatigue
- G-05: Transfer learning for sparse data (new players/pairs)

---

## 21. QA GATES — ALL MUST PASS BEFORE GO-LIVE

| Gate | Condition | Pass Criteria |
|---|---|---|
| H1 | Overround | 4–18% for 2-way markets by tier |
| H2 | Model AUC | ≥ 0.65 all disciplines |
| H3 | Brier Score | ≤ 0.24 all disciplines |
| H4 | ECE | ≤ 0.05 post-calibration |
| H5 | No data leakage | ELO updated AFTER feature extraction |
| H6 | P1 win rate balance | [0.45, 0.55] after random swap |
| H7 | Live continuity | Max 40% odds jump per update |
| H8 | SGP no-arb | SGP price never below max correlated leg |
| H9 | Outright sum | Tournament winner probabilities sum to ±0.5% |
| H10 | Minimum odds | Never below 1.01 |
| H11 | Settlement accuracy | 100% correct on 50 golden test matches |

---

## QUICK REFERENCE — WHAT TO DO BEFORE EVERY CODE CHANGE

```
1. Read CLAUDE.md (this file)
2. Glob — find ALL files that will be affected
3. Read — read every affected file
4. Identify dependents — who imports this?
5. Plan surgical minimum change
6. Implement with full type hints, structlog, domain exceptions
7. Run tests: pytest tests/ -x --tb=short
8. Run regression lock: python scripts/lock_regression_state.py verify
9. Commit: type(scope): description
```
