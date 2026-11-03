# Pre-race Strategy Forecast — First Release Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This is a proposed plan, not authorization to implement or deploy.

**Goal:** Produce a useful, explainable pre-race tyre-strategy forecast from the completed practice and qualifying sessions of a selected weekend, informed by earlier races, and publish it inside the existing dashboard.

**Architecture:** Vercel Hobby serves the existing Next.js frontend and Render Free serves a FastAPI backend for catalog, reports, race-data chunks and bounded scenario calculations. Offline preparation handles full telemetry downloads, feature extraction and training. Immutable prepared datasets and model artifacts survive through the deployment bundle; ephemeral runtime caches accelerate repeat requests. This architecture follows the user's explicit choice of a Render backend and an entirely free initial deployment.

**Tech Stack:** Existing Python, pandas, NumPy, FastF1, FastAPI and Next.js/TypeScript. Add only pytest and a reproducible Python dependency lock for the first release; use a regularized linear model implemented with NumPy as the initial forecasting baseline. Evaluate a boosted-tree challenger in a later milestone.

**Spec:** The proposed product specification and acceptance criteria below, based on the user's revised requirements of 2026-11-03. The user prioritizes pre-race predictions using practice and qualifying, wants incremental scope, and wants both ML/AI and software-engineering portfolio value.

## Proposed product specification

### Product workflow

1. Select a prepared race weekend and driver.
2. Inspect which practice sessions and qualifying data were available before the forecast cutoff.
3. Inspect estimated race pace, compound degradation trends and projected stint pace with uncertainty.
4. Compare up to three distinct tyre strategies: sequence, planned pit laps, plausible pit windows, predicted elapsed-time differences and sensitivity to pace/degradation/pit-loss assumptions.
5. Read evidence explaining the ranking and any missing information.
6. Select among prepared low/base/high degradation and pit-loss scenarios, with a bounded custom scenario endpoint once its free-instance benchmark passes. All visitors can inspect the same reports; accounts are unnecessary.

The first report is explicitly conditional on a dry, uninterrupted race and the stated tyre-set assumptions. Its ranking minimizes predicted elapsed time under a free-air model. It is a useful starting strategy, but cannot yet optimize for traffic, overtaking or finishing position. The interface must state this next to the recommendation.

### Scope of the first release

- Conventional race weekends, full race distance, dry conditions, observed event entry list.
- One- and two-stop strategies under an explicit event policy and tyre-inventory scenario.
- Historical circuit pit-loss and pace/degradation priors trained only on earlier events.
- Practice long-run evidence and qualifying-relative performance, with uncertainty and fallback behaviour.
- Projected next-lap/stint pace under a specified strategy, not a claim to know race-day telemetry.
- Prepared public reports for a small curated collection of weekends, plus a CLI workflow the owner can run after qualifying for a new weekend.
- Target three complete dry weekends across at least two circuits for the product pilot; this is a workflow gate, not enough data to establish general forecasting skill. Expand the evaluation set before making accuracy claims.

### Subsequent milestones

**Release 2 — finishing distributions:** add a fast lap-level field simulation, grid input, approximate traffic/overtaking, opponent strategy policies and paired Monte Carlo scenarios. Initially label distributions conditional on all cars finishing and green-flag running; introduce historical incident/neutralization models only after evaluation. Report forecast calibration. Integrate additional MAS team coordination only when it improves the comparison.

**Release 3 — continuous replay analysis and model improvement:** feed completed laps into a sequential observer; update driver pace/degradation beliefs, next-lap forecasts, short-horizon pace and strategy assessments. Generate an insight only on a meaningful change and show its evidence. Seeking backwards restores the correct earlier state. After a completed race passes data-quality checks, record forecast errors and include its observations once in the next offline training cycle. Promote a new global model only after chronological validation. Historical replay works offline; actual live ingestion is outside this plan.

These releases use the same pace-model and report schemas, but are separate implementation plans. The existing replay remains part of the local application; the public pilot can expose selected prepared replays when their payload delivery is ready.

### Historical relevance policy

The default current-car forecast is for the 2026 regulation era. F1 documents major changes in chassis, power units, tyres and active aerodynamics for 2026. Partition the data into 2022–2025 and 2026-onward eras; earlier eras stay available for historical viewing but are excluded from the first forecasting pilot. Do not assume every season within an era is identical.

- Current forecasts use only completed 2026 events before their cutoff, plus the target weekend's available practice/qualifying. Historical 2022–2025 reports use their own earlier same-era training pool and separate model versions.
- At the beginning of an era, report sparse support and broad uncertainty; do not silently fill it with old-car absolute pace or degradation. Unsupported combinations may produce a scenario analysis instead of a confident ranked forecast.
- Cross-era transfer is a later experiment. Compare a 2026-only baseline against carefully normalized historical features using held-out 2026 events; enable transfer only if it improves those results. An `era` feature alone is not evidence that mixing seasons is safe.
- Record circuit layout version, resurfacing/change notes when verified, event date, season, tyre construction/compound nomination when available, session type, weather and relevant race-policy version. Missing metadata remains unknown. The labels Soft/Medium/Hard do not identify the same physical compound across all events.
- Older track geometry and pit-lane information may be reused only after compatibility checks. Re-estimate pit-time loss where pit layout/speed rules or racing speeds changed; do not transfer a historical seconds figure unchanged merely because the circuit name matches.
- Use within-era recency weighting as a validated model choice, including evolving team performance and driver/team changes. Hold out complete later weekends; report results by era, season and circuit familiarity.
- Keep 2026 telemetry interpretation separate from older DRS interpretation. Do not infer active-aero state, battery state of charge or deployment from unavailable channels. Missing data must not be encoded as evidence that a system was inactive.

### Learning from completed races

There are two learning timescales. The replay observer updates a temporary driver/stint state as new laps arrive. A trained global model is updated offline after newly completed races are validated. Playback speed or the number of people watching must not change the global model.

1. Save each pre-race and sequential forecast with event/driver/lap, input cutoff, model version, scenario and uncertainty. Evaluate it only when its actual target becomes available.
2. Deduplicate observations by event/session/driver/lap and source revision. Replaying the same race ten times adds no new training examples. Corrections replace a dataset revision and trigger reevaluation.
3. Record real observed errors by compound, tyre age, circuit, traffic/flags and model support. Learn from observed lap/sector outcomes; simulated alternate-strategy gains are not ground-truth training labels.
4. Retrain a candidate model on eligible same-era history, including the new race. For a boosted-tree model, begin with reproducible batch retraining; continuing training on only the newest race is not the default because it can overfit or forget broader conditions.
5. Compare the candidate with the deployed model on a fixed recent validation window and rolling chronological folds. Keep a final untouched audit set outside model selection, and use prospective future races for further evaluation. Report clean-lap MAE, short-stint error, interval coverage/width, sparse-data slices and runtime. Finishing probability calibration is added with Release 2.
6. Promote only a candidate with better primary validation error, acceptable uncertainty calibration and no material subgroup regression under a predeclared protocol. Keep the old model if evidence is inconclusive. Save training cutoff, dataset hash, parameters and rollback artifact; preserve earlier forecast reports unchanged.

Release 1 stores the provenance and residual-evaluation data this loop needs. A future complex-model release adds `race_strategy/learning.py`, `tools/retrain_pace.py` and `tests/test_learning.py`, with contracts `build_training_revision(observations, forecast_records) -> DatasetRevision` and `compare_candidate(candidate, deployed, evaluation_set) -> PromotionReport`. Tests must cover duplicate replay, future labels, source corrections, rejected candidates and rollback. Training runs on the owner's machine, never inside Render Free request handling.

### Insights for viewers and strategy analysis

Use the patterns of established timing tools, including MultiViewer's race traces, lap-time series and race-control timeline. Implement them from available FastF1 data; they do not require MultiViewer or its subscription.

| Insight | What it helps someone understand | Delivery |
|---|---|---|
| Practice long-run comparison | Pace distributions at comparable tyre ages, usable lap counts and uncertainty about run programmes | Release 1 |
| Compound/stint chart | Which tyres were used, age, fresh/used status and changes between sets | Release 1 |
| Strategy sensitivity chart | Which pit laps remain competitive when degradation or pit loss changes | Release 1 |
| Evidence and confidence panel | How much of a forecast comes from this weekend versus historical priors | Release 1 |
| Race trace and gap trend | Who gains time against a selected driver or leader; annotate pit stops and neutralizations | Small replay follow-up after Release 1 |
| Rolling pace and consistency | Recent clean-lap median/spread, teammate comparisons, anomalies | Same replay follow-up |
| Race-control timeline | Flags, safety car/VSC, weather observations and recorded messages available by replay time | Same replay follow-up |
| Corner and sector contribution | Where a measured time gap accumulates; avoid claiming a causal driving mistake | Reuse existing comparison views, extend later |
| Pit rejoin / undercut window | Approximate position after a stop and risk of traffic; propagate pit-loss uncertainty | Release 2 |
| Team double-stack risk | Whether both cars would contend for the same pit box | Release 2 |
| Finish-position and points distributions | Range of outcomes under explicit opponent and incident assumptions | Release 2 |
| Forecast-versus-observed timeline | Whether pace/degradation changed and what observations changed the recommendation | Release 3 |

Every insight carries its timestamp, observation window, quality flag and whether it is measured, inferred or simulated. Replay cards use only information at the cursor. Publish insights on completed laps/significant changes and avoid repeated alerts. For a public demo, precompute sequential replay records with the same cutoff discipline and reveal them at their timestamps, keeping expensive computations independent of visitor playback. These are strategy-analysis aids based on public data, with no claim to private team telemetry.

## Global constraints

- Prediction features must use only information available before the declared cutoff; the target race is available only to a separate evaluator after forecasts are frozen.
- Historical backtests using data downloaded today are reconstructed forecasts, not proof of what was published before that race. Prospectively generated reports retain their original timestamps and artifacts.
- Historical weather observations are not race-weather forecasts. Start with a user-visible dry scenario; any future weather provider is a separate integration.
- Qualifying position is not necessarily the final starting grid. Keep a provisional grid and an owner-editable grid override with provenance for Release 2; do not obtain the grid from target-race results during backtests.
- Do not interpret fuel load, tyre temperature or physical wear percentage as measured features: they are absent from the available input tables.
- A driver/compound with little practice evidence must retain a historical prior and wider uncertainty, or be marked unsupported. Do not invent a weekend-specific measurement.
- Preserve the main app's existing routes. Add a small independent package; do not restructure the whole backend.
- Treat the nested MAS as an independent repository. Leave its source and history untouched in this release; port the necessary search abstraction into the main repository with provenance, checking applicable licensing before redistribution.
- Public visitors use Render endpoints for supported datasets and bounded scenarios. They cannot trigger training, full raw telemetry ingestion or unlimited simulations. Lightweight uncached lap loading is enabled only after profiling and source-rate-limit checks, with explicit availability rather than a promise that every historical event is supported.
- Data availability and forecast quality are separate statuses. A successfully downloaded session does not guarantee a credible forecast.
- The numbers below are proposed product/test settings, not measured performance or user-approved accuracy guarantees.

## Evidence from the current repository

- `api.py` loads whole sessions and includes final results; it also disables weather loading. Do not use its UI response as training input.
- Practice cards currently infer stints from compound changes. Model preparation must use FastF1's `Stint` and tyre-age information because consecutive sets can share a compound.
- Existing UI telemetry is downsampled and missing values are filled with zero. Feature preparation needs independent quality handling.
- `F1-Strategy-MAS/tools/fit_model.py` fits a single completed race and scales it to 20 laps. Its fitted parameters cannot be used as a pre-race forecast for that same race.
- `F1-Strategy-MAS/sim/search.py` contains reusable planning ideas; its A* lower bound depends on the current lap-cost formula.
- The local main cache contains approximately 2,572 MiB, 13 race timing caches and only four practice timing caches. Complete weekend coverage needs acquisition and validation.
- Previous checks: 32 MAS tests passed with one WebSocket test excluded after it stalled during Windows socket initialization; MAS frontend lint passed; main frontend lint reported 12 errors and 13 warnings. These are observations from the review, not release certification.

## Approach and deployment decision

**Selected approach: Vercel Hobby frontend + one Render Free FastAPI service.** This supersedes the previous static-only deployment recommendation. Render handles catalog/report reads, compact race data and bounded custom strategy requests. Heavy data acquisition and model training run offline. Bundle a small curated set of immutable normalized datasets and the model with the deployment; runtime caches are disposable. A static sample report in the frontend provides a useful view while Render wakes.

**Free-tier request design:** serve session metadata/laps first; fetch track positions in time chunks and detailed telemetry for only the selected drivers/laps. Replay animation runs in the browser. Use one Python server worker, a bounded memory cache, duplicate-request coalescing and at most one expensive calculation at a time. A custom calculation has a deterministic key including data version, model version, driver and scenario; return a cached result when available. Reject excess work with retry guidance rather than building an unbounded queue. Do not import the global MAS Engine or carry its one shared mutable race into the multi-user backend.

Proposed limits to validate: custom scenarios use at most three distinct strategy choices, one active compute job, a 20-second compute deadline, 10-minute ephemeral result retention and 10-second client polling. Return a job ID with HTTP 202; report completed/failed/expired explicitly. After a process restart, a missing job expires cleanly and a retry reuses the deterministic request key. Implement CPU work with bounded cancellation rather than an asyncio task that blocks the web server. Start with prepared scenarios if the job machinery/model cannot meet the memory budget; the Render API still serves all supported reports and race assets.

Retain the existing local full-session endpoint for development. In public mode, guard it so visitors cannot bypass these limits through `/api/session` or `/api/time-delta`. Public replay uses prepared artifacts; full raw FastF1 loads stay outside the free instance. An uncached lightweight lap-only endpoint is a later benchmark-gated capability, not an advertised initial guarantee.

**Loading-time contract:** the user's desired target is under 60 seconds. On an awake instance, target a first useful prepared view within 10 seconds and the selected initial replay chunk within 30 seconds on the declared test connection. These are unmeasured targets. A sleeping Render service can take about a minute to wake before application work begins, so end-to-end loading can exceed 60 seconds even for cached data. A first-ever raw race download/parse may take minutes or exceed memory; no sub-minute claim is made for it. Show actual stages: backend waking, loading timing, loading replay, preparing scenario. Do not display fabricated percentage progress.

**Storage and free-only scope:** ship only curated prepared files in the deployment bundle, cap runtime caches, and restore immutable assets from the bundle after restart. No paid disk, background worker, cron service, Redis or database is required. Do not use expiring free Postgres as permanent model storage. Larger catalogs can later move to object storage after verifying a zero-cost quota and overage behaviour; no storage service with automatic paid overages is a default dependency. Monitor transfer/build budgets and disable expensive endpoints when an application budget is reached. App-side limits reduce consumption but do not guarantee provider billing caps; configure provider spending controls wherever available.

The public service is available on demand, subject to free-tier sleep, outages and quotas; it cannot be described as guaranteed always-on. Do not use artificial keep-awake traffic. Keep the fallback sample and a clear service-status view. Paid upgrades are outside the selected scope.

Current provider observations checked 2026-11-03:
- Vercel Hobby is for personal, non-commercial use; publishing this portfolio is the assumed use. Its Functions have request/response-size and execution limits, so full telemetry and ingestion do not belong in request handlers.
- Render Free supplies 0.1 CPU and 512 MB RAM. It sleeps after 15 minutes idle; waking takes about a minute. Filesystem changes are lost on redeploy/restart/spin-down.
- Render grants 750 free instance-hours per workspace/month, shared across free services. Bandwidth/build quotas also apply; the free documentation describes suspension/overage behaviour. Recheck account allowances and billing controls at deployment rather than assuming unlimited free transfer.
- Render documents supplemental bandwidth charges when a payment method is present; without one it suspends free services after the included bandwidth is exhausted. Build usage has its own spending-limit behaviour. For the user's strict zero-cost preference, verify the account's payment/overage settings and choose suspension over paid continuation. Do not claim an application request limiter guarantees a zero bill.
- Render may restart free services; paid worker/cron/disk components are excluded. Explicitly set the Blueprint service plan to `free` because omission can select a paid default.
- Use the supplied free hosting domains. Domain purchases and paid platform upgrades are outside this plan.

## File map and shared contracts

Proposed new package `race_strategy/`:
- `schemas.py`: dataclasses for identifiers, manifests, rows, priors, forecasts, event policies and reports; JSON encoding/decoding.
- `weekend.py`: FastF1 acquisition and normalization into lap tables, with weather timestamps and a manifest.
- `features.py`: clean stint extraction and weekend summaries.
- `pace.py`: regularized baseline, prior blending and parameter uncertainty.
- `planner.py`: bounded strategy enumeration and evaluation, with event constraints.
- `report.py`: immutable JSON report assembly, explanations and scenario comparisons.
- `routes.py`: catalog/report/race-chunk routes plus bounded scenario submission/status.
- `jobs.py`: deterministic request keys, one-computation admission limit, cancellation/deadlines and ephemeral result cache.
- `regimes.py`: era membership and feature-specific historical eligibility.
- `insights.py`: evidence-linked practice comparisons and strategy sensitivity summaries.
- `evaluate.py`: chronological model evaluation, isolated from forecast generation.

Entry points: `tools/prepare_weekend.py`, `tools/build_prerace_report.py`, `tools/evaluate_prerace.py`.

Frontend: add `frontend/src/lib/strategy.ts`, `frontend/src/components/PreRaceStrategy.tsx` and integrate into `frontend/src/app/page.tsx`. Modify `frontend/src/lib/api.ts` to support a configured public API URL and prepared-artifact mode.

Key contracts (define all types in `schemas.py`):
- `WeekendKey(year: int, round: int)`; use stable event identifiers, not display names, as storage keys.
- `ForecastManifest`: key, UTC cutoff, generation timestamp, regulation era, layout/tyre metadata, source sessions and availability timestamps, input hashes, training-event IDs, model version, assumptions and reconstructed/prospective label.
- `WeekendBundle`: manifest, normalized practice/qualifying lap tables, observed weather, driver metadata and provisional qualifying order. No target-race laps/results.
- `StintSummary`: driver, session, stint, compound, usable sample count, tyre-age range, lap-time trend and quality flags.
- `PacePrior`: training cutoff, driver/team/circuit/compound effects, support counts, parameter draws and training metadata.
- `WeekendPaceModel`: prior updated with weekend evidence; `predict_lap(driver: str, compound: str, tyre_age: int, race_lap: int, draw: int) -> float` returns seconds.
- `EventPolicy`: race distance, allowed compounds, allowed start compounds, maximum stops, optional tyre-set counts, required compound use and source/override metadata.
- `StrategyPlan`: starting compound and ordered `(pit_lap, new_compound)` stops.
- `StrategyForecast`: plan, elapsed-time quantiles, time relative to the reference plan on matched draws, scenario-specific rank and pit-window sensitivity.
- `PreRaceReport`: manifest, coverage, pace summaries, assumptions, per-driver strategies and evidence messages.
- `ScenarioRequest`: weekend key, dataset/model versions, driver, up to three feasible StrategyPlans and bounded degradation/pit-loss scenario identifiers; no arbitrary code or training configuration.
- `ScenarioJob`: ID, deterministic request key, state (`queued`, `running`, `completed`, `failed`, `expired`), creation/expiry timestamps and result/error. In the first implementation, admit one running job and no additional unique queued work; coalesce duplicate requests.

## Review Focus

1. Target-race results accidentally enter priors or features: reject them at bundle construction and test output invariance when target-race labels change.
2. Consecutive stints use the same compound: separate them by session/driver/Stint, not compound transitions.
3. Practice is missing, wet or too sparse: disclose fallback evidence; never silently produce a narrow confident forecast.
4. A proposed strategy assumes unavailable tyres or a special event rule: validate against explicit policy and label inventory assumptions.
5. Public hosting, job execution or report publication fails: retain the last complete artifact version, expose restart/expiry and provide the sample fallback without initiating unbounded background downloads.

## Task 1 — Prepare and audit a complete weekend

**Files:** create `race_strategy/__init__.py`, `race_strategy/schemas.py`, `race_strategy/weekend.py`, `race_strategy/regimes.py`, `tools/prepare_weekend.py`, `tests/test_weekend.py`; update backend dependency setup as needed for tests and reproducibility.

**Interfaces:** `prepare_weekend(key: WeekendKey, cutoff_utc: str, cache_dir: Path) -> WeekendBundle`; `save_bundle(bundle: WeekendBundle, output_dir: Path) -> Path`; `prepare_training_table(training_keys: list[WeekendKey], cutoff_utc: str, cache_dir: Path) -> pd.DataFrame` loads only earlier completed events, joining their pre-race practice/qualifying features to their race-lap labels. This separate training path must reject the forecast target event and any event ending after the cutoff.

- [ ] Write tests that accept completed practice/qualifying before cutoff; reject target-race input; retain missing values; retain source hashes; mark missing sessions explicitly.
- [ ] Define `eligible_history(target: WeekendKey, candidates: list[ForecastManifest], purpose: str) -> list[ForecastManifest]`; test 2025 pace data is excluded for a 2026 target, future 2026 events are excluded, and older geometry requires matching verified layout metadata. Keep historical viewing eligibility separate from training eligibility.
- [ ] Run `python -m pytest tests/test_weekend.py -q`; confirm the new behaviour fails before implementation.
- [ ] Implement acquisition using lap/weather data without full telemetry, and explicit cache reuse. Preserve raw `Stint`, `TyreLife`, `FreshTyre`, `IsAccurate`, deleted-lap indicators where available, pit markers and flags. Persist normalized CSV plus JSON manifest initially. Collect historical training weekends through the separate training interface, keeping race labels out of the target WeekendBundle.
- [ ] Run the test file; expect all assertions to pass without network access using fixtures.
- [ ] Prepare one complete historical dry 2026 weekend with earlier 2026 training events. Record missingness and data provenance; choose subsequent pilot weekends from available coverage, not from model success. If support is inadequate, show the sparse-evidence scenario result. A 2022–2025 development example is labelled a separate-era demo, never evidence of 2026 forecast skill.
- [ ] Commit this independently reviewable unit when implementation is authorized.

## Task 2 — Build the baseline pace forecast

**Files:** create `race_strategy/features.py`, `race_strategy/pace.py`, `tests/test_pace.py`.

**Interfaces:** `summarize_stints(bundle: WeekendBundle) -> list[StintSummary]`; `fit_prior(training_laps: pd.DataFrame, cutoff_utc: str) -> PacePrior`; `forecast_weekend(bundle: WeekendBundle, prior: PacePrior, seed: int) -> WeekendPaceModel`.

- [ ] Write tests for two same-compound stints, excluded pit/flag/deleted laps, unsupported compounds, a synthetic known degradation trend, and future-event exclusion.
- [ ] Run `python -m pytest tests/test_pace.py -q`; confirm failures.
- [ ] Start with runs of at least five usable timed laps as a configurable evidence threshold; exclude contaminated observations rather than joining separate stints. Report counts and age coverage. Mark traffic-contaminated evidence when detectable; flags alone do not establish clean air.
- [ ] Fit regularized pace/tyre-age/race-progression effects on earlier race data. Treat the race-progression term as an empirical effect, not a measured fuel quantity. Blend practice stint trends toward historical priors using evidence strength; use qualifying-relative pace as a separate feature whose weight is learned on earlier weekends, never as a direct race-lap forecast.
- [ ] Estimate uncertainty using grouped historical residuals and weekend/stint resampling. Draw driver/event pace offsets consistently across projected laps, with separate lap noise, so uncertainty does not disappear unrealistically over a stint. Retain broad fallback ranges for sparse evidence.
- [ ] Run unit checks; inspect projections on the pilot weekend. Emit median and 10th/90th percentile estimates with coverage still described as unvalidated until Task 4.
- [ ] Commit.

## Task 3 — Generate useful pre-race strategy comparisons

**Files:** create `race_strategy/planner.py`, `race_strategy/report.py`, `tools/build_prerace_report.py`, `tests/test_planner.py`, `tests/test_report.py`.

**Interfaces:** `enumerate_plans(policy: EventPolicy) -> list[StrategyPlan]`; `rank_strategies(model: WeekendPaceModel, driver: str, policy: EventPolicy, pit_loss_draws: np.ndarray, seed: int) -> list[StrategyForecast]`; `build_report(bundle: WeekendBundle, model: WeekendPaceModel, policy: EventPolicy, seed: int) -> PreRaceReport`.

- [ ] Write small exhaustive examples with known cheapest strategies, tyre inventory constraints, invalid pit laps, full race distance, deterministic seeds and insufficient evidence. Test paired scenario comparisons use the same parameter draws.
- [ ] Run `python -m pytest tests/test_planner.py tests/test_report.py -q`; confirm failures.
- [ ] Enumerate feasible one-/two-stop plans on lap boundaries and evaluate them with precomputed cumulative stint costs. Reuse MAS planning concepts, but use exhaustive bounded enumeration as the transparent reference before adapting A*. Restrict the initial supported event policies rather than claiming universal sporting-rule coverage.
- [ ] Rank the base scenario by mean modeled elapsed time; return median and 10th/90th percentiles and sensitivity across low/base/high parameter scenarios. Keep up to three distinct compound/stop-count choices instead of three nearly identical neighbouring pit laps. Define a pit window as feasible shifts within one modeled second of that sequence's best base-scenario plan, explicitly a sensitivity window rather than a calibrated confidence interval.
- [ ] Use historical circuit pit losses with uncertainty, not the MAS's fixed Silverstone value. Describe assumed tyre inventory and avoid strategy recommendations that contradict known inventory.
- [ ] Assemble evidence-based text, manifest and model version; write artifacts atomically. Do not emit position probabilities from a free-air elapsed-time model.
- [ ] Run checks and generate one pilot report; commit.

## Task 4 — Evaluate forecasts before advertising accuracy

**Files:** create `race_strategy/evaluate.py`, `tools/evaluate_prerace.py`, `tests/test_evaluation.py`, `docs/prerace-model-card.md`.

**Interfaces:** `evaluate_reports(reports: list[PreRaceReport], heldout_laps: pd.DataFrame) -> dict`.

- [ ] Test chronological event splits, exclusion of target-race labels during generation, and invariance of a frozen forecast when future labels are changed. Test missing unsupported compounds are reported rather than dropped from coverage statistics.
- [ ] Run `python -m pytest tests/test_evaluation.py -q`; confirm failures.
- [ ] Evaluate observed clean race laps conditional on the actual compound/age, and short contiguous observed stints. Compare against a historical-only prior and a simple practice-based baseline; report MAE, interval coverage/width and support by event/compound. Use whole-event folds and report season transfer separately. Tune only on training/validation events.
- [ ] Compare forecast strategy with recorded strategy descriptively. Any claimed alternate strategy gain remains model-based, not observed causal improvement.
- [ ] Freeze the pilot evaluation protocol before seeing outcomes. The three-weekend workflow pilot is not a statistical accuracy claim; expand to a broader held-out set for portfolio metrics. Retain the simpler model if the challenger does not improve held-out performance. Store deduplicated actual-versus-predicted records for later retraining; test a repeated replay does not change the dataset count or the deployed model.
- [ ] Run checks; save the model card with provenance, feature availability, limitations, sample counts and measured metrics; commit.

## Task 5 — Integrate the report into the existing application

**Files:** create `race_strategy/routes.py`, `race_strategy/jobs.py`, `race_strategy/insights.py`, `frontend/src/lib/strategy.ts`, `frontend/src/components/PreRaceStrategy.tsx`, `tests/test_strategy_api.py`, `tests/test_strategy_jobs.py`; modify `api.py`, `frontend/src/app/page.tsx`, `frontend/src/lib/api.ts`.

**Interfaces:** `GET /api/strategy/weekends` returns a prepared report catalog; `GET /api/strategy/prerace/{year}/{round}` returns `PreRaceReport` or a clear not-prepared response. Both read artifacts only. `GET /api/replay/{year}/{round}/manifest` returns chunk descriptors; `GET /api/replay/{year}/{round}/chunks/{chunk_id}` returns an allowlisted prepared chunk. `POST /api/strategy/scenarios` accepts ScenarioRequest and returns an existing completed result or HTTP 202 with ScenarioJob; `GET /api/strategy/jobs/{job_id}` returns job state. Excess unique work returns 429 with Retry-After. Frontend `loadPreRaceReport(year: number, round: number): Promise<PreRaceReport>` uses the configured Render URL; static sample fallback is explicitly labelled.

- [ ] Test catalog/report schemas, nonexistent weekends, corrupt reports and absence of provider calls during reads.
- [ ] Test public-mode guards for the old full-session endpoints; validate driver/event/chunk IDs, compute limits, duplicate-request coalescing, busy response, cancellation timeout and expired jobs after restart. A second visitor must never control or mutate another visitor's scenario. Add configurable read/compute rate limits and test that limits return a clear retry response without allocating work; choose initial thresholds from the free-instance benchmark.
- [ ] Run `python -m pytest tests/test_strategy_api.py -q`; confirm failures.
- [ ] Add the new route module and frontend entry without replacing the existing analysis views. Surface session coverage, cutoff, assumptions, driver selector, projected stint curves, three strategies and evidence. Clearly distinguish historical reconstruction from a saved prospective forecast.
- [ ] Keep prepared scenario selection immediate, and add bounded custom requests only if the free-instance benchmark passes. Implement the practice long-run comparison, compound/stint chart, strategy sensitivity and evidence panel from the insight table. Test sample counts, same-compound set separation, insufficient evidence and uncertainty labels. Keep actual values distinct from model estimates.
- [ ] Run API checks, `npm run lint` and `npm run build` in `frontend/`. Resolve existing lint blockers as a bounded prerequisite, following the installed Next.js guidance in `frontend/AGENTS.md` before edits.
- [ ] Visually verify a complete weekend, sparse practice, missing report, narrow viewport and keyboard selection; commit.

## Task 6 — Publish a small, usable pilot

**Files:** create `tools/publish_reports.py`, `tools/benchmark_serving.py`, `tests/test_publication.py`, `.github/workflows/checks.yml`, `render.yaml`, `docs/deployment.md`; modify `README.md`, `frontend/next.config.ts` only if required for the chosen hosting mode.

**Interfaces:** `publish_reports(source_dir: Path, public_dir: Path) -> Path` validates complete reports, copies selected immutable artifacts and swaps the catalog last.

- [ ] Test that incomplete reports are not published, source caches are excluded and an interrupted update preserves the previous complete catalog.
- [ ] Run `python -m pytest tests/test_publication.py -q`; confirm failures, implement publication, then expect pass.
- [ ] Add CI for Python tests, frontend lint/build and schema validation with frozen fixtures. No live FastF1 downloads in routine CI.
- [ ] Benchmark separately: offline ingestion, warm report reads, chunked replay, bounded scenario requests, fresh uncached lap data and Render cold start. Record p50/p95, peak resident memory and transferred bytes for one visitor and five concurrent readers. Use a proposed 384 MiB peak-process budget to leave headroom within 512 MB; reject serving paths that exceed it. A local limit test cannot establish Render latency: repeat on the actual free instance when deployment is authorized.
- [ ] Proposed artifact targets: report at most 1 MiB compressed; replay chunk at most 2 MiB compressed; first useful prepared view within 10 seconds and first replay chunk within 30 seconds on an awake service. Report cold-start results separately; explicitly disclose that 60 seconds end-to-end is not guaranteed. If custom compute exceeds its budget, retain API-served prepared choices. Avoid claims about maximum user capacity until concurrency is measured.
- [ ] Configure exactly one free Render service with one server worker and explicit `plan: free`, no paid backing services, allowlisted Vercel CORS origin, bounded caches and public-mode guards. Bundle a small validated model and prepared catalog. Keep a static frontend sample available during backend waking, quota exhaustion or failure. Do not deploy the full raw cache or expose training endpoints.
- [ ] Prepare at least three curated reports and one documented race-weekend update procedure: acquire sessions, validate cutoff and coverage, generate report, inspect evidence, publish, save version before race start.
- [ ] Verify applicable provider/data redistribution terms for the specific published artifacts before public release. Keep this a focused release check; the software library's license alone does not grant rights to all upstream data.
- [ ] Document environment URLs, CORS, cold-start UI, runtime restart recovery, payload sizes, quota monitoring, transfer budgets, rollback and zero-paid-service configuration. Recheck free transfer/build allowances and billing controls for both accounts before deployment; stop/restrict service before an app budget is exceeded. Confirm a production preview works without localhost references or private local files. Do not promise unrestricted traffic at zero cost.
- [ ] Commit the deployment-ready result; deploy only when explicitly requested.

## Definition of done for the first release

- A visitor can select a supported weekend and driver, inspect the pre-race evidence, compare feasible strategies and understand the assumptions.
- The owner can prepare a new report after practice/qualifying availability without editing source code.
- Forecast generation never reads target-race observations; reports retain cutoff and model provenance.
- Sparse/missing practice and unsupported event types have explicit behaviour.
- The baseline is reproducible and evaluated against earlier-data alternatives; weak results are disclosed rather than hidden.
- The Render public report path uses immutable artifacts, needs no live-data subscription, and respects the free-tier memory/compute/request budgets. Custom scenarios are bounded and benchmark-gated; visitors can use the supported catalog even if custom compute is disabled.
- Current-car training is same-era by default; historical comparability, sparse support and model provenance are visible. Replaying a race does not mutate the global model or duplicate training labels.
- Checks and the production build pass; existing replay behaviour has been smoke-tested.

## Sources checked for feasibility

- FastF1 data availability and optional live recorder: https://github.com/theOehrly/Fast-F1/blob/main/docs/data_reference/index.rst
- Vercel Hobby restrictions: https://vercel.com/docs/plans/hobby
- Vercel function limits: https://vercel.com/docs/functions/limitations
- Render Free limitations: https://render.com/docs/free
- Render published compute comparison: https://render.com/articles/render-vs-railway
- Hugging Face current compute Space requirements: https://huggingface.co/docs/hub/spaces-overview
- Optional later object storage pricing: https://developers.cloudflare.com/r2/pricing/
- Render compute allocation: https://render.com/docs/compute-plans
- Render Blueprint explicit free plan: https://render.com/docs/blueprint-spec
- F1 2026 chassis and power-unit changes: https://corp.formula1.com/f1-2026-regulations-terminology-update/
- F1 2026 aero explanation: https://www.formula1.com/en/latest/article/2026-regulations-explained-all-you-need-to-know-about-f1s-new-aerodynamics.7IAt0auc32UkCEFE5ypkTB
- MultiViewer race-trace design reference: https://multiviewer.app/docs/usage/race-trace
- MultiViewer timing and insight features: https://multiviewer.app/
- CatBoost training/continuation API (later challenger): https://catboost.ai/docs/en/concepts/python-reference_train

## Self-review

Updated for the user's regulation-relevance, learning-loop, insight and free Render-backend requirements. The first-release tasks cover same-era evidence, pre-race forecasting, four focused insight groups, a multi-user API and bounded free-tier serving. Finishing distributions and sequential replay/model-retraining work remain subsequent releases; their data contracts and promotion principles are specified now. All five Review Focus cases have owning tests. Targets have not been benchmarked on Render, and cold-start sub-minute loading is explicitly not guaranteed. No source implementation, model training, account creation or public deployment was performed while revising this plan.
