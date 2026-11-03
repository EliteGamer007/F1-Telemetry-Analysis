# Pre-race baseline v1

Experimental full-distance free-air strategy comparison. Uses ridge regression on earlier same-era green accurate race laps, shrunk practice stint trends, and historical qualifying-relative effects. Reports never read the target race. Unknown deleted-lap status and traffic remain limitations of the cached observations.

The pilot contains Miami 2026, Imola 2025 and Abu Dhabi 2024. Each uses its own era; missing sessions are visible. Miami and Abu Dhabi lack prior practice/race calibration and are explicitly scenario-only. Pit losses and tyre inventory are owner scenario assumptions, not measured quantities. The three complete 2026-weekend data gate remains unmet because the upstream archive now requires authentication (FastF1 issue 978).

Frozen-report evaluation on the two available target race caches: 1,674 eligible laps, MAE 2.436 seconds, historical-effects baseline MAE 1.115 seconds, practice baseline MAE 1.622 seconds. Uncalibrated 10–90% ranges cover 71.1% with mean width 7.303 seconds. Five-lap windows: 240, mean per-lap absolute aggregate error 2.271 seconds. Imola held-out race is unavailable. This small pilot does not establish predictive skill; the fitted model underperforms the simpler baseline. Do not advertise an accuracy improvement or promote it as a validated production predictor. Keep prospective evaluation and a larger untouched audit set for future model selection.

Replay does not retrain a global model. The evaluator deduplicates event/session/driver/lap records, replacing source revisions. Counterfactual strategy gains are simulated, never training labels. Finishing distributions, traffic simulation and online replay learning are later releases.
