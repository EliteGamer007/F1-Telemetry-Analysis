# Free pilot operations

Render: Python 3.12, requirements-serve.txt, `uvicorn race_strategy.service:app --host 0.0.0.0 --port $PORT --workers 1 --limit-concurrency 25`, Free plan. Set CORS_ORIGINS to the exact Vercel production URL. No database, disk, worker or paid cron. Public reads only prepared artifacts; custom compute returns 409. Nine scenario choices are prepared offline. Per-peer reads are bounded to 120/minute, global reads 1200/minute. Free idle spin-down can make cold starts exceed one minute; warm latency must be measured after deployment.

Vercel Hobby: project root frontend; NEXT_PUBLIC_API_BASE_URL is the Render HTTPS URL; NEXT_PUBLIC_PREPARED_ONLY=true. Saved sample reports remain usable during backend unavailability. Hobby is for personal noncommercial use. Monitor both dashboards' transfer/build allowances; no traffic capacity promise has been established.

To update a weekend: `python -m tools.prepare_weekend --year 2026 --round 4 --training 1,2 --cache-only`, then `python -m tools.build_prerace_report data/prepared/2026-4 --laps 57 --pit-loss 22`, inspect coverage and assumptions, then `python -m tools.publish_reports data/reports artifacts`. Copy the published reports/catalog to frontend/public/samples. Commit immutable reports before prospective race forecasts. With uncached sessions, current upstream access requires the provider's supported authentication; never put credentials in git or the public API.

Optional selected replay: `python -m tools.prepare_replay --year 2026 --round 4`. Raw cache is excluded. Revert the publishing commit to roll back. Republishing preserves old content-addressed reports. Check upstream data terms for your intended redistribution; the repository MIT license covers code, not ownership of upstream timing data.

Basic checks: `python -m pytest tests -q`, `npm --prefix frontend run lint`, `npm --prefix frontend run build`. Routine CI uses synthetic fixtures; it does not download timing data.

## Verified production pilot — 4 November 2026

Frontend: https://f1-telemetry-sooty.vercel.app
Backend: https://f1-telemetry-api-4k13.onrender.com
Render service: srv-dbl2gt67bikc7387aq80, Singapore, Free (0.1 CPU / 512 MB), one worker. Vercel production variables are persisted and Git root directory is frontend.

25 automated tests pass, including published report/fallback parity, all replay chunks and CORS allowlisting. Production smoke: all three reports 200, warm reads 0.627–1.637 seconds; five simultaneous reads 1.534–2.477 seconds; all eight replay chunks and 22 drivers reconstructed in 12.041 seconds with sequential downloads. Reports are 37–46 KB compressed. Invalid weekend returns 404 and disabled custom compute returns 409. Both sites are publicly accessible without login. These are single-run client observations, not capacity or cold-start guarantees; production peak memory has not been measured.

Repeat: `python -m tools.benchmark_serving --api https://f1-telemetry-api-4k13.onrender.com --frontend https://f1-telemetry-sooty.vercel.app`.
