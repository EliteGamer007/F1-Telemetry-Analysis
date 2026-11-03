'use client';

import { useEffect, useState } from 'react';
import { ArrowDownUp, ArrowRight, ChartNoAxesCombined, ChevronDown, Flag, Info, Layers, RefreshCw, ShieldCheck, Timer } from 'lucide-react';
import { Compound, CurvePoint, loadPreRaceReport, loadStrategyCatalog, PreRaceReport, ScenarioLevel, StrategyPlan, Weekend } from '@/lib/strategy';

const COLORS: Record<Compound, string> = { SOFT: '#fb5a64', MEDIUM: '#f5c84c', HARD: '#e6e9ee' };
const LEVELS: ScenarioLevel[] = ['low', 'base', 'high'];
const PANEL = 'rounded-2xl border border-white/10 bg-white/[0.025]';
const seconds = (n: number) => `${Math.floor(n / 60)}:${(n % 60).toFixed(3).padStart(6, '0')}`;
const signed = (n: number) => `${n >= 0 ? '+' : '−'}${Math.abs(n).toFixed(1)}s`;

function Tyre({ compound, small = false }: { compound: Compound; small?: boolean }) {
  return <span className={`inline-flex items-center justify-center rounded-full border-[3px] font-black ${small ? 'h-6 w-6 text-[10px]' : 'h-9 w-9 text-sm'}`} style={{ color: COLORS[compound], borderColor: COLORS[compound] }} title={compound}>{compound[0]}</span>;
}
function Eyebrow({ children }: { children: React.ReactNode }) { return <p className="text-[10px] uppercase tracking-[.18em] font-bold text-white/45">{children}</p>; }

function PaceChart({ curves }: { curves: Record<Compound, CurvePoint[]> }) {
  const points = Object.values(curves).flat();
  if (!points.length) return <p className="p-10 text-sm text-white/50">Not enough practice evidence for a stint projection.</p>;
  const low = Math.floor(Math.min(...points.map(p => p.p10))), high = Math.ceil(Math.max(...points.map(p => p.p90)));
  const ageMax = Math.max(...points.map(p => p.age));
  const x = (age: number) => 54 + (age - 1) / Math.max(1, ageMax - 1) * 640;
  const y = (v: number) => 195 - (v - low) / Math.max(1, high - low) * 165;
  return <svg viewBox="0 0 720 230" role="img" aria-label="Projected lap pace by tyre age, with uncalibrated 10th to 90th percentile model ranges" className="w-full">
    {[0, 1, 2, 3].map(i => { const v = low + (high - low) * i / 3; return <g key={i}><line x1="54" x2="694" y1={y(v)} y2={y(v)} stroke="white" strokeOpacity=".07" /><text x="44" y={y(v) + 4} textAnchor="end" fill="#7c7c87" fontSize="10">{v.toFixed(1)}s</text></g>; })}
    {(Object.entries(curves) as [Compound, CurvePoint[]][]).map(([compound, rows]) => <g key={compound}>
      <polygon points={[...rows.map(p => `${x(p.age)},${y(p.p10)}`), ...[...rows].reverse().map(p => `${x(p.age)},${y(p.p90)}`)].join(' ')} fill={COLORS[compound]} opacity=".045" />
      <polyline points={rows.map(p => `${x(p.age)},${y(p.median)}`).join(' ')} fill="none" stroke={COLORS[compound]} strokeWidth="2" />
    </g>)}
    {[1, 10, 20, 30].filter(n => n <= ageMax).map(n => <text key={n} x={x(n)} y="215" fill="#7c7c87" textAnchor="middle" fontSize="10">{n}</text>)}
    <text x="375" y="229" fill="#7c7c87" fontSize="9" textAnchor="middle">TYRE AGE · LAPS</text>
  </svg>;
}

function PlanCard({ plan, rank, laps }: { plan: StrategyPlan; rank: number; laps: number }) {
  return <div className={`${PANEL} p-5 ${rank === 0 ? 'border-red-500/35 bg-red-500/[.04]' : ''}`}>
    <div className="flex justify-between items-start mb-5"><div><Eyebrow>{rank === 0 ? 'Lowest modeled time' : `Alternative ${rank}`}</Eyebrow><p className="text-sm font-semibold mt-1">{plan.pits.length} stop · {plan.compounds.map(c => c[0]).join(' → ')}</p></div><span className={`font-mono text-lg ${rank === 0 ? 'text-red-400' : 'text-white/80'}`}>{rank === 0 ? 'Best' : signed(plan.delta)}</span></div>
    <div className="flex items-center gap-3 mb-5">{plan.compounds.map((compound, i) => <div key={i} className="flex items-center gap-3"><Tyre compound={compound} />{i < plan.pits.length && <div className="text-center"><p className="text-[9px] text-white/40">PIT LAP {plan.pits[i]}</p><ArrowRight size={15} className="mx-auto text-white/25" /></div>}</div>)}</div>
    <div className="flex h-2 rounded overflow-hidden gap-0.5 mb-2">{plan.compounds.map((c, i) => <div key={i} style={{ width: `${plan.lengths[i] / laps * 100}%`, background: COLORS[c], opacity: .85 }} />)}</div>
    <div className="flex justify-between font-mono text-[10px] text-white/40 mb-5"><span>L1</span><span>{plan.lengths.join(' / ')} lap stints</span><span>L{laps}</span></div>
    <dl className="grid grid-cols-2 gap-4 text-xs"><div><dt className="text-white/40 mb-1">Pit sensitivity window</dt><dd className="font-mono">{plan.pit_windows.map(w => `L${w[0]}–${w[1]}`).join(' / ')}</dd></div><div><dt className="text-white/40 mb-1">Paired time difference</dt><dd className="font-mono">{rank === 0 ? 'Reference strategy' : `${signed(plan.p10_delta)} to ${signed(plan.p90_delta)}`}</dd></div></dl>
    <p className="mt-4 text-[10px] leading-relaxed text-white/35">Free-air elapsed time; P1 means model rank, not finishing position. Windows are within 1 modeled second of this sequence’s optimum.</p>
  </div>;
}

export default function PreRaceStrategy() {
  const [weekends, setWeekends] = useState<Weekend[]>([]);
  const [selection, setSelection] = useState('');
  const [report, setReport] = useState<PreRaceReport | null>(null);
  const [driverCode, setDriverCode] = useState('');
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [retry, setRetry] = useState(0);
  const [deg, setDeg] = useState<ScenarioLevel>('base');
  const [pit, setPit] = useState<ScenarioLevel>('base');
  const [compound, setCompound] = useState<Compound>('MEDIUM');

  useEffect(() => {
    let active = true;
    loadStrategyCatalog().then(result => {
      if (!active) return;
      setWeekends(result.data.weekends);
      setSelection(prev => prev || (result.data.weekends[0] ? `${result.data.weekends[0].year}-${result.data.weekends[0].round}` : ''));
      if (!result.data.weekends.length) { setError('No weekends have been published yet.'); setLoading(false); }
    }).catch(() => { if (active) { setError('The prepared catalog could not be loaded. Please retry.'); setLoading(false); } });
    return () => { active = false; };
  }, [retry]);
  useEffect(() => {
    if (!selection) return;
    let active = true;
    const [year, round] = selection.split('-').map(Number);
    loadPreRaceReport(year, round).then(result => {
      if (!active) return;
      setReport(result.data); setSaved(result.saved); setError(''); setLoading(false);
      const drivers = Object.values(result.data.drivers).sort((a, b) => b.practice_samples - a.practice_samples);
      setDriverCode(drivers[0]?.code ?? '');
    }).catch(e => { if (active) { setError(e.message); setLoading(false); } });
    return () => { active = false; };
  }, [selection, retry]);
  const driver = report?.drivers[driverCode];
  const plans = driver?.scenarios[`${deg}:${pit}`] ?? [];
  const practiceRows = report ? Object.values(report.drivers).flatMap(d => d.stints).filter(s => s.long_run && s.compound === compound) : [];

  return <div className="space-y-6">
    <section className="relative overflow-hidden rounded-2xl border border-white/10 bg-gradient-to-br from-[#191019] via-[#100e18] to-[#0b0c14] p-6 sm:p-8">
      <div className="absolute -right-16 -top-24 w-96 h-96 rounded-full border-[48px] border-red-500/[.035] pointer-events-none" />
      <div className="relative flex flex-wrap justify-between gap-6 items-end"><div><div className="flex items-center gap-2 text-red-400 text-[10px] uppercase tracking-[.2em] font-bold mb-3"><span className="w-1.5 h-1.5 rounded-full bg-red-500" />Strategy lab · Pre-race</div><h1 className="text-3xl sm:text-4xl font-bold tracking-tight">Every stint. A decision.</h1><p className="mt-3 text-sm text-white/50 max-w-xl leading-relaxed">Turn practice and qualifying into a race-day starting plan. Explore the evidence, compare tyre strategies, and see which assumptions change the call.</p></div><div className="flex items-center gap-2 border border-white/10 rounded-full px-3 py-2 text-[10px] text-white/55"><ShieldCheck size={13} />Same-era training · Pre-race inputs only</div></div>
    </section>

    <div className={`${PANEL} p-4 flex flex-wrap items-end gap-4`}>
      <div className="flex-1 min-w-52"><label htmlFor="strategy-weekend" className="block text-[10px] uppercase tracking-widest text-white/45 mb-2">Prepared weekend</label><select id="strategy-weekend" value={selection} onChange={e => { setSelection(e.target.value); setLoading(true); setReport(null); setDeg('base'); setPit('base'); }} className="w-full bg-[#171720] border border-white/15 rounded-lg px-3 py-2.5 text-sm"><option value="" disabled>Select a weekend</option>{weekends.map(w => <option key={`${w.year}-${w.round}`} value={`${w.year}-${w.round}`}>{w.year} · {w.event}</option>)}</select></div>
      <div className="min-w-44"><label htmlFor="strategy-driver" className="block text-[10px] uppercase tracking-widest text-white/45 mb-2">Driver</label><select id="strategy-driver" disabled={!report || loading} value={driverCode} onChange={e => setDriverCode(e.target.value)} className="w-full bg-[#171720] border border-white/15 rounded-lg px-3 py-2.5 text-sm">{report && Object.values(report.drivers).map(d => <option key={d.code} value={d.code}>{d.code} · {d.name}</option>)}</select></div>
      <button onClick={() => { setRetry(n => n + 1); setLoading(true); }} aria-label="Refresh prepared data" className="p-3 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10"><RefreshCw size={16} className={loading ? 'animate-spin' : ''} /></button>
      {report && <div className="pb-1.5 text-[11px] text-white/45"><p className="text-white/70">{report.manifest.era} car era</p><p className="mt-1">{report.manifest.kind}</p></div>}
    </div>
    {error && <div role="alert" className="rounded-xl bg-red-500/10 border border-red-500/25 text-red-300 p-4 text-sm">{error}</div>}
    {loading && <div role="status" className={`${PANEL} p-16 text-center`}><RefreshCw size={24} className="animate-spin mx-auto text-red-400 mb-4" /><p className="text-sm text-white/60">Opening the strategy workspace…</p><p className="mt-2 text-xs text-white/35">A saved report is available if the free data service is waking up.</p></div>}
    {!loading && report && driver && <>
      {saved && <div role="status" className="flex items-start gap-3 px-4 py-3 rounded-xl bg-amber-400/[.06] border border-amber-400/20 text-xs text-amber-200/80"><Info size={15} className="shrink-0" /><span>Viewing a saved report. The data service is waking or unavailable. All prepared comparisons work here; refresh to reconnect.</span></div>}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">{[
        { icon: Flag, label: 'Race distance', value: `${report.policy.laps} laps`, detail: 'Full Grand Prix · dry scenario' },
        { icon: Layers, label: 'Driver evidence', value: `${driver.practice_samples} laps`, detail: 'Usable practice long-run samples' },
        { icon: Timer, label: 'Pit-loss assumption', value: `${(report.policy.pit_loss + ({ low: -3, base: 0, high: 3 }[pit])).toFixed(0)} seconds`, detail: `Scenario ±${report.policy.pit_sigma}s · not measured` },
        { icon: ChartNoAxesCombined, label: 'Model support', value: report.support === 'scenario-only' ? 'Scenario only' : 'Limited evidence', detail: `${report.manifest.training_events.length} earlier same-era events` },
      ].map(card => <div key={card.label} className={`${PANEL} p-4`}><div className="flex items-center gap-2 mb-3"><card.icon size={13} className="text-white/40" /><Eyebrow>{card.label}</Eyebrow></div><p className="text-xl font-semibold tracking-tight">{card.value}</p><p className="text-[10px] text-white/40 mt-2">{card.detail}</p></div>)}</div>

      <p className="text-xs text-amber-200/75 rounded-xl border border-amber-300/15 p-4">Experimental comparison: on the two available held-out races, this fitted model had 2.436s lap MAE versus 1.115s for the simpler historical-effects baseline. Predictive improvement is unproven. <a className="underline" href="https://github.com/EliteGamer007/F1-Telemetry-Analysis/blob/main/docs/prerace-model-card.md">Validation details</a></p><section className={`${PANEL} p-5`}><div className="flex flex-wrap justify-between items-start gap-4"><div><Eyebrow>01 / Strategy comparison</Eyebrow><h2 className="mt-2 text-lg font-semibold">The starting plan for <span style={{ color: driver.color }}>{driver.code}</span></h2></div><div className="flex flex-wrap gap-4">{([{ name: 'Degradation', value: deg, set: setDeg }, { name: 'Pit loss', value: pit, set: setPit }]).map(control => <fieldset key={control.name}><legend className="text-[10px] text-white/45 mb-1.5">{control.name}</legend><div className="flex gap-1 bg-black/30 rounded-lg p-1">{LEVELS.map(level => <button key={level} aria-pressed={control.value === level} onClick={() => control.set(level)} className={`px-3 py-1.5 text-xs rounded-md capitalize transition ${control.value === level ? 'bg-red-500 text-white' : 'text-white/50 hover:bg-white/10'}`}>{level}</button>)}</div></fieldset>)}</div></div>
        <p className="text-xs text-white/45 mt-3 leading-relaxed">Conditional on a dry, uninterrupted race and assumed fresh tyre sets. Ranks minimize free-air time; traffic, safety cars and finishing positions are outside this model.</p>
        {report.support === 'scenario-only' && <p className="text-xs text-amber-300/80 mt-2">No historical practice-to-race calibration is available in this era’s cached sample. Treat these as comparative scenarios, not validated race-time predictions.</p>}
        <div className="grid lg:grid-cols-3 gap-4 mt-5">{plans.map((plan, index) => <PlanCard key={`${plan.compounds.join('-')}-${index}`} plan={plan} rank={index} laps={report.policy.laps} />)}</div>
        {!plans.length && <p className="py-10 text-sm text-white/50">Insufficient usable practice evidence for this driver. No strategy ranking has been fabricated.</p>}
      </section>

      <div className="grid xl:grid-cols-[1.5fr_1fr] gap-5">
        <section className={`${PANEL} p-5`}><Eyebrow>02 / Projected stint pace</Eyebrow><div className="flex flex-wrap justify-between gap-3 mt-2 mb-4"><h2 className="text-base font-semibold">How the compounds could evolve</h2><div className="flex gap-3">{(Object.keys(COLORS) as Compound[]).map(c => <span key={c} className="text-[10px] flex items-center gap-1.5" style={{ color: COLORS[c] }}><span className="w-3 h-0.5" style={{ background: COLORS[c] }} />{c[0] + c.slice(1).toLowerCase()}</span>)}</div></div><PaceChart curves={driver.curves} /><p className="text-[10px] text-white/40 mt-2">Fresh stint starting on lap 1; includes empirical race progression. Shading: uncalibrated 10–90% model range. Unknown fuel and traffic can dominate these differences.</p><div className="grid grid-cols-3 gap-3 mt-5">{(Object.keys(COLORS) as Compound[]).map(c => <div key={c} className="border-t border-white/10 pt-3"><div className="flex gap-2 items-center"><Tyre compound={c} small /><span className="font-mono text-sm">{driver.curves[c][9] ? seconds(driver.curves[c][9].median) : '—'}</span></div><p className="text-[10px] text-white/35 mt-2">Lap 10 projection · {driver.compounds[c].practice_samples} practice laps</p><p className="text-[10px] text-white/45 mt-1">{driver.compounds[c].source}</p></div>)}</div></section>
        <section className={`${PANEL} p-5`}><Eyebrow>03 / Sensitivity map</Eyebrow><h2 className="text-base font-semibold mt-2">Does the call survive a change?</h2><p className="text-xs text-white/40 mt-2 mb-4">Select a cell to inspect its best strategy. Low/high degradation scales the fitted trend by 0.65×/1.35×; pit loss shifts by ±3s.</p><div className="grid grid-cols-[auto_repeat(3,1fr)] gap-2 text-[10px]"><span className="self-end text-white/35">DEG ↓ / PIT →</span>{LEVELS.map(l => <span key={l} className="text-center text-white/40 uppercase">{l}</span>)}{LEVELS.map(d => <div className="contents" key={d}><span className="self-center uppercase text-white/40">{d}</span>{LEVELS.map(p => { const best = driver.scenarios[`${d}:${p}`]?.[0]; return <button key={p} aria-label={`${d} degradation, ${p} pit loss`} aria-pressed={deg === d && pit === p} onClick={() => { setDeg(d); setPit(p); }} className={`rounded-lg py-4 border transition ${deg === d && pit === p ? 'bg-red-500/15 border-red-500/50' : 'bg-white/[.02] border-white/5 hover:border-white/20'}`}><p className="font-mono font-semibold text-xs">{best?.compounds.map(c => c[0]).join(' → ') ?? '—'}</p><p className="text-white/40 mt-1">{best ? `${best.pits.length} stop · L${best.pits.join(', ')}` : 'No estimate'}</p></button>; })}</div>)}</div><div className="flex gap-2 mt-5 text-xs text-white/45 leading-relaxed"><ArrowDownUp size={16} className="shrink-0" /><span>A different winner means the recommendation depends on the assumptions. Small modeled gaps are not decisive evidence.</span></div></section>
      </div>

      <div className="grid xl:grid-cols-2 gap-5">
        <section className={`${PANEL} p-5`}><div className="flex flex-wrap items-center justify-between gap-3"><div><Eyebrow>04 / Observed practice</Eyebrow><h2 className="text-base font-semibold mt-2">Long-run evidence across the field</h2></div><select aria-label="Practice comparison compound" value={compound} onChange={e => setCompound(e.target.value as Compound)} className="text-xs bg-[#171720] border border-white/15 rounded-lg p-2">{Object.keys(COLORS).map(c => <option key={c}>{c}</option>)}</select></div><p className="text-xs text-white/40 mt-2 mb-4">Measured timed laps, grouped by actual tyre set. Compare the age ranges and session before judging pace; fuel programmes differ.</p><div className="max-h-72 overflow-auto"><table className="w-full text-xs text-left"><thead className="text-[10px] text-white/40 sticky top-0 bg-[#101018]"><tr>{['Driver / run', 'Age', 'Laps', 'Median'].map(label => <th key={label} className="py-2 font-medium">{label}</th>)}</tr></thead><tbody>{practiceRows.map((s, i) => <tr key={i} className={`border-t border-white/5 ${s.driver === driverCode ? 'bg-red-500/[.07]' : ''}`}><td className="py-2.5"><button onClick={() => setDriverCode(s.driver)} className="font-semibold hover:text-red-400">{s.driver}</button><span className="text-white/35 ml-2 text-[10px]">{s.session} · set {s.stint}</span></td><td className="font-mono text-white/60">{s.age_min}–{s.age_max}</td><td className="font-mono text-white/60">{s.samples}</td><td className="font-mono">{seconds(s.median_seconds)}</td></tr>)}</tbody></table>{!practiceRows.length && <p className="py-8 text-white/40 text-xs">No qualifying long runs on this compound.</p>}</div></section>
        <section className={`${PANEL} p-5`}><Eyebrow>05 / Tyre-set history</Eyebrow><h2 className="text-base font-semibold mt-2">{driver.code} · the runs behind the model</h2><p className="text-xs text-white/40 mt-2 mb-5">Separate sets stay separate, including consecutive sets of the same compound. Bars show the observed lap span.</p><div className="space-y-3 max-h-64 overflow-auto">{driver.stints.map((s, i) => <div key={i} className="grid grid-cols-[68px_1fr_70px] items-center gap-3"><span className="text-[10px] text-white/45">{s.session} / #{s.stint}</span><div className="relative bg-white/[.03] h-7 rounded"><div className="absolute h-7 rounded flex items-center px-2 text-[10px] text-black font-semibold min-w-10" style={{ left: `${(s.start_lap - 1) / 65 * 70}%`, width: `${Math.min(100 - (s.start_lap - 1) / 65 * 70, Math.max(12, (s.end_lap - s.start_lap + 1) / 65 * 70))}%`, background: COLORS[s.compound], opacity: s.long_run ? .9 : .4 }}>{s.compound[0]} · {s.samples}</div></div><span className="text-[10px] text-white/40">{s.fresh === null ? 'Fresh?' : s.fresh ? 'Fresh' : 'Used'}<br />{s.long_run ? 'Long run' : 'Short run'}</span></div>)}</div></section>
      </div>

      <section className={`${PANEL} p-5`}><div className="flex items-center gap-2"><Info size={16} className="text-red-400" /><h2 className="text-sm font-semibold">Evidence, assumptions & provenance</h2></div><div className="grid md:grid-cols-3 gap-6 mt-5"><div><Eyebrow>Available before the cutoff</Eyebrow><div className="flex flex-wrap gap-2 mt-3">{report.coverage.sessions.map(s => <span key={s.code} className="px-2.5 py-1.5 bg-emerald-400/[.07] border border-emerald-400/15 text-emerald-300 text-[11px] rounded-md">{s.code} · {s.laps} laps</span>)}{report.coverage.missing.map(s => <span key={s.code} className="px-2.5 py-1.5 border border-white/10 text-white/35 text-[11px] rounded-md">{s.code} unavailable</span>)}</div><p className="text-[11px] text-white/45 mt-3">Input cutoff: {new Date(report.manifest.cutoff).toLocaleString('en-GB', { timeZone: 'UTC' })} UTC</p><p className="text-[10px] text-white/30 mt-2">Historical reconstruction using cached observations, not a forecast published before this race. Weather observations are not a race-weather forecast.</p></div><div><Eyebrow>Model support</Eyebrow><ul className="space-y-2 mt-3 text-xs text-white/50">{driver.evidence.map(e => <li key={e}>{e}</li>)}</ul><p className="mt-3 text-[10px] text-white/35">Model {report.manifest.model_version} · {report.coverage.training_laps.toLocaleString()} clean historical laps · {report.coverage.calibration_events} practice/race calibration events</p></div><div><Eyebrow>Conditions of the comparison</Eyebrow><p className="text-xs text-white/50 mt-3 leading-relaxed">{report.policy.inventory_source}</p><p className="text-xs text-white/50 mt-2 leading-relaxed">{report.policy.pit_source}</p><p className="text-[10px] text-amber-300/65 mt-3">{report.uncertainty}</p></div></div><details className="mt-5 border-t border-white/10 pt-4"><summary className="cursor-pointer text-xs text-white/50 flex items-center gap-2"><ChevronDown size={14} />Full assumptions and source identifiers</summary><ul className="list-disc ml-5 mt-3 text-xs text-white/45 space-y-2">{report.assumptions.map(a => <li key={a}>{a}</li>)}</ul><div className="mt-4 text-[10px] text-white/30 font-mono break-all space-y-1"><p>Model: {report.manifest.model_hash}</p><p>Inputs: {report.manifest.source_hash}</p><p>Training events: {report.manifest.training_events.join(', ') || 'None'}</p>{Object.entries(report.manifest.metadata).map(([k, v]) => <p key={k}>{k}: {v}</p>)}</div></details></section>
      <p className="text-[10px] text-center text-white/25 pb-4">Independent educational project · FastF1 data · No affiliation with Formula 1 or any team · Experimental analysis, not live timing</p>
    </>}
  </div>;
}
