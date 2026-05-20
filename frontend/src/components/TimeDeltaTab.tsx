'use client';

import { SessionData } from '@/lib/api';
import { useState, useMemo, useRef, useCallback } from 'react';

interface TimeDeltaProps {
  sessionData: SessionData;
}

// ── Shared crosshair hook ────────────────────────────────────────
function useCrosshair(maxDist: number) {
  const [crossPct, setCrossPct] = useState<number | null>(null);
  const queryDist = crossPct !== null ? crossPct * maxDist : null;

  const onMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setCrossPct((e.clientX - rect.left) / rect.width);
  }, []);
  const onLeave = useCallback(() => setCrossPct(null), []);

  return { crossPct, queryDist, onMove, onLeave };
}

// ── Interpolation utility ────────────────────────────────────────
function interpAt(dists: number[], vals: number[], queryDist: number): number {
  const n = dists.length;
  if (!n) return 0;
  if (queryDist <= dists[0]) return vals[0];
  if (queryDist >= dists[n - 1]) return vals[n - 1];
  let lo = 0, hi = n - 1;
  while (lo < hi - 1) { const m = (lo + hi) >> 1; if (dists[m] <= queryDist) lo = m; else hi = m; }
  const t = (queryDist - dists[lo]) / (dists[hi] - dists[lo]);
  return vals[lo] + t * (vals[hi] - vals[lo]);
}

// ── SVG path builders ────────────────────────────────────────────
function buildLinePath(xs: number[], ys: number[], W: number, H: number, maxX: number, minY: number, maxY: number): string {
  const range = maxY - minY || 1;
  return xs.map((x, i) => {
    const px = (x / maxX) * W;
    const py = H - ((ys[i] - minY) / range) * H;
    return `${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${py.toFixed(1)}`;
  }).join(' ');
}

// ── Delta chart component ────────────────────────────────────────
interface DeltaChartProps {
  distance: number[];
  delta: number[];
  d1: string; d2: string;
  c1: string; c2: string;
  maxDist: number;
}

function DeltaChart({ distance, delta, d1, d2, c1, c2, maxDist }: DeltaChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const W = 860, H = 160;
  const { crossPct, queryDist, onMove, onLeave } = useCrosshair(maxDist);

  const absMax = Math.max(Math.abs(Math.min(...delta)), Math.abs(Math.max(...delta))) * 1.1 || 0.5;

  const toY = (v: number) => H / 2 - (v / absMax) * (H * 0.46);
  const toX = (d: number) => (d / maxDist) * W;

  // Build separate fill paths: above zero (d1 faster), below zero (d2 faster)
  const linePath = distance.map((d, i) =>
    `${i === 0 ? 'M' : 'L'}${toX(d).toFixed(1)},${toY(delta[i]).toFixed(1)}`
  ).join(' ');

  const fillPathAbove = distance.map((d, i) => {
    const y = Math.min(toY(delta[i]), H / 2);
    return `${i === 0 ? 'M' : 'L'}${toX(d).toFixed(1)},${y.toFixed(1)}`;
  }).join(' ') + ` L${toX(maxDist)},${H / 2} L${toX(distance[0])},${H / 2} Z`;

  const fillPathBelow = distance.map((d, i) => {
    const y = Math.max(toY(delta[i]), H / 2);
    return `${i === 0 ? 'M' : 'L'}${toX(d).toFixed(1)},${y.toFixed(1)}`;
  }).join(' ') + ` L${toX(maxDist)},${H / 2} L${toX(distance[0])},${H / 2} Z`;

  const tooltipDelta = queryDist !== null ? interpAt(distance, delta, queryDist) : null;
  const crossX = crossPct !== null ? crossPct * W : null;

  return (
    <div className="relative">
      <svg ref={svgRef} width="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none"
        style={{ height: H, display: 'block', cursor: 'crosshair' }}
        onMouseMove={onMove} onMouseLeave={onLeave}>
        <defs>
          <linearGradient id="gAbove" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={c1} stopOpacity="0.35" />
            <stop offset="100%" stopColor={c1} stopOpacity="0.03" />
          </linearGradient>
          <linearGradient id="gBelow" x1="0" y1="1" x2="0" y2="0">
            <stop offset="0%" stopColor={c2} stopOpacity="0.35" />
            <stop offset="100%" stopColor={c2} stopOpacity="0.03" />
          </linearGradient>
        </defs>

        {/* Grid */}
        {[-0.4, -0.2, 0, 0.2, 0.4].map(f => (
          <line key={f} x1={0} y1={H / 2 - f * H * 0.46 / absMax * absMax}
            x2={W} y2={H / 2 - f * H * 0.46 / absMax * absMax}
            stroke={f === 0 ? 'rgba(255,255,255,0.18)' : 'rgba(255,255,255,0.04)'}
            strokeWidth={f === 0 ? 1.5 : 1}
            strokeDasharray={f === 0 ? '4,4' : undefined}
            vectorEffect="non-scaling-stroke" />
        ))}

        {/* Fills */}
        <path d={fillPathAbove} fill="url(#gAbove)" />
        <path d={fillPathBelow} fill="url(#gBelow)" />

        {/* Line */}
        <path d={linePath} fill="none" stroke={c1} strokeWidth="2"
          strokeLinecap="round" vectorEffect="non-scaling-stroke" />

        {/* Crosshair */}
        {crossX !== null && (
          <>
            <line x1={crossX} y1={0} x2={crossX} y2={H}
              stroke="rgba(255,255,255,0.3)" strokeWidth="1" strokeDasharray="3,3"
              vectorEffect="non-scaling-stroke" />
            {tooltipDelta !== null && (
              <circle cx={crossX} cy={toY(tooltipDelta)} r="4"
                fill={tooltipDelta >= 0 ? c1 : c2} stroke="#000" strokeWidth="1.5"
                vectorEffect="non-scaling-stroke" />
            )}
          </>
        )}
      </svg>

      {/* Y axis labels */}
      <div className="absolute left-0 top-0 h-full flex flex-col justify-between pointer-events-none py-1 text-[9px] font-mono text-white/25 -ml-7 w-7 text-right">
        <span>+{absMax.toFixed(1)}s</span>
        <span>0</span>
        <span>-{absMax.toFixed(1)}s</span>
      </div>

      {/* Tooltip */}
      {crossX !== null && tooltipDelta !== null && (
        <div className="absolute top-2 pointer-events-none z-10"
          style={{ left: Math.min(crossX + 10, W - 160) }}>
          <div className="bg-[#0c0c18] border border-white/12 rounded-lg px-3 py-2 shadow-xl min-w-[140px]">
            <p className="text-[9px] font-mono text-white/35 mb-1">
              {((queryDist ?? 0) / 1000).toFixed(2)} km
            </p>
            <div className="flex items-center justify-between gap-3">
              <span className="text-[11px] font-bold text-white/70">
                {tooltipDelta >= 0 ? d1 : d2} faster
              </span>
              <span className="font-mono text-[12px] font-black"
                style={{ color: tooltipDelta >= 0 ? c1 : c2 }}>
                {Math.abs(tooltipDelta).toFixed(3)}s
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Speed comparison chart ────────────────────────────────────────
interface SpeedChartProps {
  distance: number[];
  speed1: number[]; speed2: number[];
  d1: string; d2: string;
  c1: string; c2: string;
  maxDist: number;
}

function SpeedChart({ distance, speed1, speed2, d1, d2, c1, c2, maxDist }: SpeedChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const W = 860, H = 110;
  const { crossPct, queryDist, onMove, onLeave } = useCrosshair(maxDist);

  const allSpeeds = [...speed1, ...speed2];
  const minS = Math.min(...allSpeeds), maxS = Math.max(...allSpeeds);

  const p1 = buildLinePath(distance, speed1, W, H, maxDist, minS, maxS);
  const p2 = buildLinePath(distance, speed2, W, H, maxDist, minS, maxS);
  const crossX = crossPct !== null ? crossPct * W : null;

  const v1 = queryDist !== null ? interpAt(distance, speed1, queryDist) : null;
  const v2 = queryDist !== null ? interpAt(distance, speed2, queryDist) : null;

  return (
    <div className="relative">
      <svg ref={svgRef} width="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none"
        style={{ height: H, display: 'block', cursor: 'crosshair' }}
        onMouseMove={onMove} onMouseLeave={onLeave}>
        {[0, 0.25, 0.5, 0.75, 1].map(f => (
          <line key={f} x1={0} y1={H * f} x2={W} y2={H * f}
            stroke="rgba(255,255,255,0.04)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
        ))}
        <path d={p1} fill="none" stroke={c1} strokeWidth="1.8"
          strokeLinecap="round" vectorEffect="non-scaling-stroke" />
        <path d={p2} fill="none" stroke={c2} strokeWidth="1.8"
          strokeLinecap="round" vectorEffect="non-scaling-stroke" />
        {crossX !== null && (
          <line x1={crossX} y1={0} x2={crossX} y2={H}
            stroke="rgba(255,255,255,0.3)" strokeWidth="1" strokeDasharray="3,3"
            vectorEffect="non-scaling-stroke" />
        )}
      </svg>

      {/* Tooltip */}
      {crossX !== null && v1 !== null && v2 !== null && (
        <div className="absolute top-1 pointer-events-none z-10"
          style={{ left: Math.min(crossX + 10, W - 150) }}>
          <div className="bg-[#0c0c18] border border-white/12 rounded-lg px-3 py-2 shadow-xl">
            <p className="text-[9px] font-mono text-white/35 mb-1.5">
              {((queryDist ?? 0) / 1000).toFixed(2)} km
            </p>
            {[{ d: d1, v: v1, c: c1 }, { d: d2, v: v2, c: c2 }].map(({ d, v, c }) => (
              <div key={d} className="flex items-center justify-between gap-4">
                <span className="flex items-center gap-1.5 text-[11px] text-white/60">
                  <span className="w-2 h-2 rounded-full" style={{ background: c }} />{d}
                </span>
                <span className="font-mono text-[11px] font-bold" style={{ color: c }}>
                  {Math.round(v)} km/h
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Y labels */}
      <div className="absolute left-0 top-0 h-full flex flex-col justify-between pointer-events-none py-0.5 text-[9px] font-mono text-white/25 -ml-8 w-8 text-right">
        <span>{Math.round(maxS)}</span>
        <span>{Math.round(minS)}</span>
      </div>
    </div>
  );
}

// ── Main component ───────────────────────────────────────────────
export default function TimeDeltaTab({ sessionData }: TimeDeltaProps) {
  const drivers = sessionData.leaderboard.map(r => r.driver);
  const driverColors: Record<string, string> = {};
  const driverTimes: Record<string, string> = {};
  for (const r of sessionData.leaderboard) {
    driverColors[r.driver] = r.color;
    driverTimes[r.driver] = r.bestLapStr;
  }

  const [d1, setD1] = useState(drivers[0] ?? '');
  const [d2, setD2] = useState(drivers[1] ?? '');
  const [error, setError] = useState('');

  // ── Compute delta client-side from existing telemetry ──────────
  const computed = useMemo(() => {
    if (!d1 || !d2 || d1 === d2) return null;
    const tel1 = sessionData.telemetry[d1];
    const tel2 = sessionData.telemetry[d2];
    if (!tel1 || !tel2) return null;

    const d1arr = tel1.Distance ?? [];
    const t1arr = tel1.Trel ?? [];
    const d2arr = tel2.Distance ?? [];
    const t2arr = tel2.Trel ?? [];
    const s1arr = tel1.Speed ?? [];
    const s2arr = tel2.Speed ?? [];

    if (!d1arr.length || !d2arr.length) return null;

    const maxDist = Math.min(Math.max(...d1arr), Math.max(...d2arr));
    const N = 600;
    const distance = Array.from({ length: N }, (_, i) => (i / (N - 1)) * maxDist);

    const delta = distance.map(d => interpAt(d1arr, t1arr, d) - interpAt(d2arr, t2arr, d));
    const speed1 = distance.map(d => interpAt(d1arr, s1arr, d));
    const speed2 = distance.map(d => interpAt(d2arr, s2arr, d));

    return { distance, delta, speed1, speed2, maxDist };
  }, [d1, d2, sessionData.telemetry]);

  const handleApply = () => {
    if (d1 === d2) { setError('Select two different drivers.'); return; }
    setError('');
  };

  const c1 = driverColors[d1] ?? '#FF8000';
  const c2 = driverColors[d2] ?? '#27F4D2';

  return (
    <div className="flex flex-col gap-5">
      {/* ── Selector card ─────────────────────────────────────── */}
      <div className="bg-white/[0.03] rounded-2xl border border-white/10 p-5">
        <h3 className="text-white font-bold text-xs uppercase tracking-widest mb-4 flex items-center gap-2">
          <span className="w-1 h-4 bg-red-500 rounded-full" />Time Delta Comparison
        </h3>

        <div className="flex items-end gap-5 flex-wrap">
          {/* Driver 1 */}
          <div>
            <p className="text-white/35 text-[10px] mb-2 uppercase tracking-wider">Reference Driver</p>
            <div className="flex flex-wrap gap-1.5">
              {drivers.map(d => (
                <button key={d} onClick={() => { setD1(d); setError(''); }}
                  className={`px-2.5 py-1 rounded-lg text-[11px] font-bold transition-all ${d1 === d ? 'shadow-md ring-2 ring-white/20' : 'bg-white/8 text-white/50 hover:bg-white/15'}`}
                  style={d1 === d ? { background: driverColors[d], color: '#fff' } : {}}>
                  {d}
                </button>
              ))}
            </div>
          </div>

          <div className="text-white/20 text-2xl font-thin pb-1">vs</div>

          {/* Driver 2 */}
          <div>
            <p className="text-white/35 text-[10px] mb-2 uppercase tracking-wider">Compare Driver</p>
            <div className="flex flex-wrap gap-1.5">
              {drivers.map(d => (
                <button key={d} onClick={() => { setD2(d); setError(''); }}
                  className={`px-2.5 py-1 rounded-lg text-[11px] font-bold transition-all ${d2 === d ? 'shadow-md ring-2 ring-white/20' : 'bg-white/8 text-white/50 hover:bg-white/15'}`}
                  style={d2 === d ? { background: driverColors[d], color: '#fff' } : {}}>
                  {d}
                </button>
              ))}
            </div>
          </div>
        </div>

        {error && <p className="mt-3 text-red-400 text-[12px]">{error}</p>}

        {/* Quick summary strip */}
        {d1 && d2 && d1 !== d2 && (
          <div className="mt-4 flex items-center gap-6">
            {[{ d: d1, c: c1 }, { d: d2, c: c2 }].map(({ d, c }) => (
              <span key={d} className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-sm" style={{ background: c }} />
                <span className="text-[12px] font-bold text-white">{d}</span>
                <span className="font-mono text-emerald-400 text-[12px]">{driverTimes[d]}</span>
              </span>
            ))}
            {computed && (() => {
              const finalDelta = computed.delta[computed.delta.length - 1];
              const faster = finalDelta < 0 ? d2 : d1;
              const margin = Math.abs(finalDelta);
              return (
                <span className="ml-auto text-[11px] text-white/50">
                  <span className="font-bold text-white">{faster}</span> ahead by{' '}
                  <span className="font-mono text-yellow-400">{margin.toFixed(3)}s</span>
                </span>
              );
            })()}
          </div>
        )}
      </div>

      {/* ── Charts ──────────────────────────────────────────────── */}
      {computed ? (
        <>
          {/* Delta chart */}
          <div className="bg-white/[0.03] rounded-2xl border border-white/10 p-5 pl-9">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-white/80 font-semibold text-[13px]">
                Time Delta
                <span className="text-white/30 text-[11px] ml-2">(+ve = {d1} slower)</span>
              </h4>
              <div className="flex gap-4 text-[11px]">
                {[{ d: d1, c: c1, label: 'faster above' }, { d: d2, c: c2, label: 'faster below' }].map(({ d, c, label }) => (
                  <span key={d} className="flex items-center gap-1.5 text-white/50">
                    <span className="w-3 h-0.5 rounded" style={{ background: c }} />
                    <span className="font-bold text-white/80">{d}</span> {label}
                  </span>
                ))}
              </div>
            </div>
            <DeltaChart
              distance={computed.distance} delta={computed.delta}
              d1={d1} d2={d2} c1={c1} c2={c2} maxDist={computed.maxDist}
            />
            <div className="flex justify-between text-[9px] font-mono text-white/20 mt-1.5">
              {[0, 0.25, 0.5, 0.75, 1].map(f => (
                <span key={f}>{(f * computed.maxDist / 1000).toFixed(1)}km</span>
              ))}
            </div>
          </div>

          {/* Speed chart */}
          <div className="bg-white/[0.03] rounded-2xl border border-white/10 p-5 pl-9">
            <h4 className="text-white/80 font-semibold text-[13px] mb-3">
              Speed Trace
              <span className="text-white/30 text-[11px] ml-2">(km/h)</span>
            </h4>
            <SpeedChart
              distance={computed.distance}
              speed1={computed.speed1} speed2={computed.speed2}
              d1={d1} d2={d2} c1={c1} c2={c2} maxDist={computed.maxDist}
            />
            <div className="flex justify-between text-[9px] font-mono text-white/20 mt-1.5">
              {[0, 0.25, 0.5, 0.75, 1].map(f => (
                <span key={f}>{(f * computed.maxDist / 1000).toFixed(1)}km</span>
              ))}
            </div>
          </div>
        </>
      ) : (
        <div className="flex items-center justify-center py-16 text-white/25 text-sm">
          {d1 === d2 && d1 ? 'Select two different drivers above' : 'Select two drivers to compare'}
        </div>
      )}
    </div>
  );
}
