'use client';

import { SessionData } from '@/lib/api';
import { useState, useMemo, useRef, useCallback } from 'react';

interface LapAnalysisProps {
  sessionData: SessionData;
}

const CHANNELS = ['Speed', 'Throttle', 'Brake', 'RPM', 'nGear', 'DRS'] as const;
type Channel = typeof CHANNELS[number];

const CHANNEL_UNITS: Record<Channel, string> = {
  Speed: 'km/h', Throttle: '%', Brake: '%', RPM: 'rpm', nGear: '', DRS: '',
};

interface TooltipData {
  x: number;
  y: number;
  values: { driver: string; color: string; value: number }[];
  distanceKm: string;
}

function Chart({
  drivers,
  channel,
  driverColors,
  telemetry,
}: {
  drivers: string[];
  channel: Channel;
  driverColors: Record<string, string>;
  telemetry: SessionData['telemetry'];
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<TooltipData | null>(null);
  const [crossX, setCrossX] = useState<number | null>(null);

  const W = 860, H = 130;

  // Build chart data
  const { paths, yMin, yMax, maxDist } = useMemo(() => {
    const allVals: number[] = [];
    const allDists: number[] = [];

    for (const drv of drivers) {
      const tel = telemetry[drv];
      if (!tel) continue;
      const vals = (tel as Record<string, number[]>)[channel] ?? [];
      allVals.push(...vals);
      allDists.push(...(tel.Distance ?? []));
    }

    if (!allVals.length) return { paths: [], yMin: 0, yMax: 1, maxDist: 1 };

    const yMin = Math.min(...allVals);
    const yMax = Math.max(...allVals);
    const maxDist = Math.max(...allDists);

    const paths = drivers.map(drv => {
      const tel = telemetry[drv];
      if (!tel) return { drv, d: '' };
      const dists = tel.Distance ?? [];
      const vals = (tel as Record<string, number[]>)[channel] ?? [];
      if (!dists.length) return { drv, d: '' };
      const range = yMax - yMin || 1;
      const pts = dists.map((dist, i) => {
        const x = (dist / maxDist) * W;
        const y = H - ((vals[i] - yMin) / range) * H;
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
      });
      return { drv, d: pts.join(' ') };
    });

    return { paths, yMin, yMax, maxDist };
  }, [drivers, channel, telemetry]);

  const handleMouseMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const relX = ((e.clientX - rect.left) / rect.width) * W;
    const normX = relX / W;
    const queryDist = normX * maxDist;
    setCrossX(relX);

    const values = drivers.map(drv => {
      const tel = telemetry[drv];
      if (!tel) return null;
      const dists = tel.Distance ?? [];
      const vals = (tel as Record<string, number[]>)[channel] ?? [];
      // Binary search for closest distance
      let lo = 0, hi = dists.length - 1;
      while (lo < hi - 1) { const m = (lo + hi) >> 1; if (dists[m] <= queryDist) lo = m; else hi = m; }
      const t = hi > lo ? (queryDist - dists[lo]) / (dists[hi] - dists[lo]) : 0;
      const value = vals[lo] + t * (vals[hi] - vals[lo]);
      return { driver: drv, color: driverColors[drv] ?? '#888', value };
    }).filter(Boolean) as { driver: string; color: string; value: number }[];

    setTooltip({
      x: e.clientX - rect.left,
      y: 0,
      values,
      distanceKm: (queryDist / 1000).toFixed(2),
    });
  }, [drivers, channel, telemetry, driverColors, maxDist]);

  const handleMouseLeave = () => { setTooltip(null); setCrossX(null); };

  const range = yMax - yMin || 1;
  const yAxisLabels = [yMax, (yMax + yMin) / 2, yMin];

  return (
    <div className="bg-white/[0.03] rounded-xl border border-white/10 p-4 relative">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-white/70 font-semibold text-[13px]">
          {channel}
          {CHANNEL_UNITS[channel] && (
            <span className="text-white/30 text-[11px] ml-1">({CHANNEL_UNITS[channel]})</span>
          )}
        </h4>
        {/* Mini legend */}
        <div className="flex gap-3">
          {drivers.map(drv => (
            <span key={drv} className="flex items-center gap-1.5 text-[11px] text-white/60">
              <span className="w-4 h-[2px] rounded inline-block" style={{ background: driverColors[drv] }} />
              {drv}
            </span>
          ))}
        </div>
      </div>

      <div className="relative flex gap-2">
        {/* Y-axis */}
        <div className="flex flex-col justify-between text-right text-[9px] font-mono text-white/25 pointer-events-none" style={{ width: 32, height: H }}>
          {yAxisLabels.map((v, i) => (
            <span key={i}>{Math.round(v)}</span>
          ))}
        </div>

        {/* SVG chart */}
        <div className="flex-1 relative">
          <svg
            ref={svgRef}
            width="100%"
            viewBox={`0 0 ${W} ${H}`}
            preserveAspectRatio="none"
            className="block rounded cursor-crosshair overflow-visible"
            style={{ height: H }}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
          >
            {/* Grid lines */}
            {[0, 0.25, 0.5, 0.75, 1].map(f => (
              <line key={f} x1={0} y1={H * f} x2={W} y2={H * f}
                stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
            ))}

            {/* Driver paths */}
            {paths.map(({ drv, d }) => d && (
              <path key={drv} d={d}
                fill="none"
                stroke={driverColors[drv] ?? '#888'}
                strokeWidth="1.8"
                strokeLinecap="round"
                vectorEffect="non-scaling-stroke"
              />
            ))}

            {/* Vertical crosshair */}
            {crossX !== null && (
              <line
                x1={crossX} y1={0} x2={crossX} y2={H}
                stroke="rgba(255,255,255,0.35)"
                strokeWidth="1"
                strokeDasharray="3,3"
                vectorEffect="non-scaling-stroke"
              />
            )}
          </svg>

          {/* Hover tooltip */}
          {tooltip && (
            <div
              className="absolute z-20 pointer-events-none bg-[#0f0f1a] border border-white/15 rounded-lg shadow-xl px-3 py-2 min-w-[120px]"
              style={{
                left: Math.min(tooltip.x + 12, (svgRef.current?.getBoundingClientRect().width ?? 800) - 140),
                top: -8,
              }}
            >
              <p className="text-white/40 text-[9px] font-mono mb-1.5">{tooltip.distanceKm} km</p>
              {tooltip.values.map(v => (
                <div key={v.driver} className="flex items-center justify-between gap-3">
                  <span className="flex items-center gap-1.5 text-[11px] text-white/70">
                    <span className="w-2 h-2 rounded-full" style={{ background: v.color }} />
                    {v.driver}
                  </span>
                  <span className="font-mono text-[11px] font-bold" style={{ color: v.color }}>
                    {channel === 'Throttle' || channel === 'Brake'
                      ? `${Math.round(v.value)}%`
                      : channel === 'RPM'
                      ? `${Math.round(v.value).toLocaleString()}`
                      : Math.round(v.value * 10) / 10}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* X-axis: distance labels */}
      <div className="flex justify-between text-[9px] font-mono text-white/20 mt-1 pl-9">
        {[0, 0.25, 0.5, 0.75, 1].map(f => (
          <span key={f}>{(f * maxDist / 1000).toFixed(1)}km</span>
        ))}
      </div>
    </div>
  );
}

export default function LapAnalysis({ sessionData }: LapAnalysisProps) {
  const allDrivers = sessionData.leaderboard.map(r => r.driver);
  const driverColors: Record<string, string> = {};
  for (const r of sessionData.leaderboard) driverColors[r.driver] = r.color;

  const [selected, setSelected] = useState<string[]>(allDrivers.slice(0, 2));
  const [channels, setChannels] = useState<Channel[]>(['Speed']);
  const [active, setActive] = useState<{ drivers: string[]; channels: Channel[] } | null>(null);

  const toggleDriver = (drv: string) =>
    setSelected(prev => prev.includes(drv)
      ? prev.filter(d => d !== drv)
      : prev.length < 4 ? [...prev, drv] : prev);

  const toggleChannel = (ch: Channel) =>
    setChannels(prev => prev.includes(ch)
      ? prev.length > 1 ? prev.filter(c => c !== ch) : prev
      : prev.length < 3 ? [...prev, ch] : prev);

  return (
    <div className="flex flex-col gap-5">
      {/* Config card */}
      <div className="bg-white/[0.03] rounded-2xl border border-white/10 p-5">
        <h3 className="text-white font-bold text-xs uppercase tracking-widest mb-4 flex items-center gap-2">
          <span className="w-1 h-4 bg-red-500 rounded-full" />Configure Analysis
        </h3>
        <div className="grid grid-cols-2 gap-6">
          <div>
            <p className="text-white/35 text-[10px] mb-2 uppercase tracking-wider">Drivers (max 4)</p>
            <div className="flex flex-wrap gap-1.5">
              {allDrivers.map(drv => (
                <button key={drv} onClick={() => toggleDriver(drv)}
                  className={`px-2.5 py-1 rounded-lg text-[11px] font-bold transition-all ${
                    selected.includes(drv) ? 'shadow-md' : 'bg-white/8 text-white/50 hover:bg-white/15'
                  }`}
                  style={selected.includes(drv) ? { background: driverColors[drv], color: '#fff' } : {}}>
                  {drv}
                </button>
              ))}
            </div>
          </div>
          <div>
            <p className="text-white/35 text-[10px] mb-2 uppercase tracking-wider">Channels (max 3)</p>
            <div className="flex flex-wrap gap-1.5">
              {CHANNELS.map(ch => (
                <button key={ch} onClick={() => toggleChannel(ch)}
                  className={`px-3 py-1 rounded-lg text-[11px] font-bold transition-all ${
                    channels.includes(ch) ? 'bg-red-500 text-white' : 'bg-white/8 text-white/50 hover:bg-white/15'
                  }`}>
                  {ch}
                </button>
              ))}
            </div>
          </div>
        </div>
        <button onClick={() => setActive({ drivers: [...selected], channels: [...channels] })}
          className="mt-5 px-6 py-2 bg-red-500 hover:bg-red-600 active:scale-95 text-white font-bold rounded-xl text-sm transition-all">
          Load Analysis
        </button>
      </div>

      {/* Charts */}
      {active && active.channels.map(ch => (
        <Chart
          key={ch}
          drivers={active.drivers}
          channel={ch}
          driverColors={driverColors}
          telemetry={sessionData.telemetry}
        />
      ))}
    </div>
  );
}
