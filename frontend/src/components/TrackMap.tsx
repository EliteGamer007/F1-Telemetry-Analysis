'use client';

import React, { useEffect, useRef, useCallback, useState } from 'react';
import { TrackData, SessionData, TrackStatusEvent } from '@/lib/api';

interface TrackMapProps {
  trackData: TrackData;
  sessionData: SessionData;
  onTimeUpdate?: (time: number) => void;
}

// Track status styling
const STATUS_STYLE: Record<string, { color: string; label: string; bg: string }> = {
  AllClear:    { color: '#22c55e', label: 'GREEN', bg: 'bg-green-500/20 border-green-500/40 text-green-400' },
  Yellow:      { color: '#facc15', label: 'YELLOW', bg: 'bg-yellow-500/20 border-yellow-500/40 text-yellow-300' },
  SCDeploying: { color: '#f97316', label: 'SC DEPLOYING', bg: 'bg-orange-500/20 border-orange-500/40 text-orange-300' },
  SafetyCar:   { color: '#f97316', label: 'SAFETY CAR', bg: 'bg-orange-500/20 border-orange-500/40 text-orange-300' },
  RedFlag:     { color: '#ef4444', label: 'RED FLAG', bg: 'bg-red-500/20 border-red-500/40 text-red-400' },
  VSCDeployed: { color: '#a78bfa', label: 'VSC', bg: 'bg-purple-500/20 border-purple-500/40 text-purple-300' },
  VSCEnding:   { color: '#c4b5fd', label: 'VSC ENDING', bg: 'bg-purple-400/20 border-purple-400/40 text-purple-200' },
};

export default function TrackMap({ trackData, sessionData, onTimeUpdate }: TrackMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const playingRef = useRef(false);
  const timeRef = useRef(0);
  const lastTimestampRef = useRef<number | null>(null);
  const speedRef = useRef(1);
  const zoomRef = useRef(1);
  const panRef = useRef({ x: 0, y: 0 });
  const isPanning = useRef(false);
  const lastPan = useRef({ x: 0, y: 0 });
  const dprRef = useRef(1);
  const lastReportedTime = useRef<number>(-1);

  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [currentTime, setCurrentTime] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [currentLap, setCurrentLap] = useState(1);
  const [currentFlag, setCurrentFlag] = useState<TrackStatusEvent | null>(null);

  const maxDuration = sessionData.maxDuration;
  const { track, corners, bounds } = trackData;
  const drivers = sessionData.leaderboard;
  const telemetry = sessionData.telemetry;
  const trackStatusEvents = sessionData.trackStatusEvents ?? [];

  // Pre-build sorted telemetry arrays once — prefer X/Y for smooth positioning
  const driverTelArrays = React.useMemo(() => {
    const result: Record<string, {
      distances: Float64Array;
      trels: Float64Array;
      xs: Float64Array | null;
      ys: Float64Array | null;
      maxTrel: number;
      hasXY: boolean;
    }> = {};
    for (const drv of Object.keys(telemetry)) {
      const t = telemetry[drv];
      const hasXY = Array.isArray(t.X) && t.X.length > 0 && Array.isArray(t.Y) && t.Y.length > 0;
      result[drv] = {
        distances: new Float64Array(t.Distance),
        trels: new Float64Array(t.Trel),
        xs: hasXY ? new Float64Array(t.X!) : null,
        ys: hasXY ? new Float64Array(t.Y!) : null,
        maxTrel: typeof t.maxTrel === 'number' ? t.maxTrel : Infinity,
        hasXY,
      };
    }
    return result;
  }, [telemetry]);

  // Pre-build track arrays
  const trackArrays = React.useMemo(() => ({
    x: new Float64Array(track.x),
    y: new Float64Array(track.y),
    distance: new Float64Array(track.distance),
    maxDist: Math.max(...track.distance),
  }), [track]);

  // Fast binary search interpolation
  const interp = useCallback((yArr: Float64Array, xArr: Float64Array, x: number): number => {
    const n = xArr.length;
    if (!n) return 0;
    if (x <= xArr[0]) return yArr[0];
    if (x >= xArr[n - 1]) return yArr[n - 1];
    let lo = 0, hi = n - 1;
    while (lo < hi - 1) { const mid = (lo + hi) >> 1; if (xArr[mid] <= x) lo = mid; else hi = mid; }
    const frac = (x - xArr[lo]) / (xArr[hi] - xArr[lo]);
    return yArr[lo] + frac * (yArr[hi] - yArr[lo]);
  }, []);

  // Map raw track coords → canvas pixels
  const toCanvas = useCallback((W: number, H: number, tx: number, ty: number) => {
    const sc = zoomRef.current;
    const { x: px, y: py } = panRef.current;
    const tw = bounds.xMax - bounds.xMin;
    const th = bounds.yMax - bounds.yMin;
    const nx = (tx - bounds.xMin) / tw;
    const ny = 1 - (ty - bounds.yMin) / th;
    return {
      x: (nx * W * 0.88 + W * 0.06) * sc + px,
      y: (ny * H * 0.88 + H * 0.06) * sc + py,
    };
  }, [bounds]);

  // Get current track status at a given time
  const getStatusAt = useCallback((t: number): TrackStatusEvent | null => {
    let current: TrackStatusEvent | null = null;
    for (const ev of trackStatusEvents) {
      if (ev.trel <= t) current = ev;
      else break;
    }
    return current;
  }, [trackStatusEvents]);

  // HiDPI canvas resize
  const resizeCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const dpr = window.devicePixelRatio || 1;
    dprRef.current = dpr;
    const rect = container.getBoundingClientRect();
    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
    const ctx = canvas.getContext('2d');
    if (ctx) ctx.scale(dpr, dpr);
  }, []);

  // DRAW
  const draw = useCallback((t: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const dpr = dprRef.current;
    const W = canvas.width / dpr;
    const H = canvas.height / dpr;

    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Background
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, W, H);

    const map = (tx: number, ty: number) => toCanvas(W, H, tx, ty);

    // Current flag status
    const flagStatus = getStatusAt(t);
    const isSC = flagStatus?.status === 'SafetyCar' || flagStatus?.status === 'SCDeploying';
    const isVSC = flagStatus?.status === 'VSCDeployed' || flagStatus?.status === 'VSCEnding';
    const isRed = flagStatus?.status === 'RedFlag';
    const isYellow = flagStatus?.status === 'Yellow';

    // Track glow color based on flag
    const trackGlow = isRed ? '#ef444430' : isSC || isVSC ? '#f9731630' : isYellow ? '#facc1520' : null;

    // ── Track surface ──────────────────────────────────────────
    if (track.x.length > 1) {
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      const p0 = map(track.x[0], track.y[0]);

      // Optional flag glow ring
      if (trackGlow) {
        ctx.beginPath();
        ctx.moveTo(p0.x, p0.y);
        for (let i = 1; i < track.x.length; i++) {
          const p = map(track.x[i], track.y[i]);
          ctx.lineTo(p.x, p.y);
        }
        ctx.closePath();
        ctx.strokeStyle = trackGlow;
        ctx.lineWidth = 32;
        ctx.stroke();
      }

      // Outer band
      ctx.beginPath();
      ctx.moveTo(p0.x, p0.y);
      for (let i = 1; i < track.x.length; i++) {
        const p = map(track.x[i], track.y[i]);
        ctx.lineTo(p.x, p.y);
      }
      ctx.closePath();
      ctx.strokeStyle = '#1c1c1c';
      ctx.lineWidth = 22;
      ctx.stroke();

      // Mid band (asphalt)
      ctx.beginPath();
      ctx.moveTo(p0.x, p0.y);
      for (let i = 1; i < track.x.length; i++) {
        const p = map(track.x[i], track.y[i]);
        ctx.lineTo(p.x, p.y);
      }
      ctx.closePath();
      ctx.strokeStyle = isSC ? '#2a2000' : isVSC ? '#1a0a2a' : isRed ? '#2a0000' : '#2e2e2e';
      ctx.lineWidth = 16;
      ctx.stroke();

      // Inner surface
      ctx.beginPath();
      ctx.moveTo(p0.x, p0.y);
      for (let i = 1; i < track.x.length; i++) {
        const p = map(track.x[i], track.y[i]);
        ctx.lineTo(p.x, p.y);
      }
      ctx.closePath();
      ctx.strokeStyle = '#3c3c3c';
      ctx.lineWidth = 10;
      ctx.stroke();

      // Centre line dashes
      ctx.save();
      ctx.setLineDash([8, 10]);
      ctx.beginPath();
      ctx.moveTo(p0.x, p0.y);
      for (let i = 1; i < track.x.length; i++) {
        const p = map(track.x[i], track.y[i]);
        ctx.lineTo(p.x, p.y);
      }
      ctx.closePath();
      ctx.strokeStyle = 'rgba(255,255,255,0.18)';
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.restore();
    }

    // ── Corner labels ────────────────────────────────────────
    for (const corner of corners) {
      const lp = map(corner.label_x, corner.label_y);
      ctx.font = `bold ${Math.round(9 * zoomRef.current)}px Inter, sans-serif`;
      ctx.fillStyle = 'rgba(255,255,255,0.55)';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`${corner.Number}${corner.Letter}`, lp.x, lp.y);
    }

    // ── Driver dots ──────────────────────────────────────────
    for (const row of drivers) {
      const drv = row.driver;
      const tel = driverTelArrays[drv];
      if (!tel) continue;

      // If past the driver's last active trel, freeze at last position
      const clampedT = Math.min(t, tel.maxTrel);
      const isGhosted = t > tel.maxTrel + 5; // 5s grace before fading

      let posX: number, posY: number;

      if (tel.hasXY && tel.xs && tel.ys) {
        // Use raw X/Y for precise, smooth positioning
        posX = interp(tel.xs, tel.trels, clampedT);
        posY = interp(tel.ys, tel.trels, clampedT);
      } else {
        // Fallback: distance-based positioning on track outline
        const dist = interp(tel.distances, tel.trels, clampedT);
        const maxDist = trackArrays.maxDist || 1;
        const normDist = (dist % maxDist) / maxDist;
        const len = trackArrays.x.length;
        const rawIdx = normDist * (len - 1);
        const idx = Math.min(Math.floor(rawIdx), len - 2);
        const frac = rawIdx - idx;
        posX = trackArrays.x[idx] + frac * (trackArrays.x[idx + 1] - trackArrays.x[idx]);
        posY = trackArrays.y[idx] + frac * (trackArrays.y[idx + 1] - trackArrays.y[idx]);
      }

      const pos = map(posX, posY);
      const color = row.color || '#888888';
      const r = Math.max(6, 7 * zoomRef.current);
      const alpha = isGhosted ? 0.35 : 1.0;

      ctx.globalAlpha = alpha;

      // Premium glow effect
      ctx.shadowColor = isGhosted ? 'transparent' : color;
      ctx.shadowBlur = isGhosted ? 0 : 10 * zoomRef.current;

      // Dot body
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
      ctx.fillStyle = isGhosted ? '#555' : color;
      ctx.fill();

      // Reset shadow before drawing the border or text
      ctx.shadowBlur = 0;

      // Subtle dark inner border to separate overlapping dots cleanly
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(0,0,0,0.6)';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Driver label
      if (zoomRef.current > 0.7) {
        ctx.font = `bold ${Math.round(8 * zoomRef.current)}px Inter, sans-serif`;
        ctx.fillStyle = isGhosted ? 'rgba(255,255,255,0.3)' : '#ffffff';
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 2.5;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.strokeText(drv, pos.x, pos.y - r - 2);
        ctx.fillText(drv, pos.x, pos.y - r - 2);
      }

      ctx.globalAlpha = 1.0;
    }

    ctx.restore();
  }, [track, corners, drivers, driverTelArrays, interp, toCanvas, trackArrays, getStatusAt]);

  // ── Animation loop ─────────────────────────────────────────
  const animate = useCallback((ts: number) => {
    if (!playingRef.current) return;
    if (lastTimestampRef.current === null) lastTimestampRef.current = ts;
    const dt = (ts - lastTimestampRef.current) / 1000;
    lastTimestampRef.current = ts;
    timeRef.current = Math.min(timeRef.current + dt * speedRef.current, maxDuration);

    // Compute current lap from leader's lap history
    if (sessionData.isRace && sessionData.lapHistory) {
      const leader = sessionData.leaderboard[0]?.driver;
      if (leader && sessionData.lapHistory[leader]) {
        const history = sessionData.lapHistory[leader];
        let lap = 1;
        for (const h of history) {
          if (h.trel <= timeRef.current) lap = h.lap;
          else break;
        }
        setCurrentLap(lap);
      }
    }

    // Current flag
    setCurrentFlag(getStatusAt(timeRef.current));

    if (onTimeUpdate && Math.abs(timeRef.current - lastReportedTime.current) > 0.5) {
      onTimeUpdate(timeRef.current);
      lastReportedTime.current = timeRef.current;
    }

    setCurrentTime(timeRef.current);
    draw(timeRef.current);
    if (timeRef.current < maxDuration) animRef.current = requestAnimationFrame(animate);
    else { setPlaying(false); playingRef.current = false; }
  }, [draw, maxDuration, sessionData, getStatusAt, onTimeUpdate]);

  // Resize + redraw on mount
  useEffect(() => {
    resizeCanvas();
    draw(timeRef.current);
    const ro = new ResizeObserver(() => { resizeCanvas(); draw(timeRef.current); });
    if (containerRef.current) ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, [resizeCanvas, draw]);

  useEffect(() => { draw(timeRef.current); }, [draw, zoom]);

  const togglePlay = () => {
    if (playing) {
      playingRef.current = false;
      lastTimestampRef.current = null;
      cancelAnimationFrame(animRef.current);
      setPlaying(false);
    } else {
      if (timeRef.current >= maxDuration) timeRef.current = 0;
      playingRef.current = true;
      setPlaying(true);
      lastTimestampRef.current = null;
      animRef.current = requestAnimationFrame(animate);
    }
  };

  const handleSpeedChange = (s: number) => { speedRef.current = s; setSpeed(s); };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const t = parseFloat(e.target.value);
    timeRef.current = t;
    setCurrentTime(t);

    // Update lap on seek
    if (sessionData.isRace && sessionData.lapHistory) {
      const leader = sessionData.leaderboard[0]?.driver;
      if (leader && sessionData.lapHistory[leader]) {
        const history = sessionData.lapHistory[leader];
        let lap = 1;
        for (const h of history) {
          if (h.trel <= t) lap = h.lap;
          else break;
        }
        setCurrentLap(lap);
      }
    }

    setCurrentFlag(getStatusAt(t));

    if (onTimeUpdate) {
      onTimeUpdate(t);
      lastReportedTime.current = t;
    }
    draw(t);
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const newZoom = Math.max(0.3, Math.min(6, zoomRef.current * (e.deltaY > 0 ? 0.9 : 1.1)));
    zoomRef.current = newZoom;
    setZoom(newZoom);
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    isPanning.current = true;
    lastPan.current = { x: e.clientX - panRef.current.x, y: e.clientY - panRef.current.y };
  };
  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isPanning.current) return;
    panRef.current = { x: e.clientX - lastPan.current.x, y: e.clientY - lastPan.current.y };
    draw(timeRef.current);
  };
  const handleMouseUp = () => { isPanning.current = false; };
  const resetView = () => { zoomRef.current = 1; panRef.current = { x: 0, y: 0 }; setZoom(1); };

  const fmt = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = (s % 60).toFixed(1);
    return `${m}:${sec.padStart(4, '0')}`;
  };

  const flagStyle = currentFlag ? STATUS_STYLE[currentFlag.status] : null;
  const showFlag = flagStyle && currentFlag?.status !== 'AllClear';

  return (
    <div className="flex flex-col gap-3 h-full">
      {/* Canvas area */}
      <div
        ref={containerRef}
        className="relative flex-1 min-h-[480px] rounded-xl overflow-hidden border border-white/10 bg-black cursor-grab active:cursor-grabbing select-none"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <canvas ref={canvasRef} className="absolute inset-0" />

        {/* Lap Counter */}
        {sessionData.isRace && (
          <div className="absolute top-3 left-3 bg-black/70 border border-white/15 px-3 py-1.5 rounded-lg flex flex-col items-center shadow-lg backdrop-blur z-10">
            <span className="text-[9px] uppercase tracking-widest text-white/40 font-bold mb-0.5">Lap</span>
            <span className="font-mono font-black text-white text-lg leading-none">
              {currentLap} <span className="text-white/30 text-sm font-semibold">/ {sessionData.totalLaps || '-'}</span>
            </span>
          </div>
        )}

        {/* Flag / Track Status Banner */}
        {showFlag && flagStyle && (
          <div className={`absolute top-3 left-1/2 -translate-x-1/2 flex items-center gap-2 px-4 py-1.5 rounded-full border backdrop-blur z-20 shadow-lg ${flagStyle.bg}`}>
            <span className="w-2 h-2 rounded-full animate-pulse" style={{ background: flagStyle.color }} />
            <span className="text-[11px] font-black tracking-widest uppercase">{flagStyle.label}</span>
          </div>
        )}

        {/* Zoom controls */}
        <div className="absolute top-3 right-3 flex flex-col gap-1.5 z-10">
          {[
            { label: '+', fn: () => { zoomRef.current = Math.min(6, zoomRef.current * 1.25); setZoom(zoomRef.current); } },
            { label: '−', fn: () => { zoomRef.current = Math.max(0.3, zoomRef.current / 1.25); setZoom(zoomRef.current); } },
            { label: '⊙', fn: resetView },
          ].map(({ label, fn }) => (
            <button key={label} onClick={fn}
              className="w-8 h-8 rounded-lg bg-black/60 border border-white/15 hover:bg-white/10 text-white font-bold flex items-center justify-center text-sm transition">
              {label}
            </button>
          ))}
        </div>

        {/* Zoom level badge */}
        <div className="absolute bottom-3 right-3 bg-black/50 text-white/40 text-[10px] font-mono px-2 py-1 rounded">
          {Math.round(zoom * 100)}%
        </div>
      </div>

      {/* Playback controls */}
      <div className="bg-white/[0.04] rounded-xl border border-white/10 px-4 py-3 flex flex-col gap-2.5">
        {/* Scrubber */}
        <div className="flex items-center gap-3">
          <span className="text-white/40 text-[11px] font-mono w-12">{fmt(currentTime)}</span>
          <div className="relative flex-1">
            <input type="range" min={0} max={maxDuration} step={0.1}
              value={currentTime} onChange={handleSeek}
              className="w-full cursor-pointer" />
            {/* Flag markers on scrubber */}
            {trackStatusEvents.filter(e => e.status !== 'AllClear').map((ev, i) => {
              const pct = (ev.trel / maxDuration) * 100;
              const style = STATUS_STYLE[ev.status];
              return (
                <div key={i}
                  className="absolute top-1/2 -translate-y-1/2 w-0.5 h-3 rounded-full opacity-70 pointer-events-none"
                  style={{ left: `${pct}%`, background: style?.color ?? '#fff' }}
                  title={ev.status}
                />
              );
            })}
          </div>
          <span className="text-white/40 text-[11px] font-mono w-12 text-right">{fmt(maxDuration)}</span>
        </div>

        {/* Buttons */}
        <div className="flex items-center gap-2">
          <button onClick={togglePlay}
            className={`px-5 py-1.5 rounded-lg font-bold text-sm transition-all ${playing ? 'bg-red-600 hover:bg-red-700' : 'bg-red-500 hover:bg-red-600'} text-white`}>
            {playing ? '⏸ Pause' : '▶ Play'}
          </button>
          <button onClick={() => { timeRef.current = 0; setCurrentTime(0); setCurrentFlag(null); draw(0); }}
            className="px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-white text-sm transition">⏮</button>

          {/* Flag indicator next to buttons */}
          {showFlag && flagStyle && (
            <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-[10px] font-bold ${flagStyle.bg}`}>
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: flagStyle.color }} />
              {flagStyle.label}
            </div>
          )}

          <div className="ml-auto flex items-center gap-1">
            <span className="text-white/30 text-[10px] mr-1">SPEED</span>
            {[1, 2, 4, 8, 16].map(s => (
              <button key={s} onClick={() => handleSpeedChange(s)}
                className={`w-9 py-1.5 rounded-lg text-xs font-bold transition ${speed === s ? 'bg-red-500 text-white' : 'bg-white/8 text-white/60 hover:bg-white/15'}`}>
                {s}×
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
