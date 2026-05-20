'use client';

import React, { useEffect, useRef, useCallback, useState } from 'react';
import { TrackData, SessionData } from '@/lib/api';

interface TrackMapProps {
  trackData: TrackData;
  sessionData: SessionData;
}

export default function TrackMap({ trackData, sessionData }: TrackMapProps) {
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

  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [currentTime, setCurrentTime] = useState(0);
  const [zoom, setZoom] = useState(1);

  const maxDuration = sessionData.maxDuration;
  const { track, corners, bounds } = trackData;
  const drivers = sessionData.leaderboard;
  const telemetry = sessionData.telemetry;

  // Pre-build sorted telemetry arrays once
  const driverTelArrays = React.useMemo(() => {
    const result: Record<string, { distances: Float64Array; trels: Float64Array }> = {};
    for (const drv of Object.keys(telemetry)) {
      result[drv] = {
        distances: new Float64Array(telemetry[drv].Distance),
        trels: new Float64Array(telemetry[drv].Trel),
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
    const t = (x - xArr[lo]) / (xArr[hi] - xArr[lo]);
    return yArr[lo] + t * (yArr[hi] - yArr[lo]);
  }, []);

  // Map normalised track coords → canvas pixels (accounts for DPR)
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

  // ── HiDPI canvas resize ────────────────────────────────────────
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

  // ── DRAW ──────────────────────────────────────────────────────
  const draw = useCallback((t: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const dpr = dprRef.current;
    const W = canvas.width / dpr;
    const W2 = canvas.height / dpr;

    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // reset to logical pixels

    // Background
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, W, W2);

    const map = (tx: number, ty: number) => toCanvas(W, W2, tx, ty);

    // ── Track surface (wide dark gray filled band) ──────────────
    if (track.x.length > 1) {
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      // Outer band (dark base)
      ctx.beginPath();
      const p0 = map(track.x[0], track.y[0]);
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
      ctx.strokeStyle = '#2e2e2e';
      ctx.lineWidth = 16;
      ctx.stroke();

      // Inner surface — slightly lighter
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

      // White centre line dashes
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

    // ── Corner labels ───────────────────────────────────────────
    for (const corner of corners) {
      const lp = map(corner.label_x, corner.label_y);
      ctx.font = `bold ${Math.round(9 * zoomRef.current)}px Inter, sans-serif`;
      ctx.fillStyle = 'rgba(255,255,255,0.55)';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`${corner.Number}${corner.Letter}`, lp.x, lp.y);
    }

    // ── Driver dots ─────────────────────────────────────────────
    for (const row of drivers) {
      const drv = row.driver;
      const tel = driverTelArrays[drv];
      if (!tel) continue;

      // Get distance at time t by interpolating Trel → Distance
      const dist = interp(tel.distances, tel.trels, t);
      const maxDist = trackArrays.maxDist || 1;
      const normDist = dist / maxDist;
      const tArr = trackArrays;
      const len = tArr.x.length;
      const rawIdx = normDist * (len - 1);
      const idx = Math.min(Math.floor(rawIdx), len - 2);
      const frac = rawIdx - idx;
      const tx = tArr.x[idx] + frac * (tArr.x[idx + 1] - tArr.x[idx]);
      const ty = tArr.y[idx] + frac * (tArr.y[idx + 1] - tArr.y[idx]);

      const pos = map(tx, ty);
      const color = row.color || '#888888';
      const r = Math.max(5, 6 * zoomRef.current);

      // Solid dot — no shadow/blur
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, r + 1.5, 0, Math.PI * 2);
      ctx.fillStyle = '#000';
      ctx.fill();

      ctx.beginPath();
      ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255,255,255,0.9)';
      ctx.lineWidth = 1.2;
      ctx.stroke();

      // Driver label — only show when zoomed in enough
      if (zoomRef.current > 0.7) {
        ctx.font = `bold ${Math.round(8 * zoomRef.current)}px Inter, sans-serif`;
        ctx.fillStyle = '#ffffff';
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 2.5;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.strokeText(drv, pos.x, pos.y - r - 2);
        ctx.fillText(drv, pos.x, pos.y - r - 2);
      }
    }

    ctx.restore();
  }, [track, corners, drivers, driverTelArrays, interp, toCanvas, trackArrays]);

  // ── Animation loop ────────────────────────────────────────────
  const animate = useCallback((ts: number) => {
    if (!playingRef.current) return;
    if (lastTimestampRef.current === null) lastTimestampRef.current = ts;
    const dt = (ts - lastTimestampRef.current) / 1000;
    lastTimestampRef.current = ts;
    timeRef.current = Math.min(timeRef.current + dt * speedRef.current, maxDuration);
    setCurrentTime(timeRef.current);
    draw(timeRef.current);
    if (timeRef.current < maxDuration) animRef.current = requestAnimationFrame(animate);
    else { setPlaying(false); playingRef.current = false; }
  }, [draw, maxDuration]);

  // Resize + redraw on mount and window resize
  useEffect(() => {
    resizeCanvas();
    draw(timeRef.current);
    const ro = new ResizeObserver(() => { resizeCanvas(); draw(timeRef.current); });
    if (containerRef.current) ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, [resizeCanvas, draw]);

  // Redraw on zoom/pan changes (refs) — trigger via state
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
          <input type="range" min={0} max={maxDuration} step={0.05}
            value={currentTime} onChange={handleSeek}
            className="flex-1 cursor-pointer" />
          <span className="text-white/40 text-[11px] font-mono w-12 text-right">{fmt(maxDuration)}</span>
        </div>

        {/* Buttons */}
        <div className="flex items-center gap-2">
          <button onClick={togglePlay}
            className={`px-5 py-1.5 rounded-lg font-bold text-sm transition-all ${playing ? 'bg-red-600 hover:bg-red-700' : 'bg-red-500 hover:bg-red-600'} text-white`}>
            {playing ? '⏸ Pause' : '▶ Play'}
          </button>
          <button onClick={() => { timeRef.current = 0; setCurrentTime(0); draw(0); }}
            className="px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-white text-sm transition">⏮</button>
          <div className="ml-auto flex items-center gap-1">
            <span className="text-white/30 text-[10px] mr-1">SPEED</span>
            {[1, 2, 4, 8].map(s => (
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
