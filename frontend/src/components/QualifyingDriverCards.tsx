'use client';

import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { PracticeDriverInfo, QualLap, SessionData } from '@/lib/api';

interface QualifyingDriverCardsProps {
  data: Record<string, PracticeDriverInfo>;
  sessionData: SessionData;
  currentTime: number;
  sessionName?: string;
  onFocusDriver?: (driver: string | null, startTrel: number) => void;
}

const CMP: Record<string, { bg: string; fg: string; label: string }> = {
  SOFT:         { bg: '#E8002D', fg: '#fff', label: 'Soft' },
  MEDIUM:       { bg: '#FFF200', fg: '#000', label: 'Medium' },
  HARD:         { bg: '#EDEDED', fg: '#000', label: 'Hard' },
  INTERMEDIATE: { bg: '#39B54A', fg: '#fff', label: 'Inter' },
  WET:          { bg: '#0067FF', fg: '#fff', label: 'Wet' },
  UNKNOWN:      { bg: '#444',    fg: '#aaa', label: '?' },
};

const LAP_TYPE_CONFIG = {
  hot:     { label: 'FLYING', bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500/40', dot: '#ef4444' },
  out:     { label: 'OUT LAP', bg: 'bg-blue-500/10', text: 'text-blue-400', border: 'border-blue-500/30', dot: '#60a5fa' },
  in:      { label: 'IN LAP', bg: 'bg-orange-500/10', text: 'text-orange-400', border: 'border-orange-500/30', dot: '#fb923c' },
  prep:    { label: 'PREP', bg: 'bg-white/5', text: 'text-white/40', border: 'border-white/10', dot: '#888' },
  unknown: { label: '—', bg: 'bg-white/5', text: 'text-white/20', border: 'border-white/10', dot: '#555' },
};

function TyreCircle({ compound, size = 24 }: { compound: string; size?: number }) {
  const c = CMP[compound] ?? CMP.UNKNOWN;
  return (
    <svg width={size} height={size} viewBox="0 0 28 28">
      <circle cx="14" cy="14" r="13" fill="#111" />
      <circle cx="14" cy="14" r="11" fill={c.bg} />
      <circle cx="14" cy="14" r="5.5" fill="#111" />
    </svg>
  );
}

function LapTypeBadge({ lapType }: { lapType: string }) {
  const cfg = LAP_TYPE_CONFIG[lapType as keyof typeof LAP_TYPE_CONFIG] ?? LAP_TYPE_CONFIG.unknown;
  return (
    <span className={`text-[8px] font-black tracking-widest uppercase px-1.5 py-0.5 rounded border ${cfg.bg} ${cfg.text} ${cfg.border}`}>
      {cfg.label}
    </span>
  );
}

type DynamicQualInfo = PracticeDriverInfo & {
  currentLapType: string;
  currentLapEntry: QualLap | null;
  dynamicBestStr: string;
  dynamicBestTime: number | null;
};

interface CompactQualCardProps {
  info: DynamicQualInfo;
  rank: number;
  onClick: () => void;
  onFollow: () => void;
}

function CompactQualCard({ info, rank, onClick, onFollow }: CompactQualCardProps) {
  const cfg = LAP_TYPE_CONFIG[info.currentLapType as keyof typeof LAP_TYPE_CONFIG] ?? LAP_TYPE_CONFIG.unknown;
  const cmp = CMP[info.currentLapEntry?.compound || info.currentCompound] ?? CMP.UNKNOWN;
  const isFlying = info.currentLapType === 'hot';

  return (
    <motion.div
      layout
      whileHover={{ scale: 1.02, y: -2 }}
      whileTap={{ scale: 0.98 }}
      className={`rounded-xl border transition-all overflow-hidden select-none relative cursor-pointer
        ${isFlying
          ? 'bg-red-500/[0.06] border-red-500/25 shadow-[0_0_16px_rgba(239,68,68,0.12)]'
          : 'bg-white/[0.04] border-white/10 hover:bg-white/[0.07] hover:border-white/20'
        }`}
      style={{ borderTopColor: info.color, borderTopWidth: 3 }}
      onClick={onClick}
    >
      {/* Flying lap pulse ring */}
      {isFlying && (
        <div className="absolute inset-0 rounded-xl pointer-events-none animate-pulse"
          style={{ boxShadow: `inset 0 0 20px ${info.color}20` }} />
      )}

      {/* Header */}
      <div className="relative h-16 overflow-hidden flex items-end px-3 pb-2 bg-gradient-to-br from-white/5 to-transparent">
        <div className="absolute inset-0 opacity-20"
          style={{ background: `linear-gradient(135deg, ${info.color}80 0%, transparent 70%)` }} />
        <span
          className="absolute left-1.5 top-0 text-[3.5rem] font-black leading-none select-none pointer-events-none"
          style={{ color: info.color, opacity: 0.12 }}>
          {info.number}
        </span>
        {info.headshotUrl && (
          <img src={info.headshotUrl} alt={info.driver}
            className="absolute -right-2 top-0 w-24 h-[120%] object-cover object-top opacity-80"
            style={{ maskImage: 'linear-gradient(to top, rgba(0,0,0,1) 15%, rgba(0,0,0,0) 100%)', WebkitMaskImage: 'linear-gradient(to right, rgba(0,0,0,0) 0%, rgba(0,0,0,1) 30%, rgba(0,0,0,1) 100%)' }}
          />
        )}
        <div className="relative z-10 drop-shadow-md flex-1">
          <div className="flex items-center gap-1.5">
            <span className="text-white/40 text-[10px] font-bold w-5 text-right">{rank}.</span>
            <p className="font-black text-white text-base leading-none tracking-wide">{info.driver}</p>
            <LapTypeBadge lapType={info.currentLapType} />
          </div>
          <p className="text-[10px] text-white/50 mt-0.5 truncate max-w-[100px] ml-7">{info.team}</p>
        </div>
      </div>

      {/* Body */}
      <div className="px-3 pb-3 pt-2 flex flex-col gap-1.5 relative z-10">
        <div className="flex items-center gap-2">
          <TyreCircle compound={info.currentLapEntry?.compound || info.currentCompound} size={20} />
          <div className="flex flex-col leading-none">
            <span className="text-[9px] font-bold" style={{ color: cmp.bg }}>{cmp.label}</span>
            <span className="text-[8px] text-white/30 mt-0.5">{info.currentLapEntry?.tyreLife ?? 0}L on tyre</span>
          </div>
          <div className="ml-auto text-right">
            <p className="font-mono text-[11px] font-bold text-purple-400">{info.dynamicBestStr}</p>
            <p className="text-[8px] text-white/30">best</p>
          </div>
        </div>

        {/* Sector times */}
        {info.currentLapEntry && (
          <div className="flex gap-1 mt-0.5">
            {['S1', 'S2', 'S3'].map(s => (
              <span key={s} className="flex-1 text-center font-mono text-[8px] bg-white/5 rounded px-1 py-0.5 text-white/50">
                {info.currentLapEntry?.[s as 'S1' | 'S2' | 'S3'] || '—'}
              </span>
            ))}
          </div>
        )}

        {/* Follow fastest lap button */}
        {info.bestLapStartTrel != null && (
          <button
            onClick={e => { e.stopPropagation(); onFollow(); }}
            className="mt-1 w-full text-[9px] font-black uppercase tracking-widest py-1 rounded bg-white/8 hover:bg-red-500/20 hover:text-red-400 text-white/40 transition-all border border-white/8 hover:border-red-500/30"
          >
            ▶ Follow Fastest Lap
          </button>
        )}
      </div>
    </motion.div>
  );
}

function QualDetailModal({ info, onClose, onFollow }: { info: DynamicQualInfo; onClose: () => void; onFollow: () => void }) {
  const [activeTab, setActiveTab] = useState<'laps' | 'stints'>('laps');

  return (
    <motion.div className="fixed inset-0 z-50 flex items-center justify-center p-4"
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      onClick={onClose}>
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" />
      <motion.div
        className="relative z-10 bg-[#0e0e1a] border border-white/15 rounded-2xl w-full max-w-lg shadow-2xl overflow-hidden"
        initial={{ scale: 0.9, y: 20 }} animate={{ scale: 1, y: 0 }} exit={{ scale: 0.9, y: 20 }}
        onClick={e => e.stopPropagation()}
        style={{ borderTopColor: info.color, borderTopWidth: 4 }}>

        {/* Header */}
        <div className="relative px-6 pt-5 pb-4 overflow-hidden"
          style={{ background: `linear-gradient(135deg, ${info.color}18 0%, transparent 60%)` }}>
          <span className="absolute right-4 top-2 text-7xl font-black opacity-10 select-none pointer-events-none leading-none"
            style={{ color: info.color }}>{info.number}</span>
          <div className="flex items-start gap-4">
            <div className="w-16 h-20 rounded-xl border border-white/10 bg-white/5 flex items-end justify-center overflow-hidden shrink-0">
              {info.headshotUrl
                ? <img src={info.headshotUrl} alt={info.driver} className="w-full h-full object-cover object-top scale-110 translate-y-2" />
                : <span className="text-white/20 text-xs font-bold mb-4">IMG</span>}
            </div>
            <div>
              <h2 className="font-black text-xl text-white flex items-center gap-2">
                {info.driver}
                <LapTypeBadge lapType={info.currentLapType} />
              </h2>
              <p className="text-white/50 text-sm">{info.team}</p>
              <div className="flex gap-2 mt-2 flex-wrap">
                {[
                  { label: 'Best', value: info.dynamicBestStr, color: 'text-purple-400' },
                  { label: 'Laps', value: String(info.totalLaps), color: 'text-white' },
                ].map(s => (
                  <div key={s.label} className="bg-white/5 rounded-lg px-3 py-1.5 text-center">
                    <p className={`font-mono text-xs font-bold ${s.color}`}>{s.value}</p>
                    <p className="text-[9px] text-white/30">{s.label}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
          {info.bestLapStartTrel != null && (
            <button onClick={() => { onClose(); onFollow(); }}
              className="mt-3 px-4 py-1.5 rounded-lg bg-red-500/20 border border-red-500/40 text-red-400 text-[11px] font-black tracking-widest uppercase hover:bg-red-500/30 transition w-full">
              ▶ Follow Fastest Lap on Track
            </button>
          )}
        </div>

        {/* Tabs */}
        <div className="flex border-b border-white/10">
          {(['laps', 'stints'] as const).map(t => (
            <button key={t} onClick={() => setActiveTab(t)}
              className={`flex-1 py-2.5 text-[11px] font-bold uppercase tracking-widest transition ${activeTab === t ? 'text-white border-b-2 border-red-500' : 'text-white/30 hover:text-white/60'}`}>
              {t === 'laps' ? 'Lap History' : 'Stints'}
            </button>
          ))}
        </div>

        <div className="overflow-y-auto max-h-[280px] px-6 py-4">
          {activeTab === 'laps' && (
            <div className="flex flex-col gap-1.5">
              {(info.qualLaps ?? []).map((ql, i) => {
                const lt = LAP_TYPE_CONFIG[ql.lapType] ?? LAP_TYPE_CONFIG.unknown;
                const cmpCfg = CMP[ql.compound] ?? CMP.UNKNOWN;
                return (
                  <div key={i} className={`flex items-center gap-2 px-3 py-2 rounded-lg border ${lt.border} ${lt.bg}`}>
                    <span className="text-white/30 text-[10px] w-5">L{ql.lap}</span>
                    <LapTypeBadge lapType={ql.lapType} />
                    <TyreCircle compound={ql.compound} size={16} />
                    <span className="text-[9px] font-bold" style={{ color: cmpCfg.bg }}>{cmpCfg.label}</span>
                    <span className="text-[10px] text-white/30 ml-auto">
                      {ql.S1} · {ql.S2} · {ql.S3}
                    </span>
                    <span className={`font-mono text-[11px] font-bold ml-2 ${ql.lapType === 'hot' ? 'text-purple-300' : 'text-white/50'}`}>
                      {ql.lapTimeStr}
                    </span>
                  </div>
                );
              })}
            </div>
          )}

          {activeTab === 'stints' && (
            <div className="flex flex-col gap-2">
              {info.stints.map((s, i) => {
                const c = CMP[s.compound] ?? CMP.UNKNOWN;
                return (
                  <div key={i} className="flex items-center gap-3">
                    <TyreCircle compound={s.compound} size={22} />
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-bold" style={{ color: c.bg }}>{c.label}</span>
                        {!s.fresh && <span className="text-[9px] bg-white/10 text-white/40 px-1.5 rounded-full">Used</span>}
                        <span className="text-[10px] text-white/40 ml-auto">L{s.startLap}–{s.endLap}</span>
                      </div>
                      <div className="mt-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
                        <div className="h-full rounded-full" style={{
                          width: `${(s.laps / Math.max(info.totalLaps, 1)) * 100}%`,
                          background: c.bg, opacity: s.fresh ? 1 : 0.5
                        }} />
                      </div>
                    </div>
                    <span className="font-mono text-[11px] text-white/60 w-14 text-right">{s.laps}L</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <button onClick={onClose}
          className="absolute top-4 right-4 w-7 h-7 rounded-full bg-white/10 hover:bg-white/20 flex items-center justify-center text-white/60 hover:text-white transition text-sm z-50">
          ✕
        </button>
      </motion.div>
    </motion.div>
  );
}

export default function QualifyingDriverCards({ data, sessionData, currentTime, sessionName, onFocusDriver }: QualifyingDriverCardsProps) {
  const [selected, setSelected] = useState<string | null>(null);

  const dynamicDrivers: DynamicQualInfo[] = useMemo(() => {
    return Object.values(data).map(info => {
      // Find current lap entry based on time
      const qualLaps = info.qualLaps ?? [];
      let currentLapEntry: QualLap | null = null;
      for (const ql of qualLaps) {
        const s = ql.lapStartTrel ?? 0;
        const e = ql.lapEndTrel ?? Infinity;
        if (s <= currentTime && currentTime <= e) {
          currentLapEntry = ql;
          break;
        }
      }

      // Best lap up to currentTime
      let dynamicBestTime: number | null = null;
      let dynamicBestStr = info.bestLapStr;
      for (const ql of qualLaps) {
        const e = ql.lapEndTrel;
        if (e !== null && e <= currentTime && ql.lapType === 'hot' && ql.lapTime !== null) {
          if (dynamicBestTime === null || ql.lapTime < dynamicBestTime) {
            dynamicBestTime = ql.lapTime;
            dynamicBestStr = ql.lapTimeStr;
          }
        }
      }

      return {
        ...info,
        currentLapType: currentLapEntry?.lapType ?? 'unknown',
        currentLapEntry,
        dynamicBestStr: dynamicBestStr || 'N/A',
        dynamicBestTime,
      };
    }).sort((a, b) => {
      // Flying laps first, then by best time
      if (a.currentLapType === 'hot' && b.currentLapType !== 'hot') return -1;
      if (b.currentLapType === 'hot' && a.currentLapType !== 'hot') return 1;
      const ta = a.dynamicBestTime ?? Infinity;
      const tb = b.dynamicBestTime ?? Infinity;
      return ta - tb;
    });
  }, [data, currentTime]);

  const selectedDriver = selected ? dynamicDrivers.find(d => d.driver === selected) : null;

  const handleFollow = (info: DynamicQualInfo) => {
    if (info.bestLapStartTrel != null && onFocusDriver) {
      onFocusDriver(info.driver, info.bestLapStartTrel);
    }
  };

  // Count flying laps
  const flyingCount = dynamicDrivers.filter(d => d.currentLapType === 'hot').length;

  return (
    <div className="mt-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-white font-bold text-sm uppercase tracking-widest flex items-center gap-2">
          <span className="w-1.5 h-4 bg-red-500 rounded-full" />
          Qualifying Status
          {sessionName && <span className="text-white/30 font-normal normal-case tracking-normal ml-1">· {sessionName}</span>}
        </h3>
        <div className="flex items-center gap-3">
          {flyingCount > 0 && (
            <span className="flex items-center gap-1.5 text-[10px] font-bold text-red-400 animate-pulse">
              <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
              {flyingCount} on flying lap
            </span>
          )}
          <span className="text-[10px] text-white/30">{dynamicDrivers.length} drivers · click for details</span>
        </div>
      </div>

      {/* Legend */}
      <div className="flex gap-3 mb-3 flex-wrap">
        {Object.entries(LAP_TYPE_CONFIG).filter(([k]) => k !== 'unknown').map(([type, cfg]) => (
          <span key={type} className={`flex items-center gap-1.5 text-[9px] font-bold uppercase tracking-wider ${cfg.text}`}>
            <span className="w-2 h-2 rounded-full" style={{ background: cfg.dot }} />
            {cfg.label}
          </span>
        ))}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-2.5">
        {dynamicDrivers.map((info, idx) => (
          <CompactQualCard
            key={info.driver}
            info={info}
            rank={idx + 1}
            onClick={() => setSelected(info.driver)}
            onFollow={() => handleFollow(info)}
          />
        ))}
      </div>

      <AnimatePresence>
        {selectedDriver && (
          <QualDetailModal
            info={selectedDriver}
            onClose={() => setSelected(null)}
            onFollow={() => { handleFollow(selectedDriver); setSelected(null); }}
          />
        )}
      </AnimatePresence>
    </div>
  );
}
