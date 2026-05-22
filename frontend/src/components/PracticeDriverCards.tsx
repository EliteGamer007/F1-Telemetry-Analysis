'use client';

import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { PracticeDriverInfo, PracticeStint, SessionData } from '@/lib/api';

interface PracticeDriverCardsProps {
  data: Record<string, PracticeDriverInfo>;
  sessionData: SessionData;
  currentTime: number;
  sessionName?: string;
}

// ── Compound visual config ──────────────────────────────────────
const CMP: Record<string, { bg: string; fg: string; letter: string; label: string; border: string }> = {
  SOFT:         { bg: '#E8002D', fg: '#fff', letter: 'S', label: 'Soft',  border: '#E8002D' },
  MEDIUM:       { bg: '#FFF200', fg: '#000', letter: 'M', label: 'Medium', border: '#FFF200' },
  HARD:         { bg: '#EDEDED', fg: '#000', letter: 'H', label: 'Hard',  border: '#EDEDED' },
  INTERMEDIATE: { bg: '#39B54A', fg: '#fff', letter: 'I', label: 'Inter', border: '#39B54A' },
  WET:          { bg: '#0067FF', fg: '#fff', letter: 'W', label: 'Wet',   border: '#0067FF' },
  UNKNOWN:      { bg: '#444',    fg: '#aaa', letter: '?', label: '?',     border: '#444' },
};

function TyreCircle({ compound, size = 28 }: { compound: string; size?: number }) {
  const c = CMP[compound] ?? CMP.UNKNOWN;
  return (
    <svg width={size} height={size} viewBox="0 0 28 28">
      <circle cx="14" cy="14" r="13" fill="#111" />
      <circle cx="14" cy="14" r="11" fill={c.bg} />
      <circle cx="14" cy="14" r="5.5" fill="#111" />
    </svg>
  );
}

function StintBar({ stints, totalLaps }: { stints: PracticeStint[]; totalLaps: number }) {
  if (!stints.length) return null;
  return (
    <div className="flex w-full h-2 rounded-full overflow-hidden gap-px mt-1 bg-white/5">
      {stints.map((s, i) => {
        const c = CMP[s.compound] ?? CMP.UNKNOWN;
        const pct = (s.laps / totalLaps) * 100;
        return (
          <div key={i} className="h-full rounded-sm flex-none relative group/tip transition-all"
            style={{ width: `${pct}%`, background: c.bg, opacity: s.fresh ? 1 : 0.55 }}
            title={`${s.compound} • ${s.laps} laps (L${s.startLap}–${s.endLap})${!s.fresh ? ' • Used' : ''}`}
          />
        );
      })}
    </div>
  );
}

// Combine static final info with dynamic real-time info
type DynamicDriverInfo = PracticeDriverInfo & {
  dynamicCompound: string;
  dynamicTyreLife: number;
  dynamicTotalLaps: number;
  dynamicLastLapStr: string;
  dynamicStints: PracticeStint[];
  isInPit: boolean;
};

function CompactCard({ info, onClick }: { info: DynamicDriverInfo; onClick: () => void }) {
  const cmp = CMP[info.dynamicCompound] ?? CMP.UNKNOWN;
  return (
    <motion.div
      layout
      whileHover={{ scale: 1.02, y: -2 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      className={`cursor-pointer rounded-xl border transition-colors overflow-hidden select-none relative
        ${info.isInPit ? 'bg-yellow-500/[0.05] border-yellow-500/20 hover:bg-yellow-500/10' : 'bg-white/[0.04] border-white/10 hover:bg-white/[0.07] hover:border-white/20'}`}
      style={{ borderTopColor: info.color, borderTopWidth: 3 }}
    >
      {/* Image + header */}
      <div className="relative h-16 bg-gradient-to-br from-white/5 to-transparent overflow-hidden flex items-end px-3 pb-2">
        <div
          className="absolute inset-0 opacity-20"
          style={{ background: `linear-gradient(135deg, ${info.color}80 0%, transparent 70%)` }}
        />
        <span
          className="absolute left-1.5 top-0 text-[3.5rem] font-black leading-none select-none pointer-events-none"
          style={{ color: info.color, opacity: 0.12 }}
        >
          {info.number}
        </span>
        {info.headshotUrl && (
          <img 
            src={info.headshotUrl}
            alt={info.driver}
            className="absolute -right-2 top-0 w-24 h-[120%] object-cover object-top opacity-80"
            style={{ maskImage: 'linear-gradient(to top, rgba(0,0,0,1) 15%, rgba(0,0,0,0) 100%)', WebkitMaskImage: 'linear-gradient(to right, rgba(0,0,0,0) 0%, rgba(0,0,0,1) 30%, rgba(0,0,0,1) 100%)' }}
          />
        )}
        <div className="relative z-10 drop-shadow-md">
          <p className="font-black text-white text-base leading-none tracking-wide">{info.driver}</p>
          <p className="text-[10px] text-white/60 mt-0.5 truncate max-w-[110px]">{info.team}</p>
        </div>
        {info.isInPit && (
          <span className="absolute top-2 left-2 text-[8px] font-black tracking-widest uppercase bg-yellow-500 text-black px-1.5 py-0.5 rounded shadow-lg z-20">
            PIT
          </span>
        )}
      </div>

      {/* Body */}
      <div className="px-3 pb-3 pt-2 flex flex-col gap-2 relative z-20">
        <div className="flex items-center gap-2">
          <TyreCircle compound={info.dynamicCompound} size={24} />
          <div className="flex flex-col leading-none">
            <span className="text-[10px] font-bold" style={{ color: cmp.bg }}>{cmp.label}</span>
            <span className="text-[9px] text-white/40 mt-0.5">{info.dynamicTyreLife} laps on tyre</span>
          </div>
          <div className="ml-auto text-right">
            <p className="font-mono text-[11px] font-bold text-emerald-400">{info.dynamicLastLapStr}</p>
            <p className="text-[9px] text-white/30">last lap</p>
          </div>
        </div>

        <div>
          <StintBar stints={info.dynamicStints} totalLaps={Math.max(info.dynamicTotalLaps, 1)} />
          <div className="flex justify-between text-[8px] text-white/25 mt-1">
            <span>L1</span>
            <span>{info.dynamicTotalLaps} laps so far</span>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

function DetailModal({ info, onClose }: { info: DynamicDriverInfo; onClose: () => void }) {
  const cmp = CMP[info.dynamicCompound] ?? CMP.UNKNOWN;

  return (
    <motion.div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onClick={onClose}
    >
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" />
      <motion.div
        className="relative z-10 bg-[#0e0e1a] border border-white/15 rounded-2xl w-full max-w-lg shadow-2xl overflow-hidden"
        initial={{ scale: 0.9, y: 20 }}
        animate={{ scale: 1, y: 0 }}
        exit={{ scale: 0.9, y: 20 }}
        onClick={e => e.stopPropagation()}
        style={{ borderTopColor: info.color, borderTopWidth: 4 }}
      >
        <div className="relative px-6 pt-5 pb-4 overflow-hidden"
          style={{ background: `linear-gradient(135deg, ${info.color}18 0%, transparent 60%)` }}>
          <span className="absolute right-4 top-2 text-7xl font-black opacity-10 select-none pointer-events-none leading-none"
            style={{ color: info.color }}>{info.number}</span>
          <div className="flex items-start gap-4">
            <div className="w-16 h-20 rounded-xl border border-white/10 bg-white/5 flex items-end justify-center overflow-hidden shrink-0 relative">
              {info.headshotUrl ? (
                <img src={info.headshotUrl} alt={info.driver} className="w-full h-full object-cover object-top scale-110 translate-y-2" />
              ) : (
                <span className="text-white/20 text-xs font-bold mb-4">IMG</span>
              )}
            </div>
            <div className="relative z-10">
              <h2 className="font-black text-xl text-white flex items-center gap-2">
                {info.driver}
                {info.isInPit && <span className="text-[9px] font-black tracking-widest uppercase bg-yellow-500 text-black px-1.5 py-0.5 rounded">PIT</span>}
              </h2>
              <p className="text-white/50 text-sm">{info.team}</p>
              <div className="flex items-center gap-3 mt-2">
                <div className="flex items-center gap-1.5">
                  <TyreCircle compound={info.dynamicCompound} size={20} />
                  <span className="text-[11px] font-bold" style={{ color: cmp.bg }}>{cmp.label}</span>
                  <span className="text-[10px] text-white/40">· {info.dynamicTyreLife} laps</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-3 border-t border-b border-white/8 divide-x divide-white/8">
          {[
            { label: 'Best Lap', value: info.bestLapStr, color: 'text-purple-400' },
            { label: 'Last Lap', value: info.dynamicLastLapStr, color: 'text-emerald-400' },
            { label: 'Current Laps', value: String(info.dynamicTotalLaps), color: 'text-white' },
          ].map(s => (
            <div key={s.label} className="py-3 px-4 text-center">
              <p className={`font-mono font-bold text-sm ${s.color}`}>{s.value}</p>
              <p className="text-[10px] text-white/30 mt-0.5">{s.label}</p>
            </div>
          ))}
        </div>

        <div className="px-6 py-4">
          <p className="text-[10px] uppercase tracking-widest text-white/30 font-bold mb-3">Stint History (Up to current time)</p>
          <div className="flex flex-col gap-2 max-h-[150px] overflow-y-auto pr-2 custom-scrollbar">
            {info.dynamicStints.map((s, i) => {
              const c = CMP[s.compound] ?? CMP.UNKNOWN;
              return (
                <div key={i} className="flex items-center gap-3">
                  <TyreCircle compound={s.compound} size={22} />
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-bold" style={{ color: c.bg }}>{c.label}</span>
                      {!s.fresh && <span className="text-[9px] bg-white/10 text-white/40 px-1.5 rounded-full">Used</span>}
                      <span className="text-[10px] text-white/40 ml-auto">Laps {s.startLap}–{s.endLap}</span>
                    </div>
                    <div className="mt-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
                      <div className="h-full rounded-full transition-all" style={{
                        width: `${(s.laps / Math.max(info.dynamicTotalLaps, 1)) * 100}%`,
                        background: c.bg,
                        opacity: s.fresh ? 1 : 0.5
                      }} />
                    </div>
                  </div>
                  <span className="font-mono text-[11px] text-white/60 w-14 text-right">{s.laps} laps</span>
                </div>
              );
            })}
          </div>
        </div>

        <button onClick={onClose}
          className="absolute top-4 right-4 w-7 h-7 rounded-full bg-white/10 hover:bg-white/20 flex items-center justify-center text-white/60 hover:text-white transition text-sm z-50">
          ✕
        </button>
      </motion.div>
    </motion.div>
  );
}

export default function PracticeDriverCards({ data, sessionData, currentTime, sessionName }: PracticeDriverCardsProps) {
  const [selected, setSelected] = useState<string | null>(null);

  // Compute dynamic state for all drivers based on currentTime
  const dynamicDrivers: DynamicDriverInfo[] = useMemo(() => {
    return Object.values(data).map(info => {
      const history = sessionData.lapHistory?.[info.driver];
      
      let currentLapEntry = history?.[0];
      let dynamicTotalLaps = 0;
      let isInPit = false;

      if (history && history.length > 0 && currentTime > 0) {
        for (const h of history) {
          if (h.trel <= currentTime) {
            currentLapEntry = h;
            dynamicTotalLaps++;
          } else break;
        }

        // Check pit status exactly like TimingTower
        for (const h of history) {
          if (h.pitInTrel !== null && h.pitOutTrel !== null) {
            if (h.pitInTrel <= currentTime && currentTime < h.pitOutTrel) {
              isInPit = true; break;
            }
          } else if (h.pitInTrel !== null && h.pitOutTrel === null) {
            if (h.pitInTrel <= currentTime) {
              const nextLapIdx = history.indexOf(h) + 1;
              if (nextLapIdx < history.length) {
                const nextLap = history[nextLapIdx];
                if (nextLap.pitOutTrel !== null && currentTime < nextLap.pitOutTrel) {
                  isInPit = true;
                }
              } else {
                isInPit = true;
              }
            }
          }
        }
      }

      // Rebuild stints up to dynamicTotalLaps
      const dynamicStints: PracticeStint[] = [];
      let currentStint: PracticeStint | null = null;
      let prevCmp = null;

      if (history) {
        for (let i = 0; i < dynamicTotalLaps; i++) {
          const h = history[i];
          const cmp = h.compound || 'UNKNOWN';
          const fresh = i === 0 || (h.tyreLife === 1 && history[i-1].tyreLife !== 0); // approximation
          if (cmp !== prevCmp) {
            if (currentStint) dynamicStints.push(currentStint);
            currentStint = { compound: cmp, startLap: h.lap, endLap: h.lap, laps: 1, fresh };
            prevCmp = cmp;
          } else if (currentStint) {
            currentStint.endLap = h.lap;
            currentStint.laps++;
          }
        }
        if (currentStint) dynamicStints.push(currentStint);
      }

      return {
        ...info,
        dynamicCompound: currentLapEntry?.compound || info.currentCompound,
        dynamicTyreLife: currentLapEntry?.tyreLife ?? 0,
        dynamicTotalLaps: dynamicTotalLaps,
        dynamicLastLapStr: currentLapEntry?.lapTimeStr || 'N/A',
        dynamicStints: dynamicStints.length > 0 ? dynamicStints : [],
        isInPit,
      };
    }).sort((a, b) => {
      // Sort by best lap time (fastest first), nulls last
      const ta = a.bestLapTime ?? Infinity;
      const tb = b.bestLapTime ?? Infinity;
      return ta - tb;
    });
  }, [data, sessionData, currentTime]);

  const selectedDriver = selected ? dynamicDrivers.find(d => d.driver === selected) : null;

  return (
    <div className="mt-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-white font-bold text-sm uppercase tracking-widest flex items-center gap-2">
          <span className="w-1.5 h-4 bg-blue-500 rounded-full" />
          Driver Status {sessionName && <span className="text-white/30 font-normal normal-case tracking-normal ml-1">· {sessionName}</span>}
        </h3>
        <span className="text-[10px] text-white/30">{dynamicDrivers.length} drivers · click for details</span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-2.5">
        {dynamicDrivers.map(info => (
          <CompactCard key={info.driver} info={info} onClick={() => setSelected(info.driver)} />
        ))}
      </div>

      <AnimatePresence>
        {selectedDriver && (
          <DetailModal info={selectedDriver} onClose={() => setSelected(null)} />
        )}
      </AnimatePresence>
    </div>
  );
}
