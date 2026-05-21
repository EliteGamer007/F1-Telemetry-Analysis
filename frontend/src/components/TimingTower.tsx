'use client';

import { SessionData, DriverRow, LapHistoryEntry } from '@/lib/api';
import { motion, AnimatePresence } from 'framer-motion';
import { useMemo } from 'react';

interface TimingTowerProps {
  leaderboard: DriverRow[];
  sessionData?: SessionData;
  currentTime?: number;
}

// Tyre compound display config
const COMPOUND: Record<string, { bg: string; fg: string; letter: string; name: string }> = {
  SOFT:         { bg: '#E8002D', fg: '#fff', letter: 'S', name: 'Soft' },
  MEDIUM:       { bg: '#FFF200', fg: '#000', letter: 'M', name: 'Medium' },
  HARD:         { bg: '#EDEDED', fg: '#000', letter: 'H', name: 'Hard' },
  INTERMEDIATE: { bg: '#39B54A', fg: '#fff', letter: 'I', name: 'Inter' },
  WET:          { bg: '#0067FF', fg: '#fff', letter: 'W', name: 'Wet' },
  UNKNOWN:      { bg: '#555555', fg: '#fff', letter: '?', name: '?' },
};

function TyreIcon({ compound, size = 20 }: { compound: string; size?: number }) {
  const cfg = COMPOUND[compound] ?? COMPOUND.UNKNOWN;
  return (
    <svg width={size} height={size} viewBox="0 0 20 20" aria-label={cfg.name}>
      <circle cx="10" cy="10" r="9" fill="#111" />
      <circle cx="10" cy="10" r="7.5" fill={cfg.bg} />
      <circle cx="10" cy="10" r="4" fill="#111" />
      <text x="10" y="10" textAnchor="middle" dominantBaseline="central"
        fontSize="5.5" fontWeight="900" fontFamily="Inter,Arial,sans-serif" fill={cfg.fg}>
        {cfg.letter}
      </text>
    </svg>
  );
}

function PosBadge({ pos }: { pos: number }) {
  const base = 'w-[22px] h-[22px] rounded flex items-center justify-center text-[10px] font-black shrink-0';
  if (pos === 1) return <span className={`${base} bg-yellow-400 text-black shadow-[0_0_8px_rgba(250,204,21,0.5)]`}>1</span>;
  if (pos === 2) return <span className={`${base} bg-slate-300 text-black`}>2</span>;
  if (pos === 3) return <span className={`${base} bg-orange-600 text-white`}>3</span>;
  return <span className={`${base} bg-white/10 text-white/70`}>{pos}</span>;
}

function SectorTime({ value, best }: { value: string; best: boolean }) {
  return (
    <span className={`font-mono text-[9px] px-1 py-[2px] rounded min-w-[40px] text-center transition-all ${
      best
        ? 'bg-purple-600 text-white font-bold shadow-[0_0_6px_rgba(168,85,247,0.5)]'
        : 'bg-white/5 text-white/40'
    }`}>
      {value}
    </span>
  );
}

interface DynamicRow extends DriverRow {
  currentLapEntry?: LapHistoryEntry;
  inPit?: boolean;
  retired?: boolean;
}

export default function TimingTower({ leaderboard, sessionData, currentTime = 0 }: TimingTowerProps) {
  
  // Compute dynamic leaderboard with per-lap data
  const currentLeaderboard: DynamicRow[] = useMemo(() => {
    if (!sessionData?.isRace || !sessionData.lapHistory || currentTime === 0) {
      return leaderboard as DynamicRow[];
    }

    const updated: DynamicRow[] = leaderboard.map(row => {
      const history = sessionData.lapHistory?.[row.driver];
      if (!history || history.length === 0) return { ...row };
      
      // Find the lap entry that is current: last lap completed before currentTime
      let currentLapEntry = history[0];
      for (const h of history) {
        if (h.trel <= currentTime) currentLapEntry = h;
        else break;
      }

      // Check if currently in pit: pitInTrel <= currentTime < pitOutTrel
      let inPit = false;
      for (const h of history) {
        if (h.pitInTrel !== null && h.pitOutTrel !== null) {
          if (h.pitInTrel <= currentTime && currentTime < h.pitOutTrel) {
            inPit = true;
            break;
          }
        } else if (h.pitInTrel !== null && h.pitOutTrel === null) {
          // Lap with pit-in but pit-out is on the next lap row
          if (h.pitInTrel <= currentTime) {
            // Check if the next lap's pitOut covers us
            const nextLapIdx = history.indexOf(h) + 1;
            if (nextLapIdx < history.length) {
              const nextLap = history[nextLapIdx];
              if (nextLap.pitOutTrel !== null && currentTime < nextLap.pitOutTrel) {
                inPit = true;
              }
            } else {
              inPit = true;
            }
          }
        }
      }

      // Determine if retired: driver status is 'Retired' and current time is past their last recorded lap
      const retired = row.status === 'Retired';
      // After their final lap, they're out
      const lastLap = history[history.length - 1];
      const isOut = retired && lastLap && currentTime > lastLap.trel + 30;

      return {
        ...row,
        position: currentLapEntry?.position || row.position,
        compound: currentLapEntry?.compound || row.compound,
        tyreLife: currentLapEntry?.tyreLife ?? row.tyreLife,
        S1: currentLapEntry?.S1 || row.S1,
        S2: currentLapEntry?.S2 || row.S2,
        S3: currentLapEntry?.S3 || row.S3,
        bestLapStr: currentLapEntry?.lapTimeStr || row.bestLapStr,
        currentLapEntry,
        inPit,
        retired: isOut,
      };
    });

    // Sort by dynamic position (0 = no position, push to end)
    updated.sort((a, b) => {
      if (a.retired && !b.retired) return 1;
      if (!a.retired && b.retired) return -1;
      const pa = a.position || 999;
      const pb = b.position || 999;
      return pa - pb;
    });

    return updated;
  }, [leaderboard, sessionData, currentTime]);

  // Collect unique compounds for legend
  const compounds = [...new Set(currentLeaderboard.map(r => r.compound))].filter(c => COMPOUND[c]);

  return (
    <div className="flex flex-col gap-0">
      {/* ── Compound legend ───────────────────────────── */}
      <div className="flex items-center gap-3 pb-2 mb-2 border-b border-white/8 flex-wrap">
        <span className="text-[9px] uppercase tracking-widest text-white/30 font-bold">Tyre</span>
        {compounds.map(c => (
          <span key={c} className="flex items-center gap-1.5">
            <TyreIcon compound={c} size={14} />
            <span className="text-[10px] text-white/50">{COMPOUND[c]?.name}</span>
          </span>
        ))}
      </div>

      {/* ── Column headers ────────────────────────────── */}
      <div className="grid items-center gap-1 mb-1 px-1"
        style={{ gridTemplateColumns: '22px 52px 40px 40px 40px 60px 44px 20px' }}>
        {['P', 'Driver', 'S1', 'S2', 'S3', 'Time', 'Gap', ''].map((h, i) => (
          <span key={i} className={`text-[8px] font-bold uppercase tracking-widest text-white/25 ${i >= 2 && i <= 4 ? 'text-center' : i >= 5 ? 'text-right' : ''}`}>
            {h}
          </span>
        ))}
      </div>

      {/* ── Rows ─────────────────────────────────────── */}
      <div className="flex flex-col gap-[2px]">
        <AnimatePresence>
          {currentLeaderboard.map(row => {
            const isRetired = row.retired;
            const isInPit = (row as DynamicRow).inPit;

            return (
              <motion.div
                key={row.driver}
                layout
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.25, layout: { duration: 0.35 } }}
                className={`group grid items-center gap-1 px-1 py-[4px] rounded-md transition-colors ${
                  isRetired
                    ? 'opacity-35 grayscale'
                    : isInPit
                    ? 'bg-yellow-500/10 border border-yellow-500/20'
                    : 'hover:bg-white/[0.04]'
                }`}
                style={{
                  gridTemplateColumns: '22px 52px 40px 40px 40px 60px 44px 20px',
                  borderLeft: `3px solid ${isRetired ? '#555' : row.color}`,
                }}
              >
                <PosBadge pos={row.position} />

                {/* Driver abbr + status badge */}
                <div className="flex items-center gap-1 min-w-0">
                  <span className={`font-black text-[12px] tracking-wide truncate ${isRetired ? 'text-white/30' : 'text-white'}`}>
                    {row.driver}
                  </span>
                  {isInPit && (
                    <span className="text-[7px] font-bold bg-yellow-500 text-black px-1 rounded leading-tight shrink-0">PIT</span>
                  )}
                  {isRetired && (
                    <span className="text-[7px] font-bold bg-red-800/60 text-red-300 px-1 rounded leading-tight shrink-0">OUT</span>
                  )}
                </div>

                <SectorTime value={row.S1} best={row.S1_best} />
                <SectorTime value={row.S2} best={row.S2_best} />
                <SectorTime value={row.S3} best={row.S3_best} />

                {/* Lap time */}
                <span className={`font-mono text-[11px] font-semibold text-right tabular-nums ${isRetired ? 'text-white/25' : 'text-emerald-400'}`}>
                  {row.bestLapStr}
                </span>

                {/* Gap */}
                <span className="font-mono text-[10px] text-right tabular-nums">
                  {row.position === 1 ? (
                    <span className="text-yellow-400 font-bold text-[9px]">LEAD</span>
                  ) : isRetired ? (
                    <span className="text-red-500/60 text-[9px]">DNF</span>
                  ) : (
                    <span className="text-red-400">{row.gap || '+0.000'}</span>
                  )}
                </span>

                {/* Tyre icon */}
                <div className="flex justify-center">
                  <TyreIcon compound={row.compound} size={17} />
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>

      {/* ── Team colour legend ────────────────────────── */}
      {(() => {
        const teams = [...new Map(currentLeaderboard.map(r => [r.team, r.color])).entries()];
        return (
          <div className="mt-3 pt-3 border-t border-white/8">
            <p className="text-[9px] uppercase tracking-widest text-white/25 font-bold mb-2">Teams</p>
            <div className="grid grid-cols-2 gap-x-3 gap-y-1">
              {teams.map(([team, color]) => (
                <span key={team} className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-sm shrink-0" style={{ background: color }} />
                  <span className="text-[10px] text-white/45 truncate">{team}</span>
                </span>
              ))}
            </div>
          </div>
        );
      })()}
    </div>
  );
}
