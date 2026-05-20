'use client';

import { DriverRow } from '@/lib/api';
import { motion, AnimatePresence } from 'framer-motion';

interface TimingTowerProps {
  leaderboard: DriverRow[];
}

// Tyre compound display config
const COMPOUND: Record<string, { bg: string; fg: string; letter: string; name: string }> = {
  SOFT:         { bg: '#E8002D', fg: '#fff', letter: 'S', name: 'Soft' },
  MEDIUM:       { bg: '#FFF200', fg: '#000', letter: 'M', name: 'Medium' },
  HARD:         { bg: '#EDEDED', fg: '#000', letter: 'H', name: 'Hard' },
  INTERMEDIATE: { bg: '#39B54A', fg: '#fff', letter: 'I', name: 'Inter' },
  WET:          { bg: '#0067FF', fg: '#fff', letter: 'W', name: 'Wet' },
};

// F1-spec tyre SVG icon (circle with compound letter)
function TyreIcon({ compound, size = 20 }: { compound: string; size?: number }) {
  const cfg = COMPOUND[compound] ?? COMPOUND.HARD;
  return (
    <svg width={size} height={size} viewBox="0 0 20 20" aria-label={cfg.name}>
      {/* Outer tyre ring */}
      <circle cx="10" cy="10" r="9" fill="#111" />
      <circle cx="10" cy="10" r="7.5" fill={cfg.bg} />
      {/* Inner hub */}
      <circle cx="10" cy="10" r="4" fill="#111" />
      {/* Compound letter */}
      <text x="10" y="10" textAnchor="middle" dominantBaseline="central"
        fontSize="5.5" fontWeight="900" fontFamily="Inter,Arial,sans-serif" fill={cfg.fg}>
        {cfg.letter}
      </text>
    </svg>
  );
}

function PosBadge({ pos }: { pos: number }) {
  const base = 'w-[26px] h-[26px] rounded flex items-center justify-center text-[11px] font-black shrink-0';
  if (pos === 1) return <span className={`${base} bg-yellow-400 text-black shadow-[0_0_8px_rgba(250,204,21,0.5)]`}>1</span>;
  if (pos === 2) return <span className={`${base} bg-slate-300 text-black`}>2</span>;
  if (pos === 3) return <span className={`${base} bg-orange-600 text-white`}>3</span>;
  return <span className={`${base} bg-white/10 text-white/70`}>{pos}</span>;
}

function SectorTime({ value, best }: { value: string; best: boolean }) {
  return (
    <span className={`font-mono text-[10px] px-1.5 py-[3px] rounded min-w-[46px] text-center transition-all ${
      best
        ? 'bg-purple-600 text-white font-bold shadow-[0_0_6px_rgba(168,85,247,0.5)]'
        : 'bg-white/5 text-white/45'
    }`}>
      {value}
    </span>
  );
}

export default function TimingTower({ leaderboard }: TimingTowerProps) {
  // Collect unique compounds for legend
  const compounds = [...new Set(leaderboard.map(r => r.compound))].filter(c => COMPOUND[c]);

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
        style={{ gridTemplateColumns: '26px 1fr 46px 46px 46px 74px 48px 20px' }}>
        {['P', 'Driver', 'S1', 'S2', 'S3', 'Time', 'Gap', ''].map((h, i) => (
          <span key={i} className={`text-[9px] font-bold uppercase tracking-widest text-white/25 ${i >= 2 && i <= 4 ? 'text-center' : i >= 5 ? 'text-right' : ''}`}>
            {h}
          </span>
        ))}
      </div>

      {/* ── Rows ─────────────────────────────────────── */}
      <div className="flex flex-col gap-[2px]">
        <AnimatePresence>
          {leaderboard.map(row => (
            <motion.div
              key={row.driver}
              layout
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="group grid items-center gap-1 px-1 py-[5px] rounded-md hover:bg-white/[0.04] transition-colors"
              style={{
                gridTemplateColumns: '26px 1fr 46px 46px 46px 74px 48px 20px',
                borderLeft: `3px solid ${row.color}`,
              }}
            >
              <PosBadge pos={row.position} />

              {/* Driver abbr + team colour dot */}
              <div className="flex items-center gap-1.5 min-w-0">
                <span className="font-bold text-[13px] text-white tracking-wide truncate">{row.driver}</span>
                <span className="w-1.5 h-1.5 rounded-full shrink-0 opacity-70" style={{ background: row.color }} />
              </div>

              <SectorTime value={row.S1} best={row.S1_best} />
              <SectorTime value={row.S2} best={row.S2_best} />
              <SectorTime value={row.S3} best={row.S3_best} />

              {/* Lap time */}
              <span className="font-mono text-[12px] text-emerald-400 font-semibold text-right tabular-nums">
                {row.bestLapStr}
              </span>

              {/* Gap */}
              <span className="font-mono text-[11px] text-right tabular-nums">
                {row.gap
                  ? <span className="text-red-400">{row.gap}</span>
                  : <span className="text-yellow-400 font-bold text-[10px]">LEADER</span>}
              </span>

              {/* Tyre icon */}
              <div className="flex justify-center">
                <TyreIcon compound={row.compound} size={18} />
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* ── Team colour legend ────────────────────────── */}
      {(() => {
        const teams = [...new Map(leaderboard.map(r => [r.team, r.color])).entries()];
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
