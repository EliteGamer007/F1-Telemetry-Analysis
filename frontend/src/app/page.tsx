'use client';

import { useEffect, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { fetchSchedule, fetchTrack, fetchSession, ScheduleRace, TrackData, SessionData } from '@/lib/api';
import TrackMap from '@/components/TrackMap';
import TimingTower from '@/components/TimingTower';
import LapAnalysis from '@/components/LapAnalysis';
import TimeDeltaTab from '@/components/TimeDeltaTab';

const YEARS = [2026, 2025, 2024];

type Tab = 'animation' | 'lapanalysis' | 'timedelta';

function Spinner({ status }: { status: string }) {
  const stages = [
    { key: 'track', label: 'Track Layout', match: 'track' },
    { key: 'telemetry', label: 'Telemetry', match: 'telemetry' },
  ];
  const activeStage = stages.findIndex(s => status.toLowerCase().includes(s.match));
  return (
    <div className="flex flex-col items-center justify-center gap-6 py-24">
      <div className="relative w-12 h-12">
        <div className="w-12 h-12 border-2 border-red-500/15 rounded-full" />
        <div className="absolute inset-0 w-12 h-12 border-2 border-transparent border-t-red-500 rounded-full animate-spin" />
      </div>
      {status && (
        <div className="text-center">
          <p className="text-white/70 text-sm font-medium">{status}</p>
          <div className="flex items-center gap-3 mt-3 justify-center">
            {stages.map((s, i) => (
              <div key={s.key} className="flex items-center gap-1.5">
                <div className={`w-2 h-2 rounded-full transition-all ${
                  i < activeStage ? 'bg-emerald-500' :
                  i === activeStage ? 'bg-red-500 animate-pulse' :
                  'bg-white/15'
                }`} />
                <span className={`text-[11px] ${
                  i < activeStage ? 'text-emerald-500/70' :
                  i === activeStage ? 'text-white/70' :
                  'text-white/20'
                }`}>{s.label}</span>
                {i < stages.length - 1 && <span className="text-white/10 text-[10px] ml-1">→</span>}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function StatusBar({ message }: { message: string }) {
  return (
    <div className="fixed bottom-4 left-1/2 -translate-x-1/2 bg-black/80 backdrop-blur border border-white/10 text-white/80 text-sm px-5 py-2.5 rounded-full z-50 shadow-xl">
      <span className="inline-block w-2 h-2 bg-red-500 rounded-full mr-2 animate-pulse" />
      {message}
    </div>
  );
}

export default function Home() {
  const [year, setYear] = useState(2025);
  const [races, setRaces] = useState<ScheduleRace[]>([]);
  const [selectedRace, setSelectedRace] = useState('');
  const [sessionType, setSessionType] = useState<'Qualifying' | 'Race'>('Qualifying');
  const [qualSession, setQualSession] = useState<'Q1' | 'Q2' | 'Q3'>('Q3');

  const [trackData, setTrackData] = useState<TrackData | null>(null);
  const [sessionData, setSessionData] = useState<SessionData | null>(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [tab, setTab] = useState<Tab>('animation');

  // Load schedule whenever year changes
  useEffect(() => {
    setRaces([]);
    setSelectedRace('');
    fetchSchedule(year)
      .then(r => { setRaces(r); setSelectedRace(r[0]?.EventName ?? ''); })
      .catch(() => setError('Failed to load race schedule.'));
  }, [year]);

  const handleLoad = useCallback(async () => {
    if (!selectedRace) return;
    setLoading(true);
    setError('');
    setTrackData(null);
    setSessionData(null);
    const sessionCode = sessionType === 'Qualifying' ? 'Q' : 'R';
    const qual = sessionType === 'Qualifying' ? qualSession : undefined;

    try {
      setStatus(`Loading track layout for ${selectedRace}...`);
      const track = await fetchTrack(year, selectedRace, sessionCode);
      setTrackData(track);

      setStatus(`Loading ${selectedRace} ${year} ${qual ?? 'Race'} telemetry...`);
      const session = await fetchSession(year, selectedRace, sessionCode, qual);
      setSessionData(session);
      setStatus('');
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load session.');
      setStatus('');
    } finally {
      setLoading(false);
    }
  }, [year, selectedRace, sessionType, qualSession]);

  const TABS: { id: Tab; label: string; qualOnly?: boolean }[] = [
    { id: 'animation', label: '🏁 Race Animation' },
    { id: 'lapanalysis', label: '📊 Lap Analysis', qualOnly: true },
    { id: 'timedelta', label: '⏱ Time Delta', qualOnly: true },
  ];

  return (
    <div className="min-h-screen bg-[#08080f] text-white font-[Inter,sans-serif]">
      {/* Header */}
      <header className="border-b border-white/5 bg-black/40 backdrop-blur sticky top-0 z-40">
        <div className="max-w-[1600px] mx-auto px-6 py-3 flex items-center gap-4">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 bg-red-500 rounded-md flex items-center justify-center text-sm font-black">F1</div>
            <span className="font-bold text-base tracking-tight">Telemetry Analysis</span>
          </div>
          <div className="h-5 w-px bg-white/10" />
          {sessionData && (
            <span className="text-white/40 text-sm">{selectedRace} · {year} · {sessionType === 'Qualifying' ? qualSession : 'Race'}</span>
          )}
        </div>
      </header>

      <main className="max-w-[1600px] mx-auto px-6 py-6 flex flex-col gap-6">
        {/* Session Selector */}
        <div className="bg-white/[0.03] rounded-2xl border border-white/10 p-5">
          <h2 className="text-white/50 text-xs font-bold uppercase tracking-widest mb-4">Session Selection</h2>
          <div className="flex flex-wrap items-end gap-4">
            {/* Year */}
            <div>
              <label className="block text-white/40 text-xs mb-1.5 uppercase tracking-wider">Year</label>
              <div className="flex gap-1.5">
                {YEARS.map(y => (
                  <button key={y} onClick={() => setYear(y)}
                    className={`px-3 py-1.5 rounded-lg text-sm font-semibold transition-all ${year === y ? 'bg-red-500 text-white' : 'bg-white/10 text-white/60 hover:bg-white/20'}`}>
                    {y}
                  </button>
                ))}
              </div>
            </div>

            {/* Grand Prix */}
            <div className="flex-1 min-w-[200px]">
              <label className="block text-white/40 text-xs mb-1.5 uppercase tracking-wider">Grand Prix</label>
              <select value={selectedRace} onChange={e => setSelectedRace(e.target.value)}
                className="w-full bg-white/10 text-white text-sm rounded-xl px-3 py-2 border border-white/15 focus:outline-none focus:border-red-500 transition">
                {races.map(r => (
                  <option key={r.EventName} value={r.EventName} style={{ background: '#0f0f1a' }}>{r.EventName}</option>
                ))}
              </select>
            </div>

            {/* Session Type */}
            <div>
              <label className="block text-white/40 text-xs mb-1.5 uppercase tracking-wider">Session</label>
              <div className="flex gap-1.5">
                {(['Qualifying', 'Race'] as const).map(s => (
                  <button key={s} onClick={() => setSessionType(s)}
                    className={`px-3 py-1.5 rounded-lg text-sm font-semibold transition-all ${sessionType === s ? 'bg-red-500 text-white' : 'bg-white/10 text-white/60 hover:bg-white/20'}`}>
                    {s}
                  </button>
                ))}
              </div>
            </div>

            {/* Qual session */}
            {sessionType === 'Qualifying' && (
              <div>
                <label className="block text-white/40 text-xs mb-1.5 uppercase tracking-wider">Phase</label>
                <div className="flex gap-1.5">
                  {(['Q1', 'Q2', 'Q3'] as const).map(q => (
                    <button key={q} onClick={() => setQualSession(q)}
                      className={`px-3 py-1.5 rounded-lg text-sm font-bold transition-all ${qualSession === q ? 'bg-red-500 text-white' : 'bg-white/10 text-white/60 hover:bg-white/20'}`}>
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}

            <button onClick={handleLoad} disabled={loading || !selectedRace}
              className="px-7 py-2.5 bg-red-500 hover:bg-red-600 disabled:opacity-40 disabled:cursor-not-allowed text-white font-bold rounded-xl text-sm transition-all active:scale-95 shadow-lg shadow-red-500/20">
              {loading ? 'Loading...' : 'Load Session'}
            </button>
          </div>

          {error && (
            <div className="mt-4 bg-red-500/10 border border-red-500/30 text-red-400 rounded-xl px-4 py-3 text-sm">
              ⚠️ {error}
            </div>
          )}
        </div>

        {/* Loading */}
        {loading && <Spinner status={status} />}

        {/* Main content */}
        {!loading && sessionData && trackData && (
          <>
            {/* Tabs */}
            <div className="flex gap-1 bg-white/5 p-1 rounded-xl w-fit border border-white/10">
              {TABS.map(t => {
                const disabled = t.qualOnly && sessionType !== 'Qualifying';
                return (
                  <button key={t.id}
                    onClick={() => !disabled && setTab(t.id)}
                    disabled={disabled}
                    className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all ${tab === t.id ? 'bg-red-500 text-white shadow-md' : disabled ? 'text-white/20 cursor-not-allowed' : 'text-white/60 hover:text-white hover:bg-white/10'}`}>
                    {t.label}
                    {disabled && <span className="ml-1.5 text-[10px] opacity-60">Qual only</span>}
                  </button>
                );
              })}
            </div>

            <AnimatePresence mode="wait">
              {tab === 'animation' && (
                <motion.div key="animation"
                  initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                  className="grid grid-cols-[1fr_300px] gap-5">
                  <div className="bg-white/[0.03] rounded-2xl border border-white/10 p-4">
                    <h2 className="text-white font-bold text-sm uppercase tracking-widest mb-4 flex items-center gap-2">
                      <span className="w-1.5 h-4 bg-red-500 rounded-full" />Live Animation
                    </h2>
                    <TrackMap trackData={trackData} sessionData={sessionData} />
                  </div>
                  <div className="bg-white/[0.03] rounded-2xl border border-white/10 p-4 overflow-y-auto max-h-[780px]">
                    <h2 className="text-white font-bold text-sm uppercase tracking-widest mb-4 flex items-center gap-2">
                      <span className="w-1.5 h-4 bg-red-500 rounded-full" />Timing Tower
                    </h2>
                    <TimingTower leaderboard={sessionData.leaderboard} />
                  </div>
                </motion.div>
              )}

              {tab === 'lapanalysis' && (
                <motion.div key="lapanalysis"
                  initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                  <LapAnalysis sessionData={sessionData} />
                </motion.div>
              )}

              {tab === 'timedelta' && (
                <motion.div key="timedelta"
                  initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                  <TimeDeltaTab sessionData={sessionData} />
                </motion.div>
              )}
            </AnimatePresence>
          </>
        )}

        {!loading && !sessionData && (
          <div className="flex flex-col items-center justify-center py-28 gap-5">
            <div className="w-16 h-16 rounded-2xl bg-red-500/10 border border-red-500/20 flex items-center justify-center text-3xl">🏎️</div>
            <div className="text-center">
              <p className="text-white/60 text-base font-semibold">No session loaded</p>
              <p className="text-white/30 text-sm mt-1">Select a year, Grand Prix and session type above, then click <span className="text-red-400 font-medium">Load Session</span></p>
            </div>
            <div className="flex gap-6 text-[11px] text-white/20 mt-2">
              {['Track Map', 'Timing Tower', 'Lap Analysis', 'Time Delta'].map(f => (
                <span key={f} className="flex items-center gap-1.5">
                  <span className="w-1 h-1 rounded-full bg-red-500/50" />{f}
                </span>
              ))}
            </div>
          </div>
        )}
      </main>

      {status && <StatusBar message={status} />}
    </div>
  );
}
