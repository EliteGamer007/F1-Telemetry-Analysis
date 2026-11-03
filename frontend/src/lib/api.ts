// API client for the FastAPI backend
export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');
export const PUBLIC_MODE = process.env.NEXT_PUBLIC_PREPARED_ONLY === 'true';

export interface ScheduleRace {
  EventName: string;
  EventDate: string;
  EventFormat: string;
  RoundNumber?: number;
  sessions?: string[];
}

export interface TrackData {
  track: { x: number[]; y: number[]; distance: number[] };
  corners: { Number: number; Letter: string; x: number; y: number; label_x: number; label_y: number; distance: number }[];
  bounds: { xMin: number; xMax: number; yMin: number; yMax: number };
}

export interface DriverRow {
  driver: string;
  team: string;
  lapTime: number;
  bestLapStr: string;
  compound: string;
  tyreLife: number;
  S1: string; S2: string; S3: string;
  S1_sec: number; S2_sec: number; S3_sec: number;
  S1_best: boolean; S2_best: boolean; S3_best: boolean;
  position: number;
  gap: string;
  color: string;
  status?: string; // 'Finished' | 'Lapped' | 'Retired'
}

export interface LapHistoryEntry {
  lap: number;
  position: number;
  trel: number;
  lapTime: number;
  lapTimeStr: string;
  compound: string;
  tyreLife: number;
  S1: string;
  S2: string;
  S3: string;
  pitInTrel: number | null;
  pitOutTrel: number | null;
}

export interface TrackStatusEvent {
  trel: number;
  endTrel: number;
  status: string; // 'AllClear' | 'Yellow' | 'SCDeploying' | 'SafetyCar' | 'RedFlag' | 'VSCDeployed' | 'VSCEnding'
  code: string;
}

export interface PracticeStint {
  compound: string;
  startLap: number;
  endLap: number;
  laps: number;
  fresh: boolean;
}

export interface QualLap {
  lap: number;
  lapType: 'out' | 'hot' | 'in' | 'prep' | 'unknown';
  lapTime: number | null;
  lapTimeStr: string;
  lapStartTrel: number | null;
  lapEndTrel: number | null;
  compound: string;
  tyreLife: number;
  S1: string;
  S2: string;
  S3: string;
  isAccurate: boolean;
}

export interface TrafficIncident {
  trel: number;
  driver1: string;
  driver2: string;
  x: number;
  y: number;
  severity: 'green' | 'yellow' | 'red' | 'blue';
  color: string;
  lapType1: string;
  lapType2: string;
  pushDriver: string | null;
  prepDriver: string | null;
  distance: number;
}

export interface PracticeDriverInfo {
  driver: string;
  team: string;
  number: string;
  color: string;
  teamColor: string;
  headshotUrl?: string | null;
  totalLaps: number;
  stints: PracticeStint[];
  currentCompound: string;
  currentTyreLife: number;
  lastLapTime: number | null;
  lastLapStr: string;
  bestLapTime: number | null;
  bestLapStr: string;
  bestLapStartTrel?: number | null;
  bestLapEndTrel?: number | null;
  compoundCounts: Record<string, number>;
  qualLaps?: QualLap[];
}

export interface SessionData {
  leaderboard: DriverRow[];
  telemetry: Record<string, { Distance: number[]; Trel: number[]; Speed?: number[]; X?: number[]; Y?: number[]; Throttle?: number[]; Brake?: number[]; RPM?: number[]; nGear?: number[]; DRS?: number[]; maxTrel?: number }>;
  fastestLapTelemetry?: Record<string, { Distance: number[]; Trel: number[]; Speed?: number[]; X?: number[]; Y?: number[]; Throttle?: number[]; Brake?: number[]; RPM?: number[]; nGear?: number[]; DRS?: number[]; maxTrel?: number }>;
  driverColors: Record<string, { color: string; team: string }>;
  maxDuration: number;
  isRace?: boolean;
  totalLaps?: number;
  lapHistory?: Record<string, LapHistoryEntry[]>;
  trackStatusEvents?: TrackStatusEvent[];
  practiceDriverData?: Record<string, PracticeDriverInfo> | null;
  trafficIncidents?: TrafficIncident[];
}

export interface TimeDeltaData {
  distance: number[];
  delta: number[];
  speed1: number[];
  speed2: number[];
}

export async function fetchSchedule(year: number): Promise<ScheduleRace[]> {
  const res = await fetch(`${API_BASE}/api/schedule/${year}`);
  if (!res.ok) throw new Error(`Schedule fetch failed: ${res.statusText}`);
  const data = await res.json();
  return data.races;
}

export async function fetchTrack(year: number, gp: string, sessionCode: string): Promise<TrackData> {
  const res = await fetch(`${API_BASE}/api/track?year=${year}&gp=${encodeURIComponent(gp)}&session_code=${sessionCode}`);
  if (!res.ok) throw new Error(`Track fetch failed: ${res.statusText}`);
  return res.json();
}

export async function fetchSession(year: number, gp: string, sessionCode: string, qualSession?: string): Promise<SessionData> {
  if (PUBLIC_MODE) {
    const races = await fetchSchedule(year);
    const race = races.find(r => r.EventName === gp);
    if (!race?.RoundNumber || sessionCode !== 'R') throw new Error('This public pilot offers the prepared race replays in the catalog.');
    const response = await fetch(`${API_BASE}/api/replay/${year}/${race.RoundNumber}/manifest`, { signal: AbortSignal.timeout(65000) });
    if (!response.ok) throw new Error('Prepared replay is temporarily unavailable. Try again after the service wakes.');
    const manifest: { session: SessionData; chunks: { id: string }[] } = await response.json();
    const data = manifest.session;
    data.telemetry = {}; data.fastestLapTelemetry = {};
    // Three bounded concurrent downloads avoid flooding the free service.
    for (let start = 0; start < manifest.chunks.length; start += 3) {
      const chunks = await Promise.all(manifest.chunks.slice(start, start + 3).map(async chunk => {
        const result = await fetch(`${API_BASE}/api/replay/${year}/${race.RoundNumber}/chunks/${chunk.id}`, { signal: AbortSignal.timeout(45000) });
        if (!result.ok) throw new Error('A replay chunk could not be loaded. Please retry.');
        return result.json() as Promise<Pick<SessionData, 'telemetry' | 'fastestLapTelemetry'>>;
      }));
      for (const chunk of chunks) { Object.assign(data.telemetry, chunk.telemetry); Object.assign(data.fastestLapTelemetry, chunk.fastestLapTelemetry); }
    }
    return data;
  }
  let url = `${API_BASE}/api/session?year=${year}&gp=${encodeURIComponent(gp)}&session_code=${sessionCode}`;
  if (qualSession) url += `&qual_session=${qualSession}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchTimeDelta(year: number, gp: string, sessionCode: string, qualSession: string | undefined, d1: string, d2: string): Promise<TimeDeltaData> {
  let url = `${API_BASE}/api/time-delta?year=${year}&gp=${encodeURIComponent(gp)}&session_code=${sessionCode}&driver1=${d1}&driver2=${d2}`;
  if (qualSession) url += `&qual_session=${qualSession}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
