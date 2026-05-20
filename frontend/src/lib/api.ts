// API client for the FastAPI backend
const API_BASE = 'http://localhost:8000';

export interface ScheduleRace {
  EventName: string;
  EventDate: string;
  EventFormat: string;
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
}

export interface SessionData {
  leaderboard: DriverRow[];
  telemetry: Record<string, { Distance: number[]; Trel: number[]; Speed?: number[]; Throttle?: number[]; Brake?: number[]; RPM?: number[]; nGear?: number[]; DRS?: number[] }>;
  driverColors: Record<string, { color: string; team: string }>;
  maxDuration: number;
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
