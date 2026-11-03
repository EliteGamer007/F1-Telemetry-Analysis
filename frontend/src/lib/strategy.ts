import { API_BASE } from './api';

export type Compound = 'SOFT' | 'MEDIUM' | 'HARD';
export type ScenarioLevel = 'low' | 'base' | 'high';
export interface StrategyPlan {
  compounds: Compound[]; pits: number[]; lengths: number[]; mean: number; median: number;
  p10: number; p90: number; delta: number; p10_delta: number; p90_delta: number; pit_windows: number[][];
}
export interface Stint {
  driver: string; session: string; stint: number; compound: Compound; samples: number;
  age_min: number; age_max: number; start_lap: number; end_lap: number; fresh: boolean | null;
  median_seconds: number; trend: number | null; long_run: boolean;
}
export interface CurvePoint { age: number; median: number; p10: number; p90: number }
export interface StrategyDriver {
  code: string; name: string; team: string; color: string; qualifying_position: number | null;
  status: string; practice_samples: number; qualifying_delta: number | null;
  compounds: Record<Compound, { slope: number; practice_samples: number; historical_samples: number; source: string }>;
  scenarios: Record<string, StrategyPlan[]>; curves: Record<Compound, CurvePoint[]>; stints: Stint[]; evidence: string[];
}
export interface PreRaceReport {
  schema_version: number; year: number; round: number; event: string; location: string; date: string; support: string;
  manifest: { kind: string; cutoff: string; cutoff_method: string; generated_at: string; era: string; model_version: string;
    model_hash: string; source_hash: string; training_hash: string; training_events: string[]; metadata: Record<string, string> };
  coverage: { sessions: { code: string; end: string; laps: number; track_temp: number | null; rainfall_observed: boolean | null }[];
    missing: { code: string; reason: string }[]; training_laps: number; calibration_events: number };
  policy: { laps: number; pit_loss: number; pit_sigma: number; pit_source: string; inventory_source: string };
  assumptions: string[]; uncertainty: string; drivers: Record<string, StrategyDriver>;
}
export interface Weekend { year: number; round: number; event: string; location: string; date: string; era: string; support: string; file?: string }
export interface StrategyCatalog { weekends: Weekend[]; replays: { year: number; round: number; event: string }[] }
export interface Loaded<T> { data: T; saved: boolean }

async function json<T>(url: string, timeout = 8000): Promise<T> {
  const response = await fetch(url, { signal: AbortSignal.timeout(timeout) });
  if (!response.ok) throw new Error(response.status === 404 ? 'This weekend has not been prepared yet.' : `Data service returned ${response.status}.`);
  return response.json();
}
export async function loadStrategyCatalog(): Promise<Loaded<StrategyCatalog>> {
  try { return { data: await json<StrategyCatalog>(`${API_BASE}/api/strategy/weekends`), saved: false }; }
  catch { return { data: await json<StrategyCatalog>('/samples/catalog.json'), saved: true }; }
}
export async function loadPreRaceReport(year: number, round: number): Promise<Loaded<PreRaceReport>> {
  try {
    const data = await json<PreRaceReport>(`${API_BASE}/api/strategy/prerace/${year}/${round}`);
    if (data.schema_version !== 1 || !data.drivers) throw new Error('Unsupported report format');
    return { data, saved: false };
  } catch {
    const catalog = await json<StrategyCatalog>('/samples/catalog.json');
    const entry = catalog.weekends.find(w => w.year === year && w.round === round);
    if (!entry?.file) throw new Error('No prepared report is available for this weekend.');
    return { data: await json<PreRaceReport>(`/samples/reports/${entry.file}`), saved: true };
  }
}
