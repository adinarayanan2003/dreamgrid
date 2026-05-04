export type PlannerName = 'random' | 'astar' | 'random_shooting' | 'cem';

export type CellPos = {
  row: number;
  col: number;
};

export type Hazard = CellPos & {
  dr: number;
  dc: number;
};

export type GridState = {
  grid_size: number;
  max_steps: number;
  step: number;
  seed: number;
  agent: CellPos;
  goal: CellPos;
  walls: CellPos[];
  hazards: Hazard[];
};

export type EpisodePayload = {
  episode_id: string;
  state: GridState;
  event: string;
  reward: number;
  done: boolean;
  action?: number;
  action_name?: string;
};

export type PlanCandidate = {
  actions: number[];
  action_names: string[];
  score: number;
  path: CellPos[];
  event: string;
};

export type PlanPayload = {
  selected_action: number;
  selected_action_name: string;
  score: number;
  candidates: PlanCandidate[];
};

export type EvalMetrics = Record<
  string,
  {
    success_rate: number;
    collision_rate: number;
    avg_steps: number;
    avg_reward: number;
    episodes: number;
  }
>;

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  return response.json() as Promise<T>;
}

export function createEpisode(seed: number, gridSize = 16) {
  return request<EpisodePayload>('/api/episodes/generate', {
    method: 'POST',
    body: JSON.stringify({ seed, grid_size: gridSize })
  });
}

export function stepEpisode(episodeId: string, planner: PlannerName) {
  return request<EpisodePayload>(`/api/episodes/${episodeId}/step`, {
    method: 'POST',
    body: JSON.stringify({ planner })
  });
}

export function planEpisode(
  episodeId: string,
  planner: PlannerName,
  horizon: number,
  numCandidates: number
) {
  return request<PlanPayload>('/api/planners/plan', {
    method: 'POST',
    body: JSON.stringify({
      episode_id: episodeId,
      planner,
      horizon,
      num_candidates: numCandidates
    })
  });
}

export function runEval() {
  return request<{ metrics: EvalMetrics }>('/api/eval/run', {
    method: 'POST',
    body: JSON.stringify({
      planners: ['random', 'astar', 'random_shooting', 'cem'],
      episodes: 6,
      grid_size: 16,
      seed: 120,
      horizon: 6,
      num_candidates: 48
    })
  });
}
