import { Activity, Gauge, GitBranch, Play, RefreshCcw, StepForward } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import {
  CellPos,
  EpisodePayload,
  EvalMetrics,
  PlanCandidate,
  PlannerName,
  createEpisode,
  planEpisode,
  runEval,
  stepEpisode
} from './api';

const plannerLabels: Record<PlannerName, string> = {
  random: 'Random',
  astar: 'A*',
  random_shooting: 'Shooting',
  cem: 'CEM'
};

export function App() {
  const [episode, setEpisode] = useState<EpisodePayload | null>(null);
  const [planner, setPlanner] = useState<PlannerName>('cem');
  const [plan, setPlan] = useState<PlanCandidate[]>([]);
  const [metrics, setMetrics] = useState<EvalMetrics>({});
  const [seed, setSeed] = useState(7);
  const [horizon, setHorizon] = useState(12);
  const [numCandidates, setNumCandidates] = useState(256);
  const [status, setStatus] = useState('Ready');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    void reset();
    void refreshEval();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function reset(nextSeed = seed) {
    setBusy(true);
    setStatus('Generating');
    try {
      const payload = await createEpisode(nextSeed);
      setEpisode(payload);
      setPlan([]);
      setStatus('Ready');
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Request failed');
    } finally {
      setBusy(false);
    }
  }

  async function refreshPlan() {
    if (!episode) return;
    setBusy(true);
    setStatus('Planning');
    try {
      const payload = await planEpisode(episode.episode_id, planner, horizon, numCandidates);
      setPlan(payload.candidates);
      setStatus(`${payload.selected_action_name} selected`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Planning failed');
    } finally {
      setBusy(false);
    }
  }

  async function step() {
    if (!episode || episode.done) return;
    setBusy(true);
    setStatus('Stepping');
    try {
      const payload = await stepEpisode(episode.episode_id, planner);
      setEpisode(payload);
      const planPayload = await planEpisode(payload.episode_id, planner, horizon, numCandidates);
      setPlan(planPayload.candidates);
      setStatus(payload.event);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Step failed');
    } finally {
      setBusy(false);
    }
  }

  async function refreshEval() {
    try {
      const payload = await runEval();
      setMetrics(payload.metrics);
    } catch {
      setMetrics({});
    }
  }

  const bestPath = useMemo(() => plan[0]?.path ?? [], [plan]);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>DreamGrid</h1>
          <p>World-model planning lab</p>
        </div>
        <div className="status-pill">
          <Activity size={16} />
          <span>{busy ? 'Working' : status}</span>
        </div>
      </header>

      <section className="workspace">
        <aside className="panel controls">
          <div className="panel-title">
            <Gauge size={18} />
            <h2>Control</h2>
          </div>

          <label>
            Planner
            <select value={planner} onChange={(event) => setPlanner(event.target.value as PlannerName)}>
              {Object.entries(plannerLabels).map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </label>

          <label>
            Seed
            <input
              type="number"
              value={seed}
              onChange={(event) => setSeed(Number(event.target.value))}
              min={0}
            />
          </label>

          <label>
            Horizon
            <input
              type="range"
              value={horizon}
              onChange={(event) => setHorizon(Number(event.target.value))}
              min={4}
              max={24}
            />
            <span className="inline-value">{horizon}</span>
          </label>

          <label>
            Candidates
            <input
              type="range"
              value={numCandidates}
              onChange={(event) => setNumCandidates(Number(event.target.value))}
              min={32}
              max={768}
              step={32}
            />
            <span className="inline-value">{numCandidates}</span>
          </label>

          <div className="button-row">
            <button type="button" onClick={() => reset(seed)} disabled={busy} title="Reset episode">
              <RefreshCcw size={16} />
              Reset
            </button>
            <button type="button" onClick={refreshPlan} disabled={busy || !episode} title="Plan">
              <GitBranch size={16} />
              Plan
            </button>
            <button type="button" onClick={step} disabled={busy || !episode || episode.done} title="Step">
              <StepForward size={16} />
              Step
            </button>
          </div>

          <button className="primary" type="button" onClick={step} disabled={busy || !episode || episode.done}>
            <Play size={16} />
            Execute Planner Step
          </button>
        </aside>

        <section className="lab">
          <div className="panel board-panel">
            <div className="panel-title">
              <h2>Live World</h2>
              {episode && (
                <span>
                  step {episode.state.step}/{episode.state.max_steps}
                </span>
              )}
            </div>
            {episode ? <Grid state={episode.state} path={bestPath} /> : <div className="empty">Loading</div>}
          </div>

          <div className="panel imagination">
            <div className="panel-title">
              <h2>Imagined Candidates</h2>
              <span>{plan.length} rollouts</span>
            </div>
            <div className="candidate-list">
              {plan.map((candidate, index) => (
                <CandidateRow key={`${candidate.score}-${index}`} candidate={candidate} index={index} />
              ))}
              {plan.length === 0 && <div className="empty">Run planner</div>}
            </div>
          </div>
        </section>

        <aside className="panel metrics">
          <div className="panel-title">
            <h2>Evaluation</h2>
            <button type="button" onClick={refreshEval} title="Refresh evaluation">
              <RefreshCcw size={15} />
            </button>
          </div>
          <div className="metric-table">
            {Object.entries(metrics).map(([name, row]) => (
              <div className="metric-row" key={name}>
                <strong>{plannerLabels[name as PlannerName] ?? name}</strong>
                <span>{Math.round(row.success_rate * 100)}%</span>
                <span>{row.avg_steps.toFixed(1)} steps</span>
              </div>
            ))}
            {Object.keys(metrics).length === 0 && <div className="empty">No metrics</div>}
          </div>
        </aside>
      </section>
    </main>
  );
}

function Grid({ state, path }: { state: EpisodePayload['state']; path: CellPos[] }) {
  const wallSet = new Set(state.walls.map(key));
  const hazardSet = new Set(state.hazards.map(key));
  const pathSet = new Set(path.map(key));
  const cells = [];

  for (let row = 0; row < state.grid_size; row += 1) {
    for (let col = 0; col < state.grid_size; col += 1) {
      const pos = { row, col };
      const classes = ['cell'];
      if (wallSet.has(key(pos))) classes.push('wall');
      if (pathSet.has(key(pos))) classes.push('path');
      if (hazardSet.has(key(pos))) classes.push('hazard');
      if (state.goal.row === row && state.goal.col === col) classes.push('goal');
      if (state.agent.row === row && state.agent.col === col) classes.push('agent');
      cells.push(<div className={classes.join(' ')} key={`${row}-${col}`} />);
    }
  }

  return (
    <div className="grid-wrap">
      <div className="grid" style={{ gridTemplateColumns: `repeat(${state.grid_size}, 1fr)` }}>
        {cells}
      </div>
    </div>
  );
}

function CandidateRow({ candidate, index }: { candidate: PlanCandidate; index: number }) {
  return (
    <div className="candidate">
      <div className="candidate-head">
        <strong>#{index + 1}</strong>
        <span>{candidate.event}</span>
        <span>{candidate.score.toFixed(3)}</span>
      </div>
      <div className="actions">{candidate.action_names.slice(0, 12).join(' -> ')}</div>
    </div>
  );
}

function key(pos: CellPos) {
  return `${pos.row}:${pos.col}`;
}

