import { Activity, BrainCircuit, Gauge, GitBranch, Play, RefreshCcw, StepForward } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import {
  CellPos,
  EpisodePayload,
  EvalMetrics,
  PlanCandidate,
  PlannerName,
  PredictionPayload,
  RolloutEvalMetrics,
  RolloutPredictionPayload,
  createEpisode,
  planEpisode,
  predictNext,
  predictRollout,
  runEval,
  runRolloutEval,
  stepEpisode
} from './api';

const plannerLabels: Record<PlannerName, string> = {
  random: 'Random',
  astar: 'A*',
  random_shooting: 'Shooting',
  cem: 'CEM',
  learned_mpc: 'Learned MPC'
};

export function App() {
  const [episode, setEpisode] = useState<EpisodePayload | null>(null);
  const [planner, setPlanner] = useState<PlannerName>('cem');
  const [plan, setPlan] = useState<PlanCandidate[]>([]);
  const [selectedActionName, setSelectedActionName] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<EvalMetrics>({});
  const [rolloutMetrics, setRolloutMetrics] = useState<RolloutEvalMetrics | null>(null);
  const [prediction, setPrediction] = useState<PredictionPayload | null>(null);
  const [rolloutPrediction, setRolloutPrediction] = useState<RolloutPredictionPayload | null>(null);
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
      setSelectedActionName(null);
      setPrediction(null);
      setRolloutPrediction(null);
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
      setSelectedActionName(payload.selected_action_name);
      setPrediction(null);
      setRolloutPrediction(null);
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
      setSelectedActionName(planPayload.selected_action_name);
      setPrediction(null);
      setRolloutPrediction(null);
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
    try {
      const payload = await runRolloutEval();
      setRolloutMetrics(payload.metrics);
    } catch {
      setRolloutMetrics(null);
    }
  }

  async function inspectModel() {
    if (!episode) return;
    setBusy(true);
    setStatus('Predicting');
    try {
      const action = plan[0]?.actions[0] ?? 3;
      const payload = await predictNext(episode.episode_id, action);
      setPrediction(payload);
      setStatus(`model predicted ${payload.action_name}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Prediction failed');
    } finally {
      setBusy(false);
    }
  }

  async function inspectRollout() {
    if (!episode) return;
    setBusy(true);
    setStatus('Rolling out');
    try {
      const actions = plan[0]?.actions.slice(0, 10) ?? [3, 3, 1, 1, 3, 1, 3, 1];
      const payload = await predictRollout(episode.episode_id, actions);
      setRolloutPrediction(payload);
      setStatus(`rollout mse ${payload.avg_frame_mse.toFixed(4)}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Rollout failed');
    } finally {
      setBusy(false);
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

          <button type="button" onClick={inspectModel} disabled={busy || !episode} title="Inspect model prediction">
            <BrainCircuit size={16} />
            Inspect Model
          </button>

          <button type="button" onClick={inspectRollout} disabled={busy || !episode} title="Inspect multi-step rollout">
            <GitBranch size={16} />
            Inspect Rollout
          </button>

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
              <span>{selectedActionName ? `executes ${selectedActionName}` : `${plan.length} rollouts`}</span>
            </div>
            {selectedActionName && (
              <div className="selected-action">
                <span>Next action</span>
                <strong>{selectedActionName}</strong>
              </div>
            )}
            <div className="candidate-list">
              {plan.map((candidate, index) => (
                <CandidateRow key={`${candidate.score}-${index}`} candidate={candidate} index={index} />
              ))}
              {plan.length === 0 && <div className="empty">Run planner</div>}
            </div>
          </div>

          <div className="panel prediction-panel">
            <div className="panel-title">
              <h2>Real vs Predicted</h2>
              {prediction && <span>{prediction.model_id}</span>}
            </div>
            {prediction ? <PredictionView prediction={prediction} /> : <div className="empty">Inspect model</div>}
          </div>

          <div className="panel rollout-panel">
            <div className="panel-title">
              <h2>Multi-Step Drift</h2>
              {rolloutPrediction && <span>avg MSE {rolloutPrediction.avg_frame_mse.toFixed(4)}</span>}
            </div>
            {rolloutPrediction ? (
              <RolloutView rollout={rolloutPrediction} />
            ) : (
              <div className="empty">Inspect rollout</div>
            )}
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
            <div className="metric-section-title">Planner Scores</div>
            {Object.entries(metrics).map(([name, row]) => (
              <div className="metric-row" key={name}>
                <strong>{plannerLabels[name as PlannerName] ?? name}</strong>
                <span>{Math.round(row.success_rate * 100)}%</span>
                <span>{row.avg_steps.toFixed(1)} steps</span>
              </div>
            ))}
            {Object.keys(metrics).length === 0 && <div className="empty">No metrics</div>}
          </div>
          <div className="rollout-metric-table">
            <div className="metric-section-title">Learned Drift</div>
            {rolloutMetrics ? (
              Object.entries(rolloutMetrics.horizons).map(([horizon, row]) => (
                <div className="rollout-metric-row" key={horizon}>
                  <strong>H{horizon}</strong>
                  <span>{row.frame_mse.toFixed(4)} mse</span>
                  <span>{row.reward_mae.toFixed(3)} mae</span>
                  <span>{Math.round(row.done_accuracy * 100)}%</span>
                </div>
              ))
            ) : (
              <div className="empty">No drift metrics</div>
            )}
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
  const firstAction = candidate.action_names[0] ?? 'stay';
  const rolloutTail = candidate.action_names.slice(1, 12);

  return (
    <div className="candidate">
      <div className="candidate-head">
        <strong>#{index + 1}</strong>
        <span>{candidate.event}</span>
        <span>{candidate.score.toFixed(3)}</span>
      </div>
      <div className="candidate-actions">
        <div className="candidate-action-primary">
          <span>First</span>
          <strong>{firstAction}</strong>
        </div>
        <div className="candidate-tail">
          <span>Rollout</span>
          <div>{rolloutTail.length > 0 ? rolloutTail.join(' -> ') : 'complete'}</div>
        </div>
      </div>
    </div>
  );
}

function PredictionView({ prediction }: { prediction: PredictionPayload }) {
  const frames = [
    ['Current', prediction.current_image],
    ['Actual Next', prediction.actual_next_image],
    ['Predicted', prediction.predicted_next_image],
    ['Error', prediction.error_image]
  ] as const;

  return (
    <div className="prediction-view">
      <div className="prediction-frames">
        {frames.map(([label, image]) => (
          <figure key={label}>
            <img src={image} alt={label} />
            <figcaption>{label}</figcaption>
          </figure>
        ))}
      </div>
      <div className="prediction-stats">
        <span>action: {prediction.action_name}</span>
        <span>actual: {prediction.actual_event}</span>
        <span>reward: {prediction.predicted_reward.toFixed(3)} / {prediction.actual_reward.toFixed(3)}</span>
        <span>done: {(prediction.predicted_done_probability * 100).toFixed(1)}%</span>
      </div>
    </div>
  );
}

function RolloutView({ rollout }: { rollout: RolloutPredictionPayload }) {
  return (
    <div className="rollout-view">
      <div className="rollout-actions">{rollout.action_names.slice(0, 12).join(' -> ')}</div>
      <div className="rollout-strip">
        {rollout.frames.map((frame) => (
          <div className="rollout-step" key={frame.step}>
            <div className="rollout-step-head">
              <strong>step {frame.step}</strong>
              <span>{frame.frame_mse.toFixed(4)}</span>
            </div>
            <div className="rollout-images">
              <img src={frame.actual_image} alt={`Actual step ${frame.step}`} />
              <img src={frame.predicted_image} alt={`Predicted step ${frame.step}`} />
              <img src={frame.error_image} alt={`Error step ${frame.step}`} />
            </div>
            <div className="rollout-meta">
              <span>{frame.action_name}</span>
              <span>{frame.actual_event}</span>
              <span>{frame.predicted_reward.toFixed(2)}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function key(pos: CellPos) {
  return `${pos.row}:${pos.col}`;
}
