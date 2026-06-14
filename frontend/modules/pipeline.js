/**
 * Pipeline visualization for progress overlay and static diagram.
 */

export const PIPELINE_STAGES = [
  { id: 'input', name: 'Input: Photo upload', icon: '📸', type: 'normal' },
  { id: 'parse', name: 'Parse: Request & decode', icon: '📋', type: 'normal' },
  { id: 'exif', name: 'EXIF Check: GPS extraction', icon: '🛰️', type: 'checkpoint' },
  { id: 'filename', name: 'Filename Hint: Keyword match', icon: '📝', type: 'checkpoint' },
  { id: 'features', name: 'Feature Extraction: Visual cues', icon: '🔍', type: 'normal' },
  { id: 'inference', name: 'Vision Inference: Ensemble fusion', icon: '🧠', type: 'major' },
  { id: 'optional', name: 'Optional Modules: Validation', icon: '🧩', type: 'normal' },
  { id: 'analysis', name: 'Additional Analysis: CLIP cues', icon: '🔬', type: 'normal' },
  { id: 'reasoning', name: 'Geo-Reasoning: Re-ranking', icon: '⚙️', type: 'major' },
  { id: 'output', name: 'Output: Location predictions', icon: '🎯', type: 'output' },
];

/** Build the mini pipeline rows inside the progress overlay */
export function buildMiniPipeline() {
  const container = document.getElementById('pipelineMini');
  if (!container) return;
  container.innerHTML = PIPELINE_STAGES.map((s) => {
    const typeClass = s.type === 'checkpoint' ? 'pipeline-mini__row--checkpoint' :
      s.type === 'major' ? 'pipeline-mini__row--major' : '';
    return `
      <div class="pipeline-mini__row ${typeClass}" data-mini-stage="${s.id}" id="mini-stage-${s.id}">
        <span class="pipeline-mini__icon">${s.icon}</span>
        <span class="pipeline-mini__name">${s.name}</span>
        <span class="pipeline-mini__status" id="mini-status-${s.id}"></span>
      </div>
    `;
  }).join('');
}

/** Update a mini pipeline row state */
export function updateMiniPipelineStage(stageId, state) {
  const row = document.getElementById(`mini-stage-${stageId}`);
  const status = document.getElementById(`mini-status-${stageId}`);
  if (!row || !status) return;

  row.classList.remove('pipeline-mini__row--active', 'pipeline-mini__row--completed', 'pipeline-mini__row--skipped');
  status.classList.remove('pipeline-mini__status--active', 'pipeline-mini__status--completed', 'pipeline-mini__status--skipped');

  if (state === 'active') {
    row.classList.add('pipeline-mini__row--active');
    status.classList.add('pipeline-mini__status--active');
    status.textContent = 'running…';
    row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } else if (state === 'completed') {
    row.classList.add('pipeline-mini__row--completed');
    status.classList.add('pipeline-mini__status--completed');
    status.textContent = '✓ done';
  } else if (state === 'skipped') {
    row.classList.add('pipeline-mini__row--skipped');
    status.classList.add('pipeline-mini__status--skipped');
    status.textContent = 'skipped';
  } else {
    status.textContent = '';
  }
}

/** Reset all pipeline stages to neutral state */
export function resetPipeline() {
  document.querySelectorAll('.pipeline-stage').forEach((stage) => {
    stage.classList.remove('pipeline-stage--active', 'pipeline-stage--completed', 'pipeline-stage--skipped');
  });
  document.querySelectorAll('.pipeline-node__box').forEach((box) => {
    box.classList.remove('pipeline-node__box--active', 'pipeline-node__box--completed');
  });
  const exec = document.getElementById('pipelineExecution');
  const stageEl = document.getElementById('pipelineExecutionStage');
  const prog = document.getElementById('pipelineExecutionProgress');
  if (exec) exec.style.display = 'none';
  if (stageEl) stageEl.textContent = '—';
  if (prog) prog.style.width = '0%';
  PIPELINE_STAGES.forEach((s) => updateMiniPipelineStage(s.id, 'neutral'));
}

/** Activate a specific pipeline stage (and mark prior stages completed) */
export function activatePipelineStage(stageId) {
  const stageIndex = PIPELINE_STAGES.findIndex((s) => s.id === stageId);
  if (stageIndex < 0) return;

  for (let i = 0; i < stageIndex; i++) {
    const priorId = PIPELINE_STAGES[i].id;
    const priorStage = document.querySelector(`.pipeline-stage[data-stage="${priorId}"]`);
    if (priorStage) {
      priorStage.classList.remove('pipeline-stage--active');
      priorStage.classList.add('pipeline-stage--completed');
    }
    const priorBox = document.querySelector(`.pipeline-stage[data-stage="${priorId}"] .pipeline-node__box`);
    if (priorBox) {
      priorBox.classList.remove('pipeline-node__box--active');
      priorBox.classList.add('pipeline-node__box--completed');
    }
    updateMiniPipelineStage(priorId, 'completed');
  }

  const stage = document.querySelector(`.pipeline-stage[data-stage="${stageId}"]`);
  if (stage) {
    stage.classList.remove('pipeline-stage--completed', 'pipeline-stage--skipped');
    stage.classList.add('pipeline-stage--active');
    stage.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
  const boxes = document.querySelectorAll(`.pipeline-stage[data-stage="${stageId}"] .pipeline-node__box`);
  boxes.forEach((box) => {
    box.classList.remove('pipeline-node__box--completed');
    box.classList.add('pipeline-node__box--active');
  });
  updateMiniPipelineStage(stageId, 'active');

  const exec = document.getElementById('pipelineExecution');
  const stageEl = document.getElementById('pipelineExecutionStage');
  const prog = document.getElementById('pipelineExecutionProgress');
  if (exec) exec.style.display = 'flex';
  if (stageEl) stageEl.textContent = PIPELINE_STAGES[stageIndex].name;
  if (prog) {
    const pct = ((stageIndex + 1) / PIPELINE_STAGES.length) * 100;
    prog.style.width = `${pct}%`;
  }
}

/** Mark a stage as skipped */
export function skipPipelineStage(stageId) {
  const stage = document.querySelector(`.pipeline-stage[data-stage="${stageId}"]`);
  if (stage) {
    stage.classList.remove('pipeline-stage--active', 'pipeline-stage--completed');
    stage.classList.add('pipeline-stage--skipped');
  }
  updateMiniPipelineStage(stageId, 'skipped');
}

/** Mark the entire pipeline as complete */
export function completePipeline() {
  PIPELINE_STAGES.forEach((s) => {
    const stage = document.querySelector(`.pipeline-stage[data-stage="${s.id}"]`);
    if (stage) {
      stage.classList.remove('pipeline-stage--active', 'pipeline-stage--skipped');
      stage.classList.add('pipeline-stage--completed');
    }
    const boxes = document.querySelectorAll(`.pipeline-stage[data-stage="${s.id}"] .pipeline-node__box`);
    boxes.forEach((box) => {
      box.classList.remove('pipeline-node__box--active');
      box.classList.add('pipeline-node__box--completed');
    });
    updateMiniPipelineStage(s.id, 'completed');
  });
  const exec = document.getElementById('pipelineExecution');
  const stageEl = document.getElementById('pipelineExecutionStage');
  const prog = document.getElementById('pipelineExecutionProgress');
  if (exec) exec.style.display = 'flex';
  if (stageEl) stageEl.textContent = 'Complete ✓';
  if (prog) prog.style.width = '100%';
}

/** Show early-return state (EXIF or filename hit) */
export function markEarlyReturn(returnStage) {
  const idx = PIPELINE_STAGES.findIndex((s) => s.id === returnStage);
  if (idx < 0) return;
  for (let i = 0; i <= idx; i++) {
    const s = PIPELINE_STAGES[i];
    const stage = document.querySelector(`.pipeline-stage[data-stage="${s.id}"]`);
    if (stage) {
      stage.classList.remove('pipeline-stage--active', 'pipeline-stage--skipped');
      stage.classList.add('pipeline-stage--completed');
    }
    const boxes = document.querySelectorAll(`.pipeline-stage[data-stage="${s.id}"] .pipeline-node__box`);
    boxes.forEach((box) => {
      box.classList.remove('pipeline-node__box--active');
      box.classList.add('pipeline-node__box--completed');
    });
    updateMiniPipelineStage(s.id, 'completed');
  }
  for (let i = idx + 1; i < PIPELINE_STAGES.length; i++) {
    skipPipelineStage(PIPELINE_STAGES[i].id);
  }
  const exec = document.getElementById('pipelineExecution');
  const stageEl = document.getElementById('pipelineExecutionStage');
  const prog = document.getElementById('pipelineExecutionProgress');
  if (exec) exec.style.display = 'flex';
  if (stageEl) stageEl.textContent = `Early return: ${PIPELINE_STAGES[idx].name}`;
  if (prog) prog.style.width = `${((idx + 1) / PIPELINE_STAGES.length) * 100}%`;
}
