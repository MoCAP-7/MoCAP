(() => {
  const $ = (selector) => document.querySelector(selector);
  const ui = {
    configPath: $('#config-path'), connection: $('#connection'), status: $('#status-pill'),
    start: $('#start-button'), manual: $('#manual-button'), stop: $('#stop-button'), task: $('#task-input'),
    editTask: $('#edit-task'), model: $('#model-input'), temperature: $('#temperature-input'),
    turns: $('#turns-input'), readinessPrior: $('#readiness-prior-input'),
    cameraPanel: $('.camera-panel'), camera: $('#camera-image'),
    cameraEmpty: $('#camera-empty'), frameTurn: $('#frame-turn'), frameTime: $('#frame-time'),
    cameraTitle: $('#camera-title'), cameraDescription: $('#camera-description'),
    llmView: $('#llm-view'), liveView: $('#live-view'),
    imageShape: $('#image-shape'), poseXY: $('#pose-xy'), poseYaw: $('#pose-yaw'),
    velocity: $('#velocity'), safety: $('#safety'), timeline: $('#timeline'),
    timelineEmpty: $('#timeline-empty'), clear: $('#clear-timeline'), tracePath: $('#trace-path'),
    toast: $('#toast'),
    programPanel: $('#program-panel'), programInput: $('#program-input'),
    runProgram: $('#run-program'), programStatus: $('#program-status'),
    reuseProgram: $('#reuse-program'),
    primitiveChips: $('#primitive-chips'), primitiveDocs: $('#primitive-docs'),
    primitiveDocsSummary: $('#primitive-docs-summary'), primitiveDocsText: $('#primitive-docs-text'),
    reviewPanel: $('#review-panel'), reviewSummary: $('#review-summary'), reviewReason: $('#review-reason'),
    reviewNote: $('#review-note'), submitReview: $('#submit-review'),
  };

  let socket = null;
  let reconnectTimer = null;
  let state = 'idle';
  let toastTimer = null;
  let cameraMode = 'llm';
  let modelFrame = null;
  let liveFrame = null;
  let navigationArticle = null;
  let navigationId = null;
  let configReady = false;
  let socketReady = false;
  // The readiness prior can only be switched on when the config names an
  // events JSON (--readiness-events or dock_to_visible_object.readiness_prior).
  let readinessPriorAvailable = false;
  // Manual policy mode: the operator types each turn's program instead of an LLM.
  let manualMode = false;
  let programReady = false;
  let programSubmitting = false;
  let lastProgram = '';
  // Experiment runs: the last episode must be reviewed before the next run.
  let pendingReview = null;
  let reviewSubmitting = false;

  function placeholderFor(param) {
    const annotation = String(param.annotation || '').toLowerCase();
    if (/\bbool\b/.test(annotation)) return 'False';
    if (/\bint\b/.test(annotation) && !/float/.test(annotation)) return '0';
    if (/float/.test(annotation)) return '0.0';
    if (/\bstr\b/.test(annotation)) return '""';
    if (/list|sequence|tuple|ndarray/.test(annotation)) return '[]';
    return '...';
  }

  function callTemplate(item) {
    const parts = [];
    (item.params || []).forEach((param) => {
      if (!param.required) return;
      if (param.kind === 'keyword_only') parts.push(`${param.name}=${placeholderFor(param)}`);
      else parts.push(placeholderFor(param));
    });
    return `${item.name}(${parts.join(', ')})`;
  }

  function insertIntoProgram(text) {
    const input = ui.programInput;
    const start = input.selectionStart ?? input.value.length;
    const end = input.selectionEnd ?? input.value.length;
    const before = input.value.slice(0, start);
    const after = input.value.slice(end);
    const needsNewline = before.length > 0 && !before.endsWith('\n');
    const inserted = `${needsNewline ? '\n' : ''}${text}`;
    input.value = `${before}${inserted}${after}`;
    // Select the first placeholder so typing replaces it.
    const open = text.indexOf('(');
    const close = text.lastIndexOf(')');
    const argStart = before.length + inserted.length - text.length + open + 1;
    if (close > open + 1) {
      const firstArg = text.slice(open + 1, close).split(',')[0];
      const eq = firstArg.indexOf('=');
      const valueOffset = eq >= 0 ? eq + 1 : 0;
      input.setSelectionRange(argStart + valueOffset, argStart + firstArg.length);
    } else {
      input.setSelectionRange(argStart, argStart);
    }
    input.focus();
  }

  function renderPrimitives(items) {
    ui.primitiveChips.innerHTML = '';
    (items || []).forEach((item) => {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = `primitive-chip${item.name === 'finish' ? ' finish' : ''}`;
      chip.textContent = item.name;
      chip.title = `${item.name}${item.signature || '()'}${item.summary ? `\n\n${item.summary}` : ''}`;
      chip.onclick = () => {
        insertIntoProgram(callTemplate(item));
        ui.primitiveDocsSummary.textContent = `${item.name}${item.signature || '()'}`;
        ui.primitiveDocsText.textContent = item.doc || '(no documentation)';
        ui.primitiveDocs.hidden = false;
      };
      ui.primitiveChips.append(chip);
    });
    ui.primitiveChips.hidden = !(items && items.length);
  }

  function setManualMode(enabled) {
    manualMode = Boolean(enabled);
    ui.programPanel.hidden = !manualMode;
    updateProgramControls();
  }

  function setProgramReady(ready, label) {
    programReady = Boolean(ready);
    if (label) ui.programStatus.textContent = label;
    updateProgramControls();
  }

  function updateProgramControls() {
    const running = state === 'running';
    const canRun = manualMode && running && programReady && !programSubmitting;
    ui.runProgram.disabled = !canRun;
    // Typing is always allowed in manual mode so the next program can be
    // prepared while the current one executes; only Run is gated.
    ui.programInput.disabled = !manualMode;
    ui.reuseProgram.disabled = !manualMode || !lastProgram;
    ui.programStatus.className = `program-status${canRun ? ' ready' : (manualMode && running ? ' busy' : '')}`;
    if (!running) ui.programStatus.textContent = manualMode ? 'Waiting for run' : '';
  }

  function showToast(message, isError = false) {
    clearTimeout(toastTimer);
    ui.toast.textContent = message;
    ui.toast.className = `toast show${isError ? ' error' : ''}`;
    toastTimer = setTimeout(() => { ui.toast.className = 'toast'; }, 3600);
  }

  function prettyState(value) {
    return String(value || 'idle').replaceAll('_', ' ');
  }

  function setState(next) {
    state = next || 'idle';
    ui.status.textContent = prettyState(state);
    ui.status.className = `status-pill ${state}`;
    const active = ['starting', 'running', 'stopping'].includes(state);
    ui.start.disabled = active || !configReady || !socketReady || Boolean(pendingReview);
    ui.manual.disabled = ui.start.disabled;
    ui.stop.disabled = !['starting', 'running'].includes(state);
    ui.task.readOnly = active;
    ui.editTask.disabled = active;
    ui.model.disabled = active;
    ui.temperature.disabled = active;
    ui.turns.disabled = active;
    ui.readinessPrior.disabled = active || !readinessPriorAvailable;
    if (state !== 'running') programReady = false;
    updateProgramControls();
  }

  function setModelSelection(provider, name) {
    const value = `${provider || 'vertex'}::${name || ''}`;
    const existing = Array.from(ui.model.options).some((option) => option.value === value);
    if (!existing) {
      const custom = document.createElement('option');
      custom.value = value;
      custom.textContent = `${provider || 'model'} / ${name || 'unnamed'}`;
      ui.model.append(custom);
    }
    ui.model.value = value;
  }

  function selectedModel() {
    const [provider, ...nameParts] = ui.model.value.split('::');
    return { provider, name: nameParts.join('::') };
  }

  function reviewValue(name) {
    const checked = document.querySelector(`input[name="${name}"]:checked`);
    return checked ? checked.value : null;
  }

  function updateReviewControls() {
    const adopt = reviewValue('review-adopt');
    ui.reviewReason.hidden = adopt !== 'false';
    ui.submitReview.disabled = !pendingReview || reviewSubmitting
      || reviewValue('review-success') == null || adopt == null;
  }

  function setReview(review) {
    const changed = (pendingReview?.episode_dir || null) !== (review?.episode_dir || null);
    pendingReview = review || null;
    ui.reviewPanel.hidden = !pendingReview;
    if (changed) {
      document.querySelectorAll('input[name="review-success"], input[name="review-adopt"]')
        .forEach((input) => { input.checked = false; });
      ui.reviewReason.value = '';
      ui.reviewNote.value = '';
    }
    if (pendingReview) {
      const elapsed = pendingReview.elapsed_s == null ? '—' : `${number(pendingReview.elapsed_s, 1)} s`;
      ui.reviewSummary.textContent = `${pendingReview.condition} · ${pendingReview.task_id} @ `
        + `${pendingReview.start_label} · ${prettyState(pendingReview.termination)} · ${elapsed}`;
    }
    updateReviewControls();
    setState(state);
  }

  async function loadConfig() {
    configReady = false;
    setState(state);
    try {
      const response = await fetch('/api/config');
      if (!response.ok) throw new Error(await response.text());
      const config = await response.json();
      ui.configPath.textContent = config.config_path;
      ui.configPath.title = config.config_path;
      ui.task.value = config.task?.instruction || '';
      setModelSelection(config.model?.provider || 'vertex', config.model?.name || '');
      ui.temperature.value = config.model?.temperature ?? 1;
      ui.turns.value = config.agent?.max_turns ?? 30;
      readinessPriorAvailable = Boolean(config.readiness_prior?.events_path);
      ui.readinessPrior.checked = !!config.readiness_prior?.enabled;
      ui.readinessPrior.disabled = !config.readiness_prior?.events_path;
      configReady = true;
      setReview(config.review || null);
      if (config.provider != null) setManualMode(config.provider === 'manual');
      setState(config.state || 'idle');
    } catch (error) {
      configReady = false;
      setState(state);
      showToast(`Could not load config: ${error.message}`, true);
    }
  }

  function connect() {
    clearTimeout(reconnectTimer);
    const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
    const browserHost = location.hostname === '0.0.0.0'
      ? `127.0.0.1${location.port ? `:${location.port}` : ''}`
      : location.host;
    const socketUrl = `${protocol}://${browserHost}/ws`;
    socketReady = false;
    setState(state);
    socket = new WebSocket(socketUrl);
    ui.connection.className = 'connection';
    ui.connection.querySelector('span').textContent = 'Connecting';
    const connectionTimeout = setTimeout(() => {
      if (socket.readyState === WebSocket.CONNECTING) {
        showToast(
          `WebSocket could not connect to ${socketUrl}. Open this console via http://127.0.0.1:${location.port || '8200'} and reload.`,
          true,
        );
        socket.close();
      }
    }, 5000);
    socket.onopen = () => {
      clearTimeout(connectionTimeout);
      socketReady = true;
      setState(state);
      ui.connection.className = 'connection connected';
      ui.connection.querySelector('span').textContent = 'Live';
      loadConfig();
    };
    socket.onmessage = (message) => {
      try { handleEvent(JSON.parse(message.data)); }
      catch (error) { console.error('Bad WebSocket event', error); }
    };
    socket.onclose = () => {
      clearTimeout(connectionTimeout);
      socketReady = false;
      setState(state);
      ui.connection.className = 'connection disconnected';
      ui.connection.querySelector('span').textContent = 'Offline';
      reconnectTimer = setTimeout(connect, 1500);
    };
    socket.onerror = () => socket.close();
  }

  function formatTime(value) {
    if (!value) return '—';
    const date = typeof value === 'number' ? new Date(value * 1000) : new Date(value);
    if (Number.isNaN(date.getTime())) return '—';
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }

  function turnBadge(turn) {
    return turn == null ? '' : `<span class="turn-badge">TURN ${escapeHtml(turn)}</span>`;
  }

  function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"]/g, (character) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;',
    })[character]);
  }

  function removeEmpty() {
    if (ui.timelineEmpty) ui.timelineEmpty.remove();
    ui.timelineEmpty = null;
  }

  function addEvent({ title, body = '', kind = '', turn = null, timestamp, extra = null }) {
    removeEmpty();
    const article = document.createElement('article');
    article.className = `event ${kind}`;
    article.innerHTML = `
      <div class="event-header">
        <div class="event-title">${turnBadge(turn)}${escapeHtml(title)}</div>
        <time class="event-time">${escapeHtml(formatTime(timestamp))}</time>
      </div>
      <div class="event-body">${body}</div>`;
    if (extra) article.querySelector('.event-body').append(extra);
    ui.timeline.append(article);
    ui.timeline.scrollTop = ui.timeline.scrollHeight;
    return article;
  }

  function codeBlock(code) {
    const wrap = document.createElement('div');
    wrap.className = 'code-wrap';
    const bar = document.createElement('div');
    bar.className = 'code-bar';
    bar.innerHTML = '<span>generated_policy.py</span>';
    const copy = document.createElement('button');
    copy.type = 'button';
    copy.className = 'copy-code';
    copy.textContent = 'Copy';
    copy.onclick = async () => {
      await navigator.clipboard.writeText(code);
      copy.textContent = 'Copied';
      setTimeout(() => { copy.textContent = 'Copy'; }, 1200);
    };
    bar.append(copy);
    const pre = document.createElement('pre');
    pre.textContent = code;
    wrap.append(bar, pre);
    return wrap;
  }

  function outputBlock(label, text, isError = false) {
    const block = document.createElement('div');
    block.className = `output${isError ? ' error' : ''}`;
    const heading = document.createElement('div');
    heading.className = 'output-label';
    heading.textContent = label;
    const pre = document.createElement('pre');
    pre.textContent = text || '(empty)';
    block.append(heading, pre);
    return block;
  }

  function showCameraFrame(frame) {
    if (!frame) {
      ui.camera.hidden = true;
      ui.cameraEmpty.hidden = false;
      return;
    }
    ui.camera.src = frame.imageUrl;
    ui.camera.hidden = false;
    ui.cameraEmpty.hidden = true;
    ui.cameraPanel.classList.add('has-frame');
    ui.frameTurn.textContent = frame.label;
    ui.frameTime.textContent = formatTime(frame.timestamp);
  }

  function setCameraMode(mode) {
    cameraMode = mode;
    const exact = mode === 'llm';
    ui.llmView.classList.toggle('active', exact);
    ui.liveView.classList.toggle('active', !exact);
    ui.cameraTitle.textContent = exact ? 'LLM camera input' : 'Live camera preview';
    ui.cameraDescription.textContent = exact
      ? 'The exact resized PNG attached to the next model request'
      : 'A read-only 2 fps preview; this stream is not sent to the model';
    showCameraFrame(exact ? modelFrame : liveFrame);
  }

  function updateCamera(event) {
    const observation = event.observation || {};
    if (event.image_url) {
      modelFrame = {
        imageUrl: event.image_url,
        label: `Next request · turn ${event.turn}`,
        timestamp: event.timestamp,
      };
      if (cameraMode === 'llm') showCameraFrame(modelFrame);
    }
    ui.imageShape.textContent = observation.rgb_shape?.join(' × ') || '—';
    const pose = observation.base?.pose_xy_yaw || [];
    ui.poseXY.textContent = pose.length >= 2 ? `${number(pose[0], 3)} / ${number(pose[1], 3)} m` : '—';
    ui.poseYaw.textContent = pose.length >= 3 && pose[2] != null
      ? `${number(pose[2], 3)} rad · ${number(pose[2] * 180 / Math.PI, 1)}°` : '—';
    const velocity = observation.base?.last_velocity || [];
    ui.velocity.textContent = velocity.length ? `[${velocity.map((v) => number(v, 2)).join(', ')}]` : '—';
    const estop = observation.base?.estop_latched || observation.arms?.estop_latched;
    ui.safety.textContent = estop ? 'E-stop latched' : 'E-stop clear';
    ui.safety.className = estop ? 'unsafe' : 'safe';
  }

  function number(value, digits) {
    return value == null || Number.isNaN(Number(value)) ? '—' : Number(value).toFixed(digits);
  }

  function handleEvent(event) {
    switch (event.type) {
      case 'session_state':
        if (event.provider != null) setManualMode(event.provider === 'manual');
        setState(event.state);
        if (Object.prototype.hasOwnProperty.call(event, 'review')) setReview(event.review);
        if (event.trace_path) {
          ui.tracePath.textContent = event.trace_path;
          ui.tracePath.title = event.trace_path;
        }
        if (event.state === 'error' && event.result?.reason) showToast(event.result.reason, true);
        break;
      case 'run_started':
        setManualMode(event.provider === 'manual');
        setState('running');
        ui.task.value = event.task || ui.task.value;
        if (event.trace_path) {
          ui.tracePath.textContent = event.trace_path;
          ui.tracePath.title = event.trace_path;
        }
        addEvent({ title: 'Run initialized', kind: 'execution', timestamp: event.timestamp,
          body: `<strong>${escapeHtml(event.provider ? `${event.provider}/${event.model}` : (event.model || 'model'))}</strong> · up to ${escapeHtml(event.max_turns)} turns` });
        break;
      case 'environment_status':
        addEvent({ title: event.status === 'ready' ? 'Robot observation ready' : 'Initializing robot environment',
          kind: 'execution', timestamp: event.timestamp,
          body: event.status === 'ready' ? 'Base checks passed and the first synchronized frame was captured.' : 'Connecting to the base RPC and atomic ZED stream…' });
        break;
      case 'model_input':
        updateCamera(event);
        {
          const article = addEvent({ title: 'Model input prepared', kind: 'model', turn: event.turn, timestamp: event.timestamp,
          body: `Current RGB frame and structured robot state attached to the request.` });
          if (event.prompt_text) {
            const details = document.createElement('details');
            details.className = 'prompt-details';
            const summary = document.createElement('summary');
            summary.textContent = 'View prompt text sent this turn';
            const pre = document.createElement('pre');
            pre.textContent = event.prompt_text;
            details.append(summary, pre);
            article.querySelector('.event-body').append(details);
          }
        }
        break;
      case 'camera_preview':
        liveFrame = {
          imageUrl: event.image_url,
          label: 'Live ZED preview · 2 fps',
          timestamp: event.timestamp,
        };
        if (cameraMode === 'live') showCameraFrame(liveFrame);
        break;
      case 'turn_started':
        break;
      case 'model_query_started':
        if (manualMode) {
          setProgramReady(true, `Turn ${event.turn ?? '—'} · ready for a program`);
          addEvent({ title: 'Waiting for operator program', kind: 'model', turn: event.turn, timestamp: event.timestamp,
            body: 'Type a program in the Program panel and press Run program.' });
        } else {
          addEvent({ title: 'Generating policy', kind: 'model', turn: event.turn, timestamp: event.timestamp,
            body: 'Waiting for the model response…' });
        }
        break;
      case 'primitives':
        renderPrimitives(event.items);
        break;
      case 'policy_submitted': {
        setProgramReady(false, 'Program queued');
        const article = addEvent({ title: 'Operator program queued', kind: 'model', timestamp: event.timestamp,
          body: event.pending > 1 ? `${escapeHtml(event.pending)} programs are queued.` : 'Handed to the agent loop as this turn’s policy.' });
        article.querySelector('.event-body').append(codeBlock(event.code || ''));
        break;
      }
      case 'model_response': {
        const article = addEvent({ title: manualMode ? 'Operator program accepted' : 'Generated policy', kind: 'model', turn: event.turn,
          timestamp: event.timestamp, body: manualMode ? 'The program parsed as one executable Python policy.' : 'The model returned one executable Python policy.' });
        article.querySelector('.event-body').append(codeBlock(event.code || ''));
        break;
      }
      case 'model_format_error':
        addEvent({ title: manualMode ? 'Operator program rejected' : 'Model response rejected', kind: 'warning', turn: event.turn,
          timestamp: event.timestamp, body: escapeHtml(event.message) });
        break;
      case 'policy_execution_started':
        if (manualMode) setProgramReady(false, `Turn ${event.turn ?? '—'} · executing`);
        addEvent({ title: 'Executing policy', kind: 'execution', turn: event.turn,
          timestamp: event.timestamp, body: 'Running in the capability-limited policy namespace.' });
        break;
      case 'primitive_call': {
        const failed = Boolean(event.error) || event.result?.success === false;
        const result = event.result == null ? '—' : JSON.stringify(event.result);
        const extra = document.createElement('div');
        extra.className = 'primitive-grid';
        const rows = [
          ['Arguments', JSON.stringify({ args: event.args || [], kwargs: event.kwargs || {} })],
          ['Result', result],
          ['Elapsed', `${number(event.elapsed_s, 3)} s`],
        ];
        rows.forEach(([label, value]) => {
          const key = document.createElement('span'); key.textContent = label;
          const content = document.createElement('span'); content.textContent = value;
          extra.append(key, content);
        });
        const article = addEvent({ title: event.name || 'Primitive call', kind: failed ? 'error' : 'primitive',
          turn: event.turn, timestamp: event.timestamp, body: '', extra });
        const chip = document.createElement('span');
        chip.className = `result-chip${failed ? ' failed' : ''}`;
        chip.textContent = failed ? 'Primitive failed' : 'Completed';
        article.querySelector('.event-body').append(chip);
        break;
      }
      case 'primitive_progress': {
        const id = event.navigation_id || `${event.primitive || 'primitive'}-${event.turn || 0}`;
        if (!navigationArticle || navigationId !== id) {
          navigationArticle = addEvent({ title: event.primitive || 'Primitive progress',
            kind: 'primitive', turn: event.turn, timestamp: event.timestamp, body: '' });
          navigationId = id;
        }
        const feedback = event.feedback || {};
        const values = [];
        if (feedback.distance_remaining_m != null) {
          values.push(`${number(feedback.distance_remaining_m, 2)} m remaining`);
        }
        if (feedback.estimated_time_remaining_s != null) {
          values.push(`ETA ${number(feedback.estimated_time_remaining_s, 1)} s`);
        }
        if (feedback.number_of_recoveries != null) {
          values.push(`${feedback.number_of_recoveries} recoveries`);
        }
        const stateText = prettyState(event.state);
        navigationArticle.querySelector('.event-title').innerHTML =
          `${turnBadge(event.turn)}${escapeHtml(event.target ? `Navigate to ${event.target}` : (event.primitive || 'Primitive progress'))}`;
        navigationArticle.querySelector('.event-body').innerHTML =
          `<span class="result-chip${event.success === false ? ' failed' : ''}">${escapeHtml(stateText)}</span>`
          + (values.length ? ` <span>${escapeHtml(values.join(' · '))}</span>` : '')
          + (event.reason ? `<br><span>${escapeHtml(event.reason)}</span>` : '');
        if (event.terminal) {
          navigationArticle.classList.toggle('error', event.success === false);
          navigationArticle = null;
          navigationId = null;
        }
        break;
      }
      case 'policy_execution_finished': {
        const execution = event.execution || {};
        const failed = Boolean(execution.error || execution.interrupted_by);
        const article = addEvent({ title: failed ? 'Policy returned feedback' : 'Policy execution complete',
          kind: failed ? 'warning' : 'execution', turn: event.turn, timestamp: event.timestamp,
          body: `${number(execution.elapsed_s, 3)} s elapsed${execution.finish_reason ? ` · finish: ${escapeHtml(execution.finish_reason)}` : ''}` });
        const body = article.querySelector('.event-body');
        if (execution.stdout) body.append(outputBlock('stdout', execution.stdout));
        if (execution.stderr) body.append(outputBlock('stderr', execution.stderr, true));
        if (execution.error) body.append(outputBlock('exception', `${execution.error.type}: ${execution.error.message}\n${execution.error.traceback || ''}`, true));
        if (execution.interrupted_by) body.append(outputBlock('primitive interruption', JSON.stringify(execution.interrupted_by, null, 2), true));
        break;
      }
      case 'run_finished':
        setState(event.status);
        addEvent({ title: event.status === 'stopped' ? 'Run stopped by operator' : 'Run ended',
          kind: event.status === 'stopped' ? 'warning' : 'execution', timestamp: event.timestamp,
          body: `<strong>${escapeHtml(event.reason)}</strong> · ${escapeHtml(event.turns)} turn${event.turns === 1 ? '' : 's'}<br><span style="color:var(--faint)">This is a loop stop reason, not a verified physical-task result.</span>` });
        break;
      case 'run_error':
        setState('error');
        addEvent({ title: event.error_type || 'Run error', kind: 'error', timestamp: event.timestamp,
          body: escapeHtml(event.message) });
        showToast(event.message || 'The run failed.', true);
        break;
      case 'episode_reviewed': {
        setReview(null);
        const success = event.task_success == null ? 'not judged' : (event.task_success ? 'yes' : 'no');
        addEvent({ title: 'Episode review saved', kind: 'execution', timestamp: event.timestamp,
          body: `${event.adopted ? 'Adopted' : 'Excluded'} · success: ${escapeHtml(success)}<br><span>${escapeHtml(event.result_path)}</span>` });
        break;
      }
      default:
        break;
    }
  }

  async function post(path, body) {
    const response = await fetch(path, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: body == null ? undefined : JSON.stringify(body),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
    return data;
  }

  ui.start.onclick = async () => {
    try {
      const model = selectedModel();
      const temperature = ui.temperature.value.trim();
      const maxTurns = ui.turns.value.trim();
      await post('/api/start', {
        instruction: ui.task.value,
        provider: model.provider,
        model: model.name,
        temperature: temperature === '' ? null : Number(temperature),
        max_turns: maxTurns === '' ? null : Number(maxTurns),
        readiness_prior: ui.readinessPrior.checked,
      });
    } catch (error) { showToast(error.message, true); }
  };

  ui.manual.onclick = async () => {
    // Debug run: same launch command and task, but the operator types each
    // program. The navigation planner is skipped so no API call is needed.
    try {
      const maxTurns = ui.turns.value.trim();
      setManualMode(true);
      await post('/api/start', {
        instruction: ui.task.value,
        provider: 'manual',
        model: 'operator',
        temperature: null,
        max_turns: maxTurns === '' ? null : Number(maxTurns),
        navigation_planner: false,
        readiness_prior: ui.readinessPrior.checked,
      });
    } catch (error) { showToast(error.message, true); }
  };

  ui.stop.onclick = async () => {
    try {
      ui.stop.disabled = true;
      const result = await post('/api/stop');
      if (result.stop_error) showToast(`Stop requested; confirmation warning: ${result.stop_error}`, true);
      else showToast('Normal stop requested. Waiting for zero confirmation.');
    } catch (error) {
      showToast(error.message, true);
      ui.stop.disabled = false;
    }
  };

  document.querySelectorAll('input[name="review-success"], input[name="review-adopt"]').forEach((input) => {
    input.onchange = updateReviewControls;
  });

  ui.submitReview.onclick = async () => {
    if (ui.submitReview.disabled) return;
    const success = reviewValue('review-success');
    reviewSubmitting = true;
    updateReviewControls();
    try {
      await post('/api/review', {
        task_success: success === 'null' ? null : success === 'true',
        adopted: reviewValue('review-adopt') === 'true',
        exclusion_reason: ui.reviewReason.value,
        note: ui.reviewNote.value,
      });
      showToast('Episode review saved.');
    } catch (error) {
      showToast(error.message, true);
    } finally {
      reviewSubmitting = false;
      updateReviewControls();
    }
  };

  async function submitProgram() {
    const code = ui.programInput.value;
    if (!code.trim()) { showToast('Type a program first.', true); return; }
    if (ui.runProgram.disabled) return;
    programSubmitting = true;
    updateProgramControls();
    try {
      await post('/api/policy', { code });
      // Clear the editor so Ctrl/Cmd+Enter cannot re-run a motion by accident;
      // "Reuse last" restores it explicitly.
      lastProgram = code;
      ui.programInput.value = '';
      programReady = false;
      showToast('Program queued for this turn.');
    } catch (error) {
      showToast(error.message, true);
    } finally {
      programSubmitting = false;
      updateProgramControls();
      ui.programInput.focus();
    }
  }

  ui.runProgram.onclick = submitProgram;
  ui.reuseProgram.onclick = () => {
    if (!lastProgram) return;
    ui.programInput.value = lastProgram;
    ui.programInput.focus();
  };
  ui.model.onchange = () => {
    // Preview the Program panel as soon as the operator picks Manual; the
    // server-side provider takes over once a run starts.
    if (!['starting', 'running', 'stopping'].includes(state)) {
      setManualMode(selectedModel().provider === 'manual');
    }
  };
  ui.programInput.addEventListener('keydown', (keyboardEvent) => {
    if ((keyboardEvent.ctrlKey || keyboardEvent.metaKey) && keyboardEvent.key === 'Enter') {
      keyboardEvent.preventDefault();
      submitProgram();
    }
  });

  ui.editTask.onclick = () => {
    const editing = ui.task.readOnly;
    ui.task.readOnly = !editing;
    ui.editTask.textContent = editing ? 'Done' : 'Edit';
    if (editing) ui.task.focus();
  };

  ui.llmView.onclick = () => setCameraMode('llm');
  ui.liveView.onclick = () => setCameraMode('live');

  ui.clear.onclick = () => {
    ui.timeline.innerHTML = `<div id="timeline-empty" class="timeline-empty">
      <div class="empty-lines"><i></i><i></i><i></i></div>
      <strong>Timeline cleared locally</strong><span>New runtime events will continue to appear here.</span></div>`;
    ui.timelineEmpty = $('#timeline-empty');
  };

  setState('idle');
  loadConfig();
  connect();
})();
