const state = {
  csrfHeaderName: "",
  csrfToken: "",
  services: [],
  serviceConfigSummaries: {},
  currentServiceConfig: null,
  pageCapabilities: {},
  applicationIdentity: {},
  latestCognitionGraph: null,
  debugCognitionGraph: null,
  debugRequestInFlight: false,
  eventSource: null,
  streamUrl: "",
  logEventSource: null,
  logStreamUrl: "",
  logRows: [],
  pendingLogRows: null,
  logStreamRevision: 0,
  logPaused: false,
  logDroppedLocal: 0,
  cognitionGraphPins: {},
  isAuthenticated: false,
};

const THEME_STORAGE_KEY = "kazusa-control-theme";
const LOG_ROW_LIMIT = 500;
const GRAPH_STALE_AFTER_MS = 10000;
const ENDPOINT_CONFLICT_MESSAGE = "configured endpoint is already in use by an unmanaged process";
const LEGACY_THEME_NAMES = {
  expo: "dark",
  ollama: "bright",
};

function qs(selector) {
  return document.querySelector(selector);
}

function qsa(selector) {
  return Array.from(document.querySelectorAll(selector));
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setPage(name) {
  if (!state.isAuthenticated && name !== "overview") return;
  const targetLink = qsa("[data-page-link]").find((link) => link.dataset.pageLink === name);
  if (targetLink && targetLink.disabled) return;
  qsa("[data-page]").forEach((page) => page.classList.toggle("active", page.dataset.page === name));
  qsa("[data-page-link]").forEach((link) => link.classList.toggle("active", link.dataset.pageLink === name));
  if (name === "logs" && state.csrfHeaderName) {
    renderLogControls();
    openLogStream();
  } else {
    closeLogStream();
  }
  if (name === "audit" && state.csrfHeaderName) refreshAudit().catch(reportActionError);
  if (name === "character" && state.csrfHeaderName) refreshCharacter().catch(reportActionError);
  if (name === "memory" && state.csrfHeaderName) refreshMemory(false).catch(reportActionError);
  if (name === "style" && state.csrfHeaderName) refreshStyle(false).catch(reportActionError);
  if (name === "calendar" && state.csrfHeaderName) refreshCalendar().catch(reportActionError);
  if (name === "background" && state.csrfHeaderName) refreshBackground().catch(reportActionError);
}

function setAuthState(isAuthenticated) {
  state.isAuthenticated = isAuthenticated;
  document.body.dataset.authState = isAuthenticated ? "authenticated" : "locked";
  qsa("[data-page-link]").forEach((link) => {
    link.disabled = !isAuthenticated;
    if (isAuthenticated) link.removeAttribute("aria-disabled");
    else link.setAttribute("aria-disabled", "true");
  });
  renderPageCapabilities();
}

function renderPageCapabilities() {
  qsa("[data-page-link]").forEach((link) => {
    const capability = state.pageCapabilities[link.dataset.pageLink] || {};
    const status = capability.status || "unknown";
    const disabled = !state.isAuthenticated || status === "disabled";
    link.disabled = disabled;
    link.dataset.capabilityStatus = status;
    link.title = capability.reason || "";
    if (disabled) link.setAttribute("aria-disabled", "true");
    else link.removeAttribute("aria-disabled");
  });
}

function renderBrand(identity = {}) {
  const name = identity.character_name || "not connected";
  const connected = identity.status === "available" && name !== "not connected";
  qs("#brand-name").textContent = name;
  qs("#brand-subtitle").textContent = connected ? "Control console" : "database not connected";
  document.title = connected ? `${name} Control Console` : "not connected";
}

function setTheme(name) {
  const theme = name === "dark" ? "dark" : "bright";
  document.body.dataset.theme = theme;
  qsa("[data-theme-choice]").forEach((button) => button.classList.toggle("active", button.dataset.themeChoice === theme));
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    // Theme persistence is optional; visual switching still works without it.
  }
}

function initializeTheme() {
  let storedTheme = "";
  try {
    storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY) || "";
  } catch {
    storedTheme = "";
  }
  setTheme(LEGACY_THEME_NAMES[storedTheme] || storedTheme);
}

function badgeClass(status) {
  if (status === "running" || status === "healthy") return "badge success";
  if (["conflict", "crashed", "unhealthy"].includes(status)) return "badge danger";
  if (status === "starting" || status === "stopping") return "badge warn";
  return "badge";
}

function renderShellStatus(payload) {
  const brainService = serviceById("brain");
  const brainState = brainService ? brainService.actual_state : "unavailable";
  const dot = qs(".status-dot");
  const statusText = qs("#shell-status-text");
  if (!state.isAuthenticated) {
    dot.dataset.state = "locked";
    statusText.textContent = "Sign in to inspect local services.";
    return;
  }

  dot.dataset.state = brainState;
  const streamState = payload.ui_capabilities.event_stream ? "stream ready" : "stream off";
  if (brainState === "running") {
    statusText.textContent = `Brain running; ${streamState}.`;
    return;
  }
  if (isEndpointConflict(brainService)) {
    statusText.textContent = "Brain endpoint already running outside the console; lifecycle is unmanaged.";
    return;
  }
  if (brainState === "conflict") {
    statusText.textContent = "Brain has a stale lifecycle conflict; inspect Services.";
    return;
  }
  if (brainState === "unavailable") {
    statusText.textContent = "Brain unavailable; check the service registry.";
    return;
  }
  statusText.textContent = `Brain ${brainState}; lifecycle controls available.`;
}

function isEndpointConflict(service) {
  return Boolean(service)
    && service.actual_state === "conflict"
    && service.last_error_preview === ENDPOINT_CONFLICT_MESSAGE;
}

function isServiceHttpAvailable(service) {
  return Boolean(service)
    && (service.actual_state === "running" || isEndpointConflict(service));
}

function renderDebugAvailability() {
  const brainService = serviceById("brain");
  const brainState = brainService ? brainService.actual_state : "unavailable";
  const available = state.isAuthenticated && isServiceHttpAvailable(brainService);
  const statusBadge = qs("#debug-brain-status");
  statusBadge.textContent = isEndpointConflict(brainService)
    ? "brain unmanaged"
    : `brain ${brainState}`;
  statusBadge.className = available ? "badge success" : "badge";
  qsa("[data-debug-input]").forEach((control) => {
    control.disabled = !available;
  });
  qs("[name='message_text']").placeholder = available
    ? "Send a debug message through /chat"
    : "Start or connect the brain service before sending a debug message";
  qs("#debug-send").disabled = !available || state.debugRequestInFlight;
}

function showNotice(message, tone = "info") {
  const notice = qs("#ui-notice");
  notice.hidden = false;
  notice.dataset.tone = tone;
  notice.textContent = message;
}

function clearNotice() {
  const notice = qs("#ui-notice");
  notice.hidden = true;
  notice.dataset.tone = "idle";
  notice.textContent = "";
}

async function runButtonAction(button, loadingMessage, successMessage, action) {
  button.disabled = true;
  showNotice(loadingMessage, "info");
  try {
    await action();
    if (successMessage) showNotice(successMessage, "success");
  } catch (error) {
    showNotice(error.message, "danger");
  } finally {
    button.disabled = false;
  }
}

function reportActionError(error) {
  showNotice(error.message, "danger");
}

async function api(path, options = {}) {
  const headers = {"content-type": "application/json", ...(options.headers || {})};
  if (options.csrf && state.csrfHeaderName) headers[state.csrfHeaderName] = state.csrfToken;
  const response = await fetch(path, {...options, headers});
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      const detail = payload.detail;
      if (typeof detail === "string") message = `${response.status} ${detail}`;
      else if (detail && typeof detail.message === "string") message = `${response.status} ${detail.message}`;
    } catch {
      // Keep the HTTP status when the response body is not JSON.
    }
    throw new Error(message);
  }
  return response.json();
}

async function login() {
  const token = qs("#token").value;
  const payload = await api("/api/auth/login", {method: "POST", body: JSON.stringify({token})});
  state.csrfHeaderName = payload.csrf_header_name;
  state.csrfToken = payload.csrf_token;
  qs("#token").value = "";
  setAuthState(true);
  qs("#session-state").textContent = payload.operator.operator_id;
  await bootstrap();
  showNotice("Signed in.", "success");
}

async function bootstrap(options = {}) {
  clearNotice();
  const reconnectStream = options.reconnectStream !== false;
  const payload = await api("/api/bootstrap");
  state.csrfHeaderName = payload.csrf_header_name || "";
  state.csrfToken = payload.csrf_token || "";
  state.services = payload.services;
  state.serviceConfigSummaries = payload.service_config_summaries || {};
  state.pageCapabilities = payload.page_capabilities || {};
  state.applicationIdentity = payload.application_identity || {};
  state.latestCognitionGraph = payload.latest_cognition_graph || payload.overview?.latest_cognition_graph || null;
  setAuthState(true);
  qs("#session-state").textContent = payload.operator ? payload.operator.operator_id : "signed in";
  renderBrand(payload.application_identity || {});
  renderPageCapabilities();
  renderShellStatus(payload);
  renderDebugAvailability();
  renderOverview(payload);
  renderHealth(payload.overview || {});
  renderServices();
  renderLogControls();
  renderAudit(payload.recent_audit_events || []);
  if (reconnectStream) openStream(payload.stream_url);
}

function lockSession() {
  state.csrfHeaderName = "";
  state.csrfToken = "";
  state.services = [];
  state.serviceConfigSummaries = {};
  state.currentServiceConfig = null;
  state.pageCapabilities = {};
  state.latestCognitionGraph = null;
  state.debugCognitionGraph = null;
  if (state.eventSource) state.eventSource.close();
  closeLogStream();
  state.eventSource = null;
  state.streamUrl = "";
  setAuthState(false);
  qs("#session-state").textContent = "signed out";
  renderBrand({status: "unavailable", character_name: "not connected"});
  renderDebugAvailability();
  showNotice("Sign in to inspect local services.", "info");
}

async function resumeSession() {
  try {
    const session = await api("/api/auth/session");
    if (!session.authenticated) {
      lockSession();
      return;
    }
    state.csrfHeaderName = session.csrf_header_name || "";
    state.csrfToken = session.csrf_token || "";
    await bootstrap();
  } catch {
    lockSession();
  }
}

function renderOverview(payload) {
  const grid = qs("#overview-grid");
  grid.innerHTML = "";
  const runningCount = payload.services.filter((service) => service.actual_state === "running").length;
  const visibleWorkflowCount = qsa("[data-page-link]").length;
  const cards = [
    ["Brain service", serviceStatus("brain"), "local child process"],
    ["Managed services", `${runningCount} / ${payload.services.length}`, "registry declared"],
    ["Event stream", payload.ui_capabilities.event_stream ? "ready" : "off", payload.stream_url],
    ["Visible workflows", visibleWorkflowCount, "primary navigation"],
  ];
  cards.forEach(([label, value, note]) => {
    grid.insertAdjacentHTML("beforeend", `<article class="metric" data-component="Card"><div class="metric-label">${escapeHtml(label)}</div><div class="metric-value">${escapeHtml(value)}</div><div class="metric-label">${escapeHtml(note)}</div></article>`);
  });
  qs("#overview-runtime-table").innerHTML = [
    ["Services", payload.services.length],
    ["Audit events", payload.recent_audit_events.length],
    ["CSRF header", payload.csrf_header_name],
    ["Stream URL", payload.stream_url],
  ].map(([key, value]) => `<tr><td>${escapeHtml(key)}</td><td>${escapeHtml(value)}</td></tr>`).join("");
  qs("#overview-audit-table").innerHTML = auditRows(payload.recent_audit_events);
  renderCapabilitySummary();
  renderOverviewCognitionGraph(state.latestCognitionGraph);
}

function renderOverviewCognitionGraph(snapshot) {
  renderCognitionGraph({
    containerSelector: "#overview-cognition-graph",
    statusSelector: "#overview-cognition-status",
    snapshot,
    emptyMessage: "No latest cognition graph has been reported by the brain.",
  });
}

function renderDebugCognitionGraph(snapshot) {
  renderCognitionGraph({
    containerSelector: "#debug-cognition-graph",
    statusSelector: "#debug-cognition-status",
    snapshot,
    emptyMessage: "No debug cognition graph has been reported for this turn.",
  });
}

function renderCognitionGraph({containerSelector, statusSelector, snapshot, emptyMessage}) {
  const container = qs(containerSelector);
  const status = qs(statusSelector);
  if (!container || !status) return;

  const graph = snapshot || {};
  const nodes = Array.isArray(graph.nodes) ? graph.nodes : [];
  const edges = Array.isArray(graph.edges) ? graph.edges : [];
  const graphStatus = graph.status || "not_reported";
  status.textContent = graphStatus.replaceAll("_", " ");
  status.className = cognitionGraphStatusBadgeClass(graphStatus);

  if (!nodes.length) {
    container.innerHTML = `<p class="graph-empty">${escapeHtml(emptyMessage)}</p>`;
    return;
  }

  const lanes = cognitionGraphLanes(nodes);
  const maxColumn = nodes.reduce((maximum, node) => Math.max(maximum, Number(node.column) || 1), 1);
  const model = cognitionGraphModel({graph, nodes, edges, lanes, maxColumn});
  container.innerHTML = `
    <div class="cognition-graph-shell" data-graph-source="${escapeHtml(model.source)}" data-graph-run-id="${escapeHtml(model.runId)}" data-graph-current-node-id="${escapeHtml(model.currentNode?.id || "")}" data-graph-selected-node-id="${escapeHtml(model.selectedNode?.id || "")}">
      ${cognitionGraphSummaryMarkup(model)}
      <div class="graph-body">
        ${cognitionGraphStageMarkup(model)}
        ${cognitionGraphInspectorMarkup(model)}
      </div>
    </div>
  `;
  container.querySelectorAll("[data-graph-node]").forEach((button) => {
    button.addEventListener("click", () => {
      setCognitionGraphPinnedNode(model.source, model.runId, button.dataset.nodeId || "");
      renderCognitionGraph({containerSelector, statusSelector, snapshot, emptyMessage});
    });
  });
  const returnButton = container.querySelector("[data-graph-return-current]");
  if (returnButton) {
    returnButton.addEventListener("click", () => {
      setCognitionGraphPinnedNode(model.source, model.runId, "");
      renderCognitionGraph({containerSelector, statusSelector, snapshot, emptyMessage});
    });
  }
}

function cognitionGraphLanes(nodes) {
  const preferred = ["input", "cognition", "memory", "decision", "surface"];
  const seen = new Set();
  const lanes = [];
  preferred.forEach((lane) => {
    if (nodes.some((node) => node.lane === lane)) {
      seen.add(lane);
      lanes.push(lane);
    }
  });
  nodes.forEach((node) => {
    const lane = node.lane || "cognition";
    if (!seen.has(lane)) {
      seen.add(lane);
      lanes.push(lane);
    }
  });
  return lanes.length ? lanes : ["cognition"];
}

function cognitionGraphModel({graph, nodes, edges, lanes, maxColumn}) {
  const source = graph.source || "overview_latest";
  const runId = graph.run_id || "run id not reported";
  const currentNode = cognitionGraphCurrentNode(nodes, lanes);
  const highlightedIds = new Set(
    nodes.filter((node) => node.status === "running").map((node) => node.id),
  );
  if (currentNode) highlightedIds.add(currentNode.id);
  const pinnedNodeId = cognitionGraphPinnedNodeId(source, runId, nodes);
  const selectedNode = nodes.find((node) => node.id === pinnedNodeId) || currentNode || nodes[0];
  const generatedAt = Date.parse(graph.generated_at || "");
  const ageMs = Number.isFinite(generatedAt) ? Math.max(0, Date.now() - generatedAt) : null;
  const stale = graph.status === "running" && ageMs !== null && ageMs > GRAPH_STALE_AFTER_MS;
  const focusKind = cognitionGraphFocusKind(graph.status, currentNode);
  return {
    graph,
    nodes,
    edges,
    lanes,
    maxColumn,
    source,
    runId,
    currentNode,
    selectedNode,
    highlightedIds,
    pinned: Boolean(pinnedNodeId),
    ageMs,
    stale,
    focusKind,
    freshness: cognitionGraphFreshnessLabel(ageMs, stale),
    latestEvent: cognitionGraphLatestEvent(currentNode || selectedNode, edges, nodes),
  };
}

function cognitionGraphFocusKind(graphStatus, node) {
  if (!node) return "selected";
  if (graphStatus === "completed" && node.status === "completed") return "final";
  if (node.status === "failed") return "failed";
  if (node.status === "running") return "current";
  if (node.status === "skipped") return "terminated";
  return "selected";
}

function cognitionGraphCurrentNode(nodes, lanes) {
  const running = nodes.filter((node) => node.status === "running");
  if (running.length) return cognitionGraphFurthestNode(running, lanes);
  const failed = nodes.filter((node) => node.status === "failed");
  if (failed.length) return cognitionGraphFurthestNode(failed, lanes);
  const completed = nodes.filter((node) => node.status === "completed");
  if (completed.length) return cognitionGraphFurthestNode(completed, lanes);
  const pending = nodes.filter((node) => node.status === "pending" || node.status === "skipped");
  if (pending.length) return cognitionGraphEarliestNode(pending, lanes);
  return nodes[0] || null;
}

function cognitionGraphFurthestNode(nodes, lanes) {
  return [...nodes].sort((left, right) => {
    const columnDelta = (Number(right.column) || 1) - (Number(left.column) || 1);
    if (columnDelta) return columnDelta;
    return cognitionGraphLaneIndex(left, lanes) - cognitionGraphLaneIndex(right, lanes);
  })[0] || null;
}

function cognitionGraphEarliestNode(nodes, lanes) {
  return [...nodes].sort((left, right) => {
    const columnDelta = (Number(left.column) || 1) - (Number(right.column) || 1);
    if (columnDelta) return columnDelta;
    return cognitionGraphLaneIndex(left, lanes) - cognitionGraphLaneIndex(right, lanes);
  })[0] || null;
}

function cognitionGraphLaneIndex(node, lanes) {
  const lane = node.lane || "cognition";
  const index = lanes.indexOf(lane);
  return index >= 0 ? index : lanes.length;
}

function cognitionGraphPinnedNodeId(source, runId, nodes) {
  const pin = state.cognitionGraphPins[source];
  if (!pin || pin.runId !== runId) return "";
  return nodes.some((node) => node.id === pin.nodeId) ? pin.nodeId : "";
}

function setCognitionGraphPinnedNode(source, runId, nodeId) {
  if (!nodeId) {
    delete state.cognitionGraphPins[source];
    return;
  }
  state.cognitionGraphPins[source] = {runId, nodeId};
}

function cognitionGraphStatusBadgeClass(status) {
  const label = cognitionGraphStatusLabel(status);
  if (label === "completed") return "badge success";
  if (label === "failed") return "badge danger";
  if (label === "running" || label === "partial") return "badge warn";
  if (label === "terminated") return "badge terminal";
  if (label === "pending") return "badge pending";
  return "badge";
}

function cognitionGraphStatusLabel(status) {
  if (status === "skipped" || status === "terminated") return "terminated";
  return String(status || "not_reported").replaceAll("_", " ");
}

function cognitionGraphFreshnessLabel(ageMs, stale) {
  if (ageMs === null) return "timestamp not reported";
  const age = cognitionGraphAgeLabel(ageMs);
  return stale ? `stale · updated ${age} ago` : `updated ${age} ago`;
}

function cognitionGraphAgeLabel(ageMs) {
  const seconds = Math.floor(ageMs / 1000);
  if (seconds < 1) return "just now";
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h`;
}

function cognitionGraphLatestEvent(node, edges, nodes) {
  if (!node) return "No cognition node selected.";
  const detail = node.detail || {};
  const summary = detail.summary || detail.conclusion || detail.decision || detail.status || detail.blocker;
  const outgoing = edges
    .filter((edge) => edge.source === node.id)
    .map((edge) => nodes.find((candidate) => candidate.id === edge.target))
    .filter(Boolean);
  const nextRunning = outgoing.find((candidate) => candidate.status === "running");
  const nextPending = outgoing.find((candidate) => candidate.status === "pending");
  if (nextRunning) return `${node.label || node.id} advanced to ${nextRunning.label || nextRunning.id}.`;
  if (nextPending) return `${node.label || node.id} is waiting for ${nextPending.label || nextPending.id}.`;
  if (summary) return `${node.label || node.id}: ${cognitionGraphValue(summary)}`;
  return `${node.label || node.id} is ${String(node.status || "not reported").replaceAll("_", " ")}.`;
}

function cognitionGraphSummaryMarkup(model) {
  const current = model.currentNode;
  const status = cognitionGraphStatusLabel(model.graph.status || "not_reported");
  const currentLabel = current
    ? `${model.focusKind} · ${current.stage || "stage"} · ${current.label || current.id}`
    : "no current node";
  return `
    <div class="graph-run-summary">
      <div class="graph-run-title">
        <strong>${escapeHtml(model.runId)}</strong>
        <span>${escapeHtml(model.source.replaceAll("_", " "))}</span>
      </div>
      <div class="badge-stack">
        <span class="${escapeHtml(cognitionGraphStatusBadgeClass(model.graph.status || "not_reported"))}" data-component="Badge">${escapeHtml(status)}</span>
        <span class="badge${model.stale ? " warn" : ""}" data-component="Badge">${escapeHtml(model.freshness)}</span>
        <span class="badge" data-component="Badge">${escapeHtml(currentLabel)}</span>
      </div>
      <div class="graph-latest-event">${escapeHtml(model.latestEvent)}</div>
    </div>
  `;
}

function cognitionGraphStageMarkup(model) {
  const groups = cognitionGraphStageGroups(model);
  const columns = Math.max(1, groups.length);
  const stageGroups = groups.map((group, index) => (
    cognitionGraphStageGroupMarkup(group, model, index, groups)
  )).join("");
  return `
    <div class="cognition-graph-stage" data-component="ScrollArea">
      <div class="graph-stage-rail" style="--graph-columns: ${columns};">
        ${stageGroups}
      </div>
    </div>
  `;
}

function cognitionGraphStageGroups(model) {
  const columns = [...new Set(model.nodes.map((node) => (
    Math.max(1, Number(node.column) || 1)
  )))].sort((left, right) => left - right);
  return columns.map((column) => {
    const nodes = model.nodes
      .filter((node) => Math.max(1, Number(node.column) || 1) === column)
      .sort((left, right) => {
        const laneDelta = cognitionGraphLaneIndex(left, model.lanes) - cognitionGraphLaneIndex(right, model.lanes);
        if (laneDelta) return laneDelta;
        return String(left.label || left.id).localeCompare(String(right.label || right.id));
      });
    return {
      column,
      nodes,
      title: cognitionGraphStageTitle(column, nodes),
      status: cognitionGraphGroupStatus(nodes),
      lanes: [...new Set(nodes.map((node) => node.lane || "cognition"))],
    };
  });
}

function cognitionGraphStageTitle(column, nodes) {
  if (!nodes.length) return `Step ${column}`;
  const stages = [...new Set(nodes.map((node) => node.stage).filter(Boolean))];
  if (stages.length === 1) return stages[0];
  if (nodes.some((node) => (node.lane || "") === "input")) return "Input";
  if (nodes.some((node) => (node.lane || "") === "gate")) return "Decision";
  if (nodes.some((node) => (node.lane || "") === "surface")) return "Surface";
  return `Step ${column}`;
}

function cognitionGraphGroupStatus(nodes) {
  if (nodes.some((node) => node.status === "running")) return "running";
  if (nodes.some((node) => node.status === "failed")) return "failed";
  if (nodes.some((node) => node.status === "pending")) return "pending";
  if (nodes.every((node) => node.status === "completed")) return "completed";
  if (
    nodes.some((node) => node.status === "skipped")
    && nodes.every((node) => node.status === "completed" || node.status === "skipped")
  ) {
    return "terminated";
  }
  return "partial";
}

function cognitionGraphStageGroupMarkup(group, model, index, groups) {
  const lanes = group.lanes.join(", ");
  const nodes = group.nodes.map((node) => cognitionGraphNodeMarkup(node, model)).join("");
  return `
    <section class="graph-stage-group status-${escapeHtml(group.status)}" aria-label="${escapeHtml(group.title)}">
      <div class="graph-stage-header">
        <div>
          <span>Step ${escapeHtml(group.column)}</span>
          <strong>${escapeHtml(group.title)}</strong>
        </div>
        <span class="${escapeHtml(cognitionGraphStatusBadgeClass(group.status))}" data-component="Badge">${escapeHtml(cognitionGraphStatusLabel(group.status))}</span>
      </div>
      <div class="graph-branch-stack">
        ${nodes}
      </div>
      <div class="graph-stage-meta">${escapeHtml(lanes || "cognition")}</div>
      ${cognitionGraphConnectorMarkup(index, groups)}
    </section>
  `;
}

function cognitionGraphConnectorMarkup(index, groups) {
  if (index >= groups.length - 1) return "";
  const status = cognitionGraphConnectorStatus(groups[index], groups[index + 1]);
  return `<span class="graph-connector status-${escapeHtml(status)}" aria-hidden="true"></span>`;
}

function cognitionGraphConnectorStatus(group, nextGroup) {
  if (nextGroup?.status === "terminated" || group?.status === "terminated") return "terminated";
  if (group?.status === "failed" || nextGroup?.status === "failed") return "failed";
  if (group?.status === "running" || nextGroup?.status === "running") return "running";
  return "default";
}

function cognitionGraphNodeMarkup(node, model) {
  const status = node.status || "not_reported";
  const statusLabel = cognitionGraphStatusLabel(status);
  const selected = model.selectedNode && model.selectedNode.id === node.id;
  const highlighted = model.highlightedIds.has(node.id);
  const current = model.currentNode && model.currentNode.id === node.id;
  const summary = cognitionGraphNodeSummary(node);
  const branch = node.branch ? `<span>${escapeHtml(node.branch)}</span>` : "";
  return `
    <button class="graph-node status-${escapeHtml(status)}${statusLabel === "terminated" ? " is-terminal" : ""}${current ? " is-current" : ""}${selected ? " is-selected" : ""}${highlighted ? " is-highlighted" : ""}" type="button" data-graph-node data-node-id="${escapeHtml(node.id)}" aria-pressed="${selected ? "true" : "false"}" title="${escapeHtml(summary)}">
      <span class="node-header">
        <span class="node-stage">${escapeHtml(node.stage || "stage")}</span>
        <span class="${escapeHtml(cognitionGraphStatusBadgeClass(status))} node-status" data-component="Badge">${escapeHtml(statusLabel)}</span>
      </span>
      <strong>${escapeHtml(node.label || node.id)}</strong>
      <span class="node-meta"><span>${escapeHtml(node.lane || "cognition")}</span>${branch}</span>
      <span class="node-summary">${escapeHtml(summary)}</span>
    </button>
  `;
}

function cognitionGraphInspectorMarkup(model) {
  const node = model.selectedNode;
  const currentId = model.currentNode?.id || "";
  const selectedId = node?.id || "";
  const showReturn = model.pinned && currentId && selectedId !== currentId;
  const rows = cognitionGraphInspectorRows(node).map(([label, value]) => `
    <div class="graph-inspector-row">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(cognitionGraphValue(value))}</strong>
    </div>
  `).join("");
  const title = node ? `${node.stage || "stage"} · ${node.label || node.id}` : "No selected node";
  return `
    <aside class="graph-inspector" aria-label="Cognition node detail">
      <div class="graph-inspector-header">
        <div>
          <span>${escapeHtml(model.pinned ? "Selected node detail" : `${model.focusKind[0].toUpperCase()}${model.focusKind.slice(1)} node detail`)}</span>
          <strong>${escapeHtml(title)}</strong>
        </div>
        <span class="${escapeHtml(cognitionGraphStatusBadgeClass(node?.status || "not_reported"))}" data-component="Badge">${escapeHtml(cognitionGraphStatusLabel(node?.status || "not_reported"))}</span>
      </div>
      <div class="graph-inspector-rows">${rows}</div>
      <div class="graph-inspector-actions">
        ${showReturn ? `<button class="btn" type="button" data-graph-return-current>Return to current</button>` : ""}
      </div>
    </aside>
  `;
}

function cognitionGraphInspectorRows(node) {
  if (!node) return [["Status", "No cognition node selected."]];
  const detail = node.detail || {};
  const rows = [];
  const priorityKeys = [
    "summary",
    "conclusion",
    "reasoning",
    "internal_monologue",
    "important_signal",
    "decision",
    "logical_stance",
    "character_intent",
    "judgment_note",
    "status",
    "blocker",
  ];
  priorityKeys.forEach((key) => {
    if (detail[key] !== null && detail[key] !== undefined && detail[key] !== "") {
      rows.push([key.replaceAll("_", " "), detail[key]]);
    }
  });
  const hasBoundedDetail = rows.length > 0;
  if (!hasBoundedDetail) rows.push(["Detail", "This node reported no bounded detail."]);
  rows.push(["stage", node.stage || "stage"]);
  rows.push(["lane", node.lane || "cognition"]);
  if (node.branch) rows.push(["branch", node.branch]);
  return rows.slice(0, 7);
}

function cognitionGraphNodeSummary(node) {
  const detail = node.detail || {};
  const value = detail.summary
    || detail.conclusion
    || detail.reasoning
    || detail.internal_monologue
    || detail.important_signal
    || detail.decision
    || detail.status
    || "No bounded detail reported.";
  return cognitionGraphValue(value);
}

function cognitionGraphValue(value) {
  if (Array.isArray(value)) return value.join(", ");
  if (value && typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function renderCapabilitySummary() {
  const labels = {
    overview: "Overview",
    services: "Services",
    logs: "Live logs",
    debug: "Debug chat",
    events: "Event monitor",
    character: "Character",
    memory: "Memory",
    style: "Interaction style",
    calendar: "Calendar",
    background: "Background work",
    health: "Health/cache",
    audit: "Audit",
  };
  const navPages = new Set(qsa("[data-page-link]").map((link) => link.dataset.pageLink));
  const visibleRows = Object.entries(state.pageCapabilities)
    .filter(([key, capability]) => navPages.has(key) && capability.status !== "disabled")
    .map(([key, capability]) => `<tr><td>${escapeHtml(labels[key] || key)}</td><td>${escapeHtml(capability.label || capability.status || "ready")}</td></tr>`);
  const unavailableRows = Object.entries(state.pageCapabilities)
    .filter(([, capability]) => capability.status === "disabled")
    .map(([key, capability]) => `<tr><td>${escapeHtml(labels[key] || key)}</td><td>${escapeHtml(capability.reason || "not available")}</td></tr>`);

  qs("#overview-capability-table").innerHTML = visibleRows.length
    ? visibleRows.join("")
    : "<tr><td>Status</td><td>No product-ready workflows loaded.</td></tr>";
  qs("#overview-unavailable-table").innerHTML = unavailableRows.length
    ? unavailableRows.join("")
    : "<tr><td>Status</td><td>No disabled workflows.</td></tr>";
}

function renderHealth(overview) {
  const health = overview.brain_health || {};
  const runtime = overview.runtime_status || {};
  const cache2 = overview.cache2 || {};
  const agents = Array.isArray(cache2.agents) ? cache2.agents : [];
  const healthStatus = health.status || "unavailable";
  qs("#health-brain-status").textContent = healthStatus;
  qs("#health-brain-detail").textContent = health.reason || (health.db === true ? "database reachable" : "health loaded");
  qs("#health-cache-status").textContent = agents.length ? `${agents.length} agents` : healthStatus;
  qs("#health-cache-table").innerHTML = agents.length
    ? agents.map((agent) => `<tr><td>${escapeHtml(agent.agent_name || "agent")}</td><td>hits ${escapeHtml(agent.hit_count || 0)} / misses ${escapeHtml(agent.miss_count || 0)}</td></tr>`).join("")
    : `<tr><td>Status</td><td>${escapeHtml(health.reason || "no Cache2 agent stats reported")}</td></tr>`;
  qs("#health-runtime-status").textContent = runtime.worker_error_level || runtime.status || healthStatus;
  const runtimeRows = Object.entries(runtime)
    .filter(([, value]) => ["string", "number", "boolean"].includes(typeof value))
    .map(([key, value]) => `<tr><td>${escapeHtml(key)}</td><td>${escapeHtml(value)}</td></tr>`);
  qs("#health-runtime-table").innerHTML = runtimeRows.length
    ? runtimeRows.join("")
    : `<tr><td>Status</td><td>${escapeHtml(runtime.reason || "runtime status not available")}</td></tr>`;
}

function serviceById(serviceId) {
  return state.services.find((item) => item.id === serviceId);
}

function serviceStatus(serviceId) {
  const service = serviceById(serviceId);
  return service ? service.actual_state : "unavailable";
}

function dependenciesAvailable(service) {
  return (service.dependencies || []).every((serviceId) => {
    const dependency = serviceById(serviceId);
    return Boolean(dependency)
      && (dependency.actual_state === "running" || isEndpointConflict(dependency));
  });
}

function serviceActionEnabled(service, action) {
  if (action === "start") return ["stopped", "crashed", "unhealthy"].includes(service.actual_state) && dependenciesAvailable(service);
  if (action === "restart" || action === "stop") return service.actual_state === "running";
  return false;
}

function serviceActionButton(service, action, label, variant = "") {
  const enabled = serviceActionEnabled(service, action);
  const disabled = enabled ? "" : " disabled aria-disabled=\"true\"";
  const className = variant ? `btn ${variant}` : "btn";
  return `<button class="${className}" data-action="${action}" data-service="${escapeHtml(service.id)}" data-version="${escapeHtml(service.version)}"${disabled}>${label}</button>`;
}

function serviceConfigButton(service) {
  const summary = state.serviceConfigSummaries[service.id] || {};
  if (!summary.configurable) return "";
  return `<button class="btn" data-config-service="${escapeHtml(service.id)}" type="button">Configure</button>`;
}

function serviceLogsButton(service) {
  return `<button class="btn" data-log-service="${escapeHtml(service.id)}" type="button">Logs</button>`;
}

function serviceConfigBadge(service) {
  const summary = state.serviceConfigSummaries[service.id] || {};
  if (!summary.configurable) return "";
  const configState = summary.state || "default";
  const className = configState === "override_active" ? "badge warn" : "badge";
  return `<span class="${className}">${escapeHtml(configState.replaceAll("_", " "))}</span>`;
}

function renderServices() {
  const grid = qs("#service-grid");
  grid.innerHTML = "";
  state.services.forEach((service) => {
    const startButton = serviceActionButton(service, "start", "Start", "primary");
    const restartButton = serviceActionButton(service, "restart", "Restart");
    const stopButton = serviceActionButton(service, "stop", "Stop", "danger");
    const configButton = serviceConfigButton(service);
    const logsButton = serviceLogsButton(service);
    const configBadge = serviceConfigBadge(service);
    const serviceError = service.last_error_preview ? `<div class="service-error">${escapeHtml(service.last_error_preview)}</div>` : "";
    grid.insertAdjacentHTML("beforeend", `
      <article class="service-card" data-component="Card" data-service-card="${escapeHtml(service.id)}">
        <div class="service-card-header">
          <div><strong>${escapeHtml(service.display_name)}</strong><br><code>${escapeHtml(service.id)}</code></div>
          <div class="badge-stack">
            <span class="${badgeClass(service.actual_state)}">${escapeHtml(service.actual_state)}</span>
            ${configBadge}
          </div>
        </div>
        <div class="service-card-body">
          <div class="kv"><span>desired</span><strong>${escapeHtml(service.desired_state)}</strong></div>
          <div class="kv"><span>version</span><strong>${escapeHtml(service.version)}</strong></div>
          <div class="kv"><span>pid</span><strong>${escapeHtml(service.pid || "-")}</strong></div>
          <div class="kv"><span>depends</span><code>${escapeHtml((service.dependencies || []).join(", ") || "-")}</code></div>
        </div>
        ${serviceError}
        <div class="service-card-actions">
          ${startButton}
          ${restartButton}
          ${stopButton}
          ${logsButton}
          ${configButton}
        </div>
      </article>
    `);
  });
}

function auditRows(events) {
  if (!events.length) {
    return "<tr><td>No audit events</td><td>local JSONL is ready</td><td>redacted</td></tr>";
  }
  return events.map((event) => `<tr><td><code>${escapeHtml(event.created_at)}</code></td><td>${escapeHtml(event.operator_id)}</td><td>${escapeHtml(event.event_type)}</td><td>${escapeHtml(event.service_id || "-")}</td><td>${escapeHtml(event.reason || "-")}</td></tr>`).join("");
}

function renderAudit(events) {
  const rows = auditRows(events);
  qs("#audit-table").innerHTML = rows;
}

async function refreshAudit() {
  const payload = await api("/api/audit?limit=25");
  renderAudit(payload.items || []);
}

async function serviceAction(event) {
  const button = event.target.closest("[data-action]");
  if (!button) return;
  if (button.disabled) return;
  const serviceId = button.dataset.service;
  const action = button.dataset.action;
  const expectedVersion = Number(button.dataset.version);
  const service = serviceById(serviceId) || {};
  const serviceName = service.display_name || serviceId;
  const actionLabel = lifecycleActionLabel(action);
  button.disabled = true;
  showNotice(`${actionLabel} ${serviceName}...`, "info");
  try {
    await api(`/api/services/${serviceId}/${action}`, {
      method: "POST",
      csrf: true,
      body: JSON.stringify({reason: "operator console action", expected_version: expectedVersion}),
    });
    await bootstrap();
    showNotice(`${serviceName} ${lifecycleActionDoneLabel(action)}.`, "success");
  } catch (error) {
    await bootstrap();
    throw error;
  } finally {
    button.disabled = false;
  }
}

function handleServiceGridClick(event) {
  const logButton = event.target.closest("[data-log-service]");
  if (logButton) {
    openServiceLogs(logButton.dataset.logService);
    return;
  }
  const configButton = event.target.closest("[data-config-service]");
  if (configButton) {
    openServiceConfig(configButton.dataset.configService).catch(reportActionError);
    return;
  }
  serviceAction(event).catch(reportActionError);
}

function openServiceLogs(serviceId) {
  const serviceFilter = qs("#log-service-filter");
  if (serviceFilter) serviceFilter.value = serviceId;
  setPage("logs");
}

function lifecycleActionLabel(action) {
  if (action === "start") return "Starting";
  if (action === "stop") return "Stopping";
  if (action === "restart") return "Restarting";
  return "Updating";
}

function lifecycleActionDoneLabel(action) {
  if (action === "start") return "started";
  if (action === "stop") return "stopped";
  if (action === "restart") return "restarted";
  return "updated";
}

async function openServiceConfig(serviceId) {
  const payload = await api(`/api/services/${encodeURIComponent(serviceId)}/config`);
  state.currentServiceConfig = payload;
  renderServiceConfigDialog(payload);
  qs("#service-config-dialog").hidden = false;
}

function closeServiceConfig() {
  qs("#service-config-dialog").hidden = true;
  state.currentServiceConfig = null;
}

function renderServiceConfigDialog(config) {
  const service = serviceById(config.service_id) || {};
  const serviceLabel = service.display_name || config.service_id;
  qs("#service-config-title").textContent = config.title || serviceLabel;
  qs("#service-config-description").textContent = config.description || "Service runtime override.";
  qs("#service-config-state").textContent = (config.state || "default").replaceAll("_", " ");
  qs("#service-config-state").className = config.state === "override_active" ? "badge warn" : "badge";
  const running = service.actual_state === "running";
  qs("#service-config-restart-note").textContent = running
    ? "Apply and restart"
    : "Applies on next start";
  qs("#service-config-apply").textContent = running ? "Apply and restart" : "Apply override";
  qs("#service-config-fields").innerHTML = (config.fields || []).map((field) => renderConfigField(field)).join("");
}

function renderConfigField(field) {
  const control = configFieldControl(field);
  const defaultValue = configDisplayValue(field.default_value);
  const effectiveValue = configDisplayValue(field.effective_value);
  const overrideValue = field.override_value === null || field.override_value === undefined
    ? "none"
    : configDisplayValue(field.override_value);
  const source = field.default_source || "descriptor default";
  const validation = configValidationText(field.validation || {});
  return `
    <section class="config-field field-set" data-component="FieldSet">
      <div class="field-legend">${escapeHtml(field.label || field.key)}</div>
      <p class="field-description">${escapeHtml(field.description || "")}</p>
      <div class="config-state-grid">
        <div class="kv"><span>default source</span><code>${escapeHtml(source)}</code></div>
        <div class="kv"><span>default</span><strong>${escapeHtml(defaultValue)}</strong></div>
        <div class="kv"><span>effective</span><strong>${escapeHtml(effectiveValue)}</strong></div>
        <div class="kv"><span>override</span><strong>${escapeHtml(overrideValue)}</strong></div>
      </div>
      ${validation ? `<p class="field-description">${escapeHtml(validation)}</p>` : ""}
      ${control}
    </section>
  `;
}

function configFieldControl(field) {
  const key = escapeHtml(field.key);
  const value = field.override_value === null || field.override_value === undefined
    ? field.effective_value
    : field.override_value;
  if (field.value_type === "string_list") {
    const textValue = Array.isArray(value) ? value.join("\n") : "";
    return `
      <label class="field">
        Runtime override
        <textarea class="textarea config-input" data-config-input="${key}" data-config-type="${escapeHtml(field.value_type)}" placeholder="one value per line">${escapeHtml(textValue)}</textarea>
      </label>
    `;
  }
  if (field.value_type === "boolean") {
    const checked = value === true ? " checked" : "";
    return `
      <label class="check-field config-check">
        <input type="checkbox" data-config-input="${key}" data-config-type="${escapeHtml(field.value_type)}"${checked} />
        Runtime override enabled
      </label>
    `;
  }
  return `
    <label class="field">
      Runtime override
      <input class="input config-input" data-config-input="${key}" data-config-type="${escapeHtml(field.value_type)}" value="${escapeHtml(value ?? "")}" />
    </label>
  `;
}

function configDisplayValue(value) {
  if (Array.isArray(value)) return value.length ? value.join(", ") : "empty";
  if (value === null || value === undefined || value === "") return "empty";
  if (typeof value === "boolean") return value ? "true" : "false";
  return String(value);
}

function configValidationText(validation) {
  const parts = [];
  if (validation.pattern) parts.push(`pattern ${validation.pattern}`);
  if (validation.max_items) parts.push(`max ${validation.max_items} items`);
  if (validation.max_item_length) parts.push(`max ${validation.max_item_length} chars per item`);
  if (Array.isArray(validation.options) && validation.options.length) {
    parts.push(`options ${validation.options.join(", ")}`);
  }
  return parts.join("; ");
}

function collectServiceConfigValues() {
  const values = {};
  qsa("[data-config-input]").forEach((input) => {
    const key = input.dataset.configInput;
    const type = input.dataset.configType;
    if (type === "string_list") {
      values[key] = input.value.split(/[\n,\s]+/).map((item) => item.trim()).filter(Boolean);
      return;
    }
    if (type === "boolean") {
      values[key] = input.checked;
      return;
    }
    if (type === "integer") {
      values[key] = Number(input.value);
      return;
    }
    values[key] = input.value;
  });
  return values;
}

async function applyServiceConfig() {
  const config = state.currentServiceConfig;
  if (!config) return;
  const service = serviceById(config.service_id) || {};
  const payload = {
    reason: "operator console action",
    expected_version: service.version,
    values: collectServiceConfigValues(),
  };
  const result = await api(`/api/services/${encodeURIComponent(config.service_id)}/config`, {
    method: "PUT",
    csrf: true,
    body: JSON.stringify(payload),
  });
  state.currentServiceConfig = result.config;
  await bootstrap();
  renderServiceConfigDialog(result.config);
  if (result.restart && result.restart.succeeded === false) {
    showNotice(
      `${service.display_name || config.service_id} configuration saved, but restart failed. ${result.restart.reason || ""}`,
      "error",
    );
    return;
  }
  const restartText = result.restart && result.restart.attempted
    ? " Override applied and restart attempted."
    : " Override stored for next start.";
  showNotice(`${service.display_name || config.service_id} configuration saved.${restartText}`, "success");
}

async function resetServiceConfig() {
  const config = state.currentServiceConfig;
  if (!config) return;
  const service = serviceById(config.service_id) || {};
  const result = await api(`/api/services/${encodeURIComponent(config.service_id)}/config/reset`, {
    method: "POST",
    csrf: true,
    body: JSON.stringify({
      reason: "operator console action",
      expected_version: service.version,
    }),
  });
  state.currentServiceConfig = result.config;
  await bootstrap();
  renderServiceConfigDialog(result.config);
  if (result.restart && result.restart.succeeded === false) {
    showNotice(
      `${service.display_name || config.service_id} configuration reset, but restart failed. ${result.restart.reason || ""}`,
      "error",
    );
    return;
  }
  showNotice(`${service.display_name || config.service_id} configuration reset to default.`, "success");
}

async function sendDebug(event) {
  event.preventDefault();
  const sendButton = qs("#debug-send");
  const form = new FormData(event.target);
  const payload = Object.fromEntries(form.entries());
  const selectedMode = form.get("debug_mode");
  const debugModes = form.getAll("debug_modes");
  if (selectedMode && selectedMode !== "visible_reply") debugModes.push(selectedMode);
  payload.debug_modes = debugModes;
  delete payload.debug_mode;
  const messageText = String(payload.message_text || "").trim();
  state.debugRequestInFlight = true;
  sendButton.disabled = true;
  state.debugCognitionGraph = pendingDebugCognitionGraph(messageText);
  appendChatMessage({
    label: "operator",
    body: messageText || "Debug message sent.",
    meta: "awaiting brain response",
  });
  renderDebugCognitionGraph(state.debugCognitionGraph);
  try {
    const result = await api("/api/debug-chat", {method: "POST", csrf: true, body: JSON.stringify(payload)});
    state.debugCognitionGraph = result.cognition_graph || null;
    const label = result.brain_available ? "brain" : "unavailable";
    const body = debugResponseBody(result);
    const meta = debugResponseMeta(result);
    appendChatMessage({label, body, meta});
    renderDebugCognitionGraph(state.debugCognitionGraph);
  } catch (error) {
    state.debugCognitionGraph = failedDebugCognitionGraph(error);
    renderDebugCognitionGraph(state.debugCognitionGraph);
    appendChatMessage({
      label: "error",
      body: error.message,
      meta: "debug request failed",
    });
    throw error;
  } finally {
    state.debugRequestInFlight = false;
    renderDebugAvailability();
  }
}

function pendingDebugCognitionGraph(messageText) {
  return {
    source: "debug_latest",
    status: "running",
    run_id: "debug request in progress",
    nodes: [
      {
        id: "debug.input",
        label: "Debug input",
        stage: "Input",
        lane: "input",
        column: 1,
        branch: "debug",
        status: "completed",
        detail: {summary: messageText || "message submitted"},
      },
      {
        id: "debug.cognition",
        label: "Cognition",
        stage: "Brain",
        lane: "cognition",
        column: 2,
        branch: "live",
        status: "running",
        detail: {summary: "waiting for debug cognition result"},
      },
    ],
    edges: [
      {source: "debug.input", target: "debug.cognition", kind: "sequence"},
    ],
  };
}

function failedDebugCognitionGraph(error) {
  return {
    source: "debug_latest",
    status: "failed",
    run_id: "debug request failed",
    nodes: [
      {
        id: "debug.error",
        label: "Request failed",
        stage: "Error",
        lane: "cognition",
        column: 1,
        branch: "debug",
        status: "failed",
        detail: {summary: error.message},
      },
    ],
    edges: [],
  };
}

function appendChatMessage({label, body, meta}) {
  qs("#chat-history").insertAdjacentHTML("beforeend", `<article class="message"><div class="meta">${escapeHtml(label)}</div><p>${escapeHtml(body)}</p><div class="meta">${escapeHtml(meta)}</div></article>`);
}

function debugResponseBody(result) {
  if (result.error) return result.error.code || "debug request failed";
  const response = result.response || {};
  const messages = Array.isArray(response.messages) ? response.messages : [];
  if (!messages.length) return "No visible reply messages.";
  const visibleMessages = messages.map((message) => debugMessageText(message));
  return visibleMessages.join("\n");
}

function debugMessageText(message) {
  if (typeof message === "string") return message;
  if (message && typeof message === "object") {
    return message.text || message.content || "Structured message returned.";
  }
  return String(message ?? "");
}

function debugResponseMeta(result) {
  const response = result.response || {};
  const parts = [];
  if (result.tracking_id) parts.push(`tracking ${result.tracking_id}`);
  if (Number.isFinite(result.latency_ms)) parts.push(`${result.latency_ms} ms`);
  if (Number.isFinite(response.delivery_mention_count)) parts.push(`${response.delivery_mention_count} mentions`);
  if (Number.isFinite(response.attachment_count)) parts.push(`${response.attachment_count} attachments`);
  return parts.length ? parts.join(" | ") : "redacted response summary";
}

async function refreshCharacter() {
  const status = await api("/api/character/status");
  const growth = await api("/api/character/growth");
  const summary = status.summary || {};
  const statusRows = Object.entries(summary).map(([key, value]) => `<tr><td>${escapeHtml(key)}</td><td>${escapeHtml(value)}</td></tr>`);
  qs("#character-state-table").innerHTML = statusRows.length
    ? statusRows.join("")
    : `<tr><td>Status</td><td>${escapeHtml(status.status || "unavailable")}</td></tr><tr><td>Reason</td><td>${escapeHtml(status.reason || "no character status rows")}</td></tr>`;

  const items = Array.isArray(growth.items) ? growth.items : [];
  qs("#character-growth-table").innerHTML = items.length
    ? items.map((item) => `<tr><td>${escapeHtml(item.growth_axis || item.trait_id || "trait")}</td><td>${escapeHtml(item.status || growth.status || "available")}</td></tr>`).join("")
    : `<tr><td>Status</td><td>${escapeHtml(growth.status || "unavailable")}</td></tr><tr><td>Reason</td><td>${escapeHtml(growth.reason || "no growth traits")}</td></tr>`;
}

async function refreshMemory(showNeedsInput = true) {
  const platform = qs("#memory-platform").value.trim();
  const platformUserId = qs("#memory-platform-user-id").value.trim();
  const query = qs("#memory-query").value.trim();
  if (!platform || !platformUserId) {
    qs("#memory-status").textContent = "needs input";
    qs("#memory-status").className = "badge";
    if (showNeedsInput) {
      qs("#memory-table").innerHTML = "<tr><td>Status</td><td>Enter platform and platform user ID to load scoped memory units.</td></tr>";
    }
    return;
  }

  const params = new URLSearchParams({platform, platform_user_id: platformUserId, limit: "25"});
  if (query) params.set("query", query);
  const payload = await api(`/api/lookups/memory?${params.toString()}`);
  const status = payload.status || "unavailable";
  qs("#memory-status").textContent = status;
  qs("#memory-status").className = status === "available" ? "badge success" : "badge";
  const items = Array.isArray(payload.items) ? payload.items : [];
  if (!items.length) {
    qs("#memory-table").innerHTML = `<tr><td>Status</td><td>${escapeHtml(payload.reason || "No memory rows matched the current lookup.")}</td></tr>`;
    return;
  }
  qs("#memory-table").innerHTML = items.map((item) => `
    <tr>
      <td><code>${escapeHtml(item.unit_id || "-")}</code></td>
      <td>${escapeHtml(item.unit_type || "-")}</td>
      <td>${escapeHtml(item.fact || item.relationship_signal || "-")}</td>
      <td>${escapeHtml(item.status || "-")}</td>
    </tr>
  `).join("");
}

async function refreshStyle(showNeedsInput = true) {
  const platform = qs("#style-platform").value.trim();
  const platformUserId = qs("#style-platform-user-id").value.trim();
  const groupId = qs("#style-channel-id").value.trim();
  if (!platformUserId && !groupId) {
    qs("#style-status").textContent = "needs input";
    qs("#style-status").className = "badge";
    if (showNeedsInput) {
      qs("#style-table").innerHTML = "<tr><td>Status</td><td>Enter a platform user or group scope to load interaction-style guidance.</td></tr>";
    }
    return;
  }
  if (!platform) {
    qs("#style-status").textContent = "needs input";
    qs("#style-status").className = "badge";
    if (showNeedsInput) {
      qs("#style-table").innerHTML = "<tr><td>Status</td><td>Enter platform with the user or group scope.</td></tr>";
    }
    return;
  }

  const params = new URLSearchParams({limit: "25"});
  if (platform) params.set("platform", platform);
  if (platformUserId) params.set("platform_user_id", platformUserId);
  if (groupId) params.set("group_id", groupId);
  const payload = await api(`/api/lookups/style?${params.toString()}`);
  const status = payload.status || "unavailable";
  qs("#style-status").textContent = status;
  qs("#style-status").className = status === "available" ? "badge success" : "badge";
  const items = Array.isArray(payload.items) ? payload.items : [];
  if (!items.length) {
    qs("#style-table").innerHTML = `<tr><td>Status</td><td>${escapeHtml(payload.reason || "No interaction-style guidance matched the current lookup.")}</td></tr>`;
    return;
  }
  qs("#style-table").innerHTML = items.map((item) => `
    <tr>
      <td>${escapeHtml(item.scope || "-")}</td>
      <td>${escapeHtml(item.field || "-")}</td>
      <td>${escapeHtml((item.guidelines || []).join("; "))}</td>
      <td>${escapeHtml(item.confidence || "-")}</td>
    </tr>
  `).join("");
}

async function refreshCalendar() {
  const payload = await api("/api/lookups/calendar?limit=25");
  const status = payload.status || "unavailable";
  qs("#calendar-status").textContent = status;
  qs("#calendar-status").className = status === "available" ? "badge success" : "badge";
  const items = Array.isArray(payload.items) ? payload.items : [];
  if (!items.length) {
    qs("#calendar-table").innerHTML = `<tr><td>Status</td><td>${escapeHtml(payload.reason || "No due calendar runs.")}</td></tr>`;
    return;
  }
  qs("#calendar-table").innerHTML = items.map((item) => `
    <tr>
      <td><code>${escapeHtml(item.run_id || "-")}</code></td>
      <td>${escapeHtml(item.trigger_kind || "-")}</td>
      <td>${escapeHtml(item.status || "-")}</td>
      <td>${escapeHtml(item.due_at || "-")}</td>
      <td>${escapeHtml(item.attempt_count || 0)} / ${escapeHtml(item.max_attempts || "-")}</td>
    </tr>
  `).join("");
}

async function refreshBackground() {
  const payload = await api("/api/lookups/background?limit=25");
  const status = payload.status || "unavailable";
  qs("#background-status").textContent = status;
  qs("#background-status").className = status === "available" ? "badge success" : "badge";
  const items = Array.isArray(payload.items) ? payload.items : [];
  if (!items.length) {
    qs("#background-table").innerHTML = `<tr><td>Status</td><td>${escapeHtml(payload.reason || "No background worker events.")}</td></tr>`;
    return;
  }
  qs("#background-table").innerHTML = items.map((item) => `
    <tr>
      <td><code>${escapeHtml(item.event_id || "-")}</code></td>
      <td>${escapeHtml(item.event_type || "-")}</td>
      <td>${escapeHtml(item.status || item.level || "-")}</td>
      <td>${escapeHtml(item.created_at || "-")}</td>
      <td>${escapeHtml(item.message || "-")}</td>
    </tr>
  `).join("");
}

async function refreshEvents() {
  const source = qs("#event-source").value;
  const params = new URLSearchParams({source, limit: "25"});
  const requestId = qs("#event-request-id").value.trim();
  const trackingId = qs("#event-tracking-id").value.trim();
  if (requestId) params.set("request_id", requestId);
  if (trackingId) params.set("tracking_id", trackingId);
  const payload = await api(`/api/events?${params.toString()}`);
  if (!payload.items.length) {
    qs("#event-table").innerHTML = "<tr><td>No events</td><td>no rows for the selected source and filters</td><td>redacted</td></tr>";
    return;
  }
  qs("#event-table").innerHTML = payload.items.map((event) => `
    <tr>
      <td>${escapeHtml(event.source || "-")}</td>
      <td>${escapeHtml(event.component || event.service_id || "-")}</td>
      <td>${escapeHtml(event.event_type || "-")}</td>
      <td>${escapeHtml(event.status || event.level || "-")}</td>
      <td>${escapeHtml(event.created_at || "-")}</td>
    </tr>
  `).join("");
}

function renderLogControls() {
  const serviceFilter = qs("#log-service-filter");
  if (!serviceFilter) return;
  const selected = serviceFilter.value || "all";
  const options = ['<option value="all">all services</option>'].concat(
    state.services.map((service) => `<option value="${escapeHtml(service.id)}">${escapeHtml(service.display_name || service.id)}</option>`),
  );
  serviceFilter.innerHTML = options.join("");
  serviceFilter.value = state.services.some((service) => service.id === selected) ? selected : "all";
  updateLogBufferStatus();
}

function logStreamUrl() {
  const params = new URLSearchParams({
    service_id: qs("#log-service-filter").value || "all",
    streams: qs("#log-stream-filter").value || "stdout,stderr,supervisor",
    tail: "100",
  });
  return `/api/logs/stream?${params.toString()}`;
}

function openLogStream(options = {}) {
  if (!state.isAuthenticated) return;
  const url = logStreamUrl();
  if (state.logEventSource && state.logStreamUrl === url && !options.replaceOnReady) return;
  closeLogStream();
  if (options.replaceOnReady) {
    state.pendingLogRows = [];
  } else {
    state.pendingLogRows = null;
  }
  state.logStreamRevision += 1;
  const revision = state.logStreamRevision;
  state.logStreamUrl = url;
  setLogStreamStatus(options.replaceOnReady ? "updating" : "connecting", "badge warn");
  state.logEventSource = new EventSource(url);
  ["log.snapshot", "log.line"].forEach((eventName) => {
    state.logEventSource.addEventListener(eventName, (event) => {
      if (revision !== state.logStreamRevision) return;
      appendLogRow(JSON.parse(event.data), {retained: eventName === "log.snapshot"});
    });
  });
  state.logEventSource.addEventListener("log.ready", () => {
    if (revision !== state.logStreamRevision) return;
    if (state.pendingLogRows) {
      state.logRows = state.pendingLogRows;
      state.pendingLogRows = null;
    }
    renderBufferedLogRows();
    setLogStreamStatus("live", "badge success");
  });
  state.logEventSource.addEventListener("log.gap", (event) => {
    if (revision !== state.logStreamRevision) return;
    const payload = JSON.parse(event.data);
    appendLogStatus(`gap: ${payload.reason || "replay unavailable"}`);
    setLogStreamStatus("gap", "badge warn");
  });
  state.logEventSource.addEventListener("log.status", (event) => {
    if (revision !== state.logStreamRevision) return;
    const payload = JSON.parse(event.data);
    appendLogStatus(payload.message || payload.status || "log status changed");
  });
  state.logEventSource.addEventListener("error", () => {
    if (revision !== state.logStreamRevision) return;
    setLogStreamStatus("reconnecting", "badge warn");
  });
}

function closeLogStream() {
  state.logStreamRevision += 1;
  if (state.logEventSource) state.logEventSource.close();
  state.logEventSource = null;
  state.logStreamUrl = "";
  state.pendingLogRows = null;
}

function setLogStreamStatus(text, className = "badge") {
  const badge = qs("#log-stream-status");
  if (!badge) return;
  badge.textContent = state.logPaused ? "paused locally" : text;
  badge.className = className;
}

function appendLogStatus(message) {
  appendLogRow({
    service_id: "console",
    stream: "supervisor",
    created_at: new Date().toISOString(),
    line: message,
  }, {retained: true});
}

function appendLogRow(row, options = {}) {
  if (state.logPaused && !options.retained && !state.pendingLogRows) {
    state.logDroppedLocal += 1;
    updateLogBufferStatus();
    return;
  }
  const targetRows = state.pendingLogRows || state.logRows;
  targetRows.push(row);
  while (targetRows.length > LOG_ROW_LIMIT) targetRows.shift();
  if (state.pendingLogRows) return;
  renderBufferedLogRows();
}

function renderBufferedLogRows() {
  const table = qs("#log-table");
  const rows = state.logRows.filter(logRowMatches);
  if (!rows.length) {
    renderLogPlaceholder(emptyLogMessage());
    return;
  }
  table.innerHTML = rows.map(renderLogRow).join("");
  if (qs("#log-autoscroll").checked) qs("#log-viewport").scrollTop = qs("#log-viewport").scrollHeight;
  updateLogBufferStatus();
}

function emptyLogMessage() {
  const filter = qs("#log-text-filter").value.trim();
  if (filter) return "No retained rows match this filter. Watching live logs...";
  return "No retained rows for this selection. Watching live logs...";
}

function logRowMatches(row) {
  const filter = qs("#log-text-filter").value.trim().toLowerCase();
  const line = String(row.line || "");
  const matches = !filter || line.toLowerCase().includes(filter);
  return matches;
}

function renderLogRow(row) {
  const wrap = qs("#log-wrap-lines").checked ? " wrap" : "";
  const timestamp = row.created_at || new Date().toISOString();
  const label = `${row.service_id || "-"} ${row.stream || "-"}`;
  const line = String(row.line || "");
  const renderedRow = `
    <tr class="log-row${wrap}">
      <td><code>${escapeHtml(timestamp)}</code><br>${escapeHtml(label)}</td>
      <td>${highlightLogLine(line)}</td>
      <td><button class="btn log-copy" data-copy-log="${escapeHtml(line)}" type="button">Copy</button></td>
    </tr>
  `;
  return renderedRow;
}

function renderLogPlaceholder(message) {
  const table = qs("#log-table");
  table.innerHTML = `<tr class="log-row log-placeholder wrap"><td>Status</td><td>${escapeHtml(message)}</td><td></td></tr>`;
  updateLogBufferStatus();
}

function highlightLogLine(line) {
  const highlight = qs("#log-highlight-filter").value.trim();
  const escapedLine = escapeHtml(line);
  if (!highlight) return escapedLine;
  const escapedHighlight = escapeHtml(highlight);
  return escapedLine.replaceAll(escapedHighlight, `<mark>${escapedHighlight}</mark>`);
}

function updateLogBufferStatus() {
  const table = qs("#log-table");
  const badge = qs("#log-buffer-status");
  if (!table || !badge) return;
  const count = table.querySelectorAll(".log-row:not(.log-placeholder)").length;
  const suffix = state.logDroppedLocal ? `; ${state.logDroppedLocal} paused` : "";
  badge.textContent = `${count} rows${suffix}`;
}

function toggleLogPause() {
  state.logPaused = !state.logPaused;
  qs("#log-pause").textContent = state.logPaused ? "Resume" : "Pause";
  setLogStreamStatus(
    state.logPaused ? "paused locally" : "live",
    state.logPaused ? "badge warn" : "badge success",
  );
}

function clearLogRows() {
  state.logRows = [];
  state.pendingLogRows = null;
  renderLogPlaceholder("Log view cleared locally. New matching lines will appear here.");
  state.logDroppedLocal = 0;
  updateLogBufferStatus();
}

function refreshLogStream() {
  openLogStream({replaceOnReady: true});
}

function copyLogRow(event) {
  const button = event.target.closest("[data-copy-log]");
  if (!button) return;
  if (!navigator.clipboard) return;
  navigator.clipboard.writeText(button.dataset.copyLog || "").catch(() => {});
}

function openStream(url) {
  if (state.eventSource && state.streamUrl === url) return;
  if (state.eventSource) state.eventSource.close();
  state.streamUrl = url;
  state.eventSource = new EventSource(url);
  state.eventSource.addEventListener("control.gap", () => bootstrap({reconnectStream: false}));
  state.eventSource.addEventListener("control.cognition_graph_invalidated", () => bootstrap({reconnectStream: false}));
}

initializeTheme();
qsa("[data-page-link]").forEach((link) => link.addEventListener("click", () => setPage(link.dataset.pageLink)));
qsa("[data-theme-choice]").forEach((button) => button.addEventListener("click", () => setTheme(button.dataset.themeChoice)));
qs("#login").addEventListener("click", () => runButtonAction(
  qs("#login"),
  "Signing in...",
  "Signed in.",
  login,
));
qs("#token").addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    runButtonAction(qs("#login"), "Signing in...", "Signed in.", login);
  }
});
qs("#service-grid").addEventListener("click", handleServiceGridClick);
qs("#log-service-filter").addEventListener("change", refreshLogStream);
qs("#log-stream-filter").addEventListener("change", refreshLogStream);
qs("#log-text-filter").addEventListener("input", renderBufferedLogRows);
qs("#log-highlight-filter").addEventListener("input", renderBufferedLogRows);
qs("#log-pause").addEventListener("click", toggleLogPause);
qs("#log-clear").addEventListener("click", clearLogRows);
qs("#log-wrap-lines").addEventListener("change", () => {
  qsa(".log-row").forEach((row) => row.classList.toggle("wrap", qs("#log-wrap-lines").checked));
});
qs("#log-table").addEventListener("click", copyLogRow);
qs("#service-config-close").addEventListener("click", closeServiceConfig);
qs("#service-config-apply").addEventListener("click", () => runButtonAction(
  qs("#service-config-apply"),
  "Saving service configuration...",
  "",
  applyServiceConfig,
));
qs("#service-config-reset").addEventListener("click", () => runButtonAction(
  qs("#service-config-reset"),
  "Resetting service configuration...",
  "",
  resetServiceConfig,
));
qs("#service-config-dialog").addEventListener("click", (event) => {
  if (event.target === qs("#service-config-dialog")) closeServiceConfig();
});
qs("#debug-form").addEventListener("submit", (event) => sendDebug(event).catch(reportActionError));
qs("#refresh-events").addEventListener("click", () => runButtonAction(
  qs("#refresh-events"),
  "Loading events...",
  "Events updated.",
  refreshEvents,
));
qs("#refresh-memory").addEventListener("click", () => runButtonAction(
  qs("#refresh-memory"),
  "Loading memory...",
  "Memory lookup updated.",
  refreshMemory,
));
qs("#refresh-style").addEventListener("click", () => runButtonAction(
  qs("#refresh-style"),
  "Loading interaction style...",
  "Interaction style updated.",
  refreshStyle,
));
qs("#refresh-calendar").addEventListener("click", () => runButtonAction(
  qs("#refresh-calendar"),
  "Loading calendar...",
  "Calendar updated.",
  refreshCalendar,
));
qs("#refresh-background").addEventListener("click", () => runButtonAction(
  qs("#refresh-background"),
  "Loading background work...",
  "Background work updated.",
  refreshBackground,
));
window.addEventListener("resize", () => {
  renderOverviewCognitionGraph(state.latestCognitionGraph);
  renderDebugCognitionGraph(state.debugCognitionGraph);
});
resumeSession();
