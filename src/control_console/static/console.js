const state = {
  csrfHeaderName: "",
  csrfToken: "",
  services: [],
  serviceConfigSummaries: {},
  currentServiceConfig: null,
  brainModelRoutes: [],
  brainModelServiceState: {},
  selectedBrainRouteKey: "",
  brainRouteFilters: {search: "", group: "all", source: "all", family: "all"},
  dirtyBrainRouteValues: {},
  availableModelCache: {},
  brainRouteActionInFlight: false,
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

function optionalElement(target) {
  return typeof target === "string" ? qs(target) : target;
}

function setHtml(target, html) {
  const element = optionalElement(target);
  if (!element) return null;
  element.innerHTML = html;
  return element;
}

function appendHtml(target, position, html) {
  const element = optionalElement(target);
  if (!element) return null;
  element.insertAdjacentHTML(position, html);
  return element;
}

function setText(target, text) {
  const element = optionalElement(target);
  if (!element) return null;
  element.textContent = text;
  return element;
}

function setClassName(target, className) {
  const element = optionalElement(target);
  if (!element) return null;
  element.className = className;
  return element;
}

function setHidden(target, hidden) {
  const element = optionalElement(target);
  if (!element) return null;
  element.hidden = hidden;
  return element;
}

function setDisabled(target, disabled) {
  const element = optionalElement(target);
  if (!element) return null;
  element.disabled = disabled;
  return element;
}

function setValue(target, value) {
  const element = optionalElement(target);
  if (!element) return null;
  element.value = value;
  return element;
}

function getValue(target, fallback = "") {
  const element = optionalElement(target);
  if (!element) return fallback;
  return element.value ?? fallback;
}

function isChecked(target, fallback = false) {
  const element = optionalElement(target);
  if (!element) return fallback;
  return Boolean(element.checked);
}

function setPlaceholder(target, placeholder) {
  const element = optionalElement(target);
  if (!element) return null;
  element.placeholder = placeholder;
  return element;
}

function bind(target, eventName, handler) {
  const element = optionalElement(target);
  if (!element) return null;
  element.addEventListener(eventName, handler);
  return element;
}

function scrollToBottom(target) {
  const element = optionalElement(target);
  if (!element) return null;
  element.scrollTop = element.scrollHeight;
  return element;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatLookupLabel(value) {
  return String(value ?? "")
    .replaceAll("_", " ")
    .replaceAll("-", " ");
}

function formatLookupValue(value, depth = 0) {
  if (value === null || value === undefined || value === "") return "-";
  if (Array.isArray(value)) {
    if (!value.length) return "-";
    const visibleItems = value.slice(0, 6).map((item) => formatLookupValue(item, depth + 1));
    const extraCount = value.length - visibleItems.length;
    if (extraCount > 0) visibleItems.push(`+${extraCount} more`);
    return visibleItems.join("; ");
  }
  if (typeof value === "object") {
    const entries = Object.entries(value)
      .filter(([, item]) => item !== null && item !== undefined && item !== "")
      .slice(0, 8);
    if (!entries.length) return "-";
    return entries
      .map(([key, item]) => `${formatLookupLabel(key)}: ${formatLookupValue(item, depth + 1)}`)
      .join("; ");
  }
  return String(value);
}

function isKeyValueItems(items) {
  return items.length > 0 && items.every((item) => (
    item
    && typeof item === "object"
    && Object.prototype.hasOwnProperty.call(item, "key")
    && Object.prototype.hasOwnProperty.call(item, "value")
  ));
}

function memoryMeta(parts) {
  return parts
    .filter((part) => part !== null && part !== undefined && part !== "")
    .map((part) => formatLookupValue(part))
    .join(" · ");
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
  if (name === "users" && state.csrfHeaderName) refreshUsers(false).catch(reportActionError);
  if (name === "groups" && state.csrfHeaderName) refreshGroups(false).catch(reportActionError);
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
  setText("#brand-name", name);
  setText("#brand-subtitle", connected ? "Control console" : "database not connected");
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
  if (!dot || !statusText) return;
  if (!state.isAuthenticated) {
    dot.dataset.state = "locked";
    setText(statusText, "Sign in to inspect local services.");
    return;
  }

  dot.dataset.state = brainState;
  const streamState = payload.ui_capabilities.event_stream ? "stream ready" : "stream off";
  if (brainState === "running") {
    setText(statusText, `Brain running; ${streamState}.`);
    return;
  }
  if (isEndpointConflict(brainService)) {
    setText(statusText, "Brain endpoint already running outside the console; lifecycle is unmanaged.");
    return;
  }
  if (brainState === "conflict") {
    setText(statusText, "Brain has a stale lifecycle conflict; inspect Services.");
    return;
  }
  if (brainState === "unavailable") {
    setText(statusText, "Brain unavailable; check the service registry.");
    return;
  }
  setText(statusText, `Brain ${brainState}; lifecycle controls available.`);
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
  setText(statusBadge, isEndpointConflict(brainService)
    ? "brain unmanaged"
    : `brain ${brainState}`);
  setClassName(statusBadge, available ? "badge success" : "badge");
  qsa("[data-debug-input]").forEach((control) => {
    control.disabled = !available;
  });
  setPlaceholder("[name='message_text']", available
    ? "Send a debug message through /chat"
    : "Start or connect the brain service before sending a debug message");
  setDisabled("#debug-send", !available || state.debugRequestInFlight);
}

function showNotice(message, tone = "info") {
  const notice = qs("#ui-notice");
  if (!notice) return;
  setHidden(notice, false);
  notice.dataset.tone = tone;
  setText(notice, message);
}

function clearNotice() {
  const notice = qs("#ui-notice");
  if (!notice) return;
  setHidden(notice, true);
  notice.dataset.tone = "idle";
  setText(notice, "");
}

async function runButtonAction(button, loadingMessage, successMessage, action) {
  setDisabled(button, true);
  showNotice(loadingMessage, "info");
  try {
    await action();
    if (successMessage) showNotice(successMessage, "success");
  } catch (error) {
    showNotice(error.message, "danger");
  } finally {
    setDisabled(button, false);
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
  const token = getValue("#token");
  const payload = await api("/api/auth/login", {method: "POST", body: JSON.stringify({token})});
  state.csrfHeaderName = payload.csrf_header_name;
  state.csrfToken = payload.csrf_token;
  setValue("#token", "");
  setAuthState(true);
  setText("#session-state", payload.operator.operator_id);
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
  setText("#session-state", payload.operator ? payload.operator.operator_id : "signed in");
  renderBrand(payload.application_identity || {});
  renderPageCapabilities();
  renderShellStatus(payload);
  renderDebugAvailability();
  renderOverview(payload);
  renderHealth(payload.overview || {});
  renderServices();
  await refreshBrainModelRoutes({silent: true});
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
  state.brainModelRoutes = [];
  state.brainModelServiceState = {};
  state.selectedBrainRouteKey = "";
  state.brainRouteFilters = {search: "", group: "all", source: "all", family: "all"};
  state.dirtyBrainRouteValues = {};
  state.availableModelCache = {};
  state.brainRouteActionInFlight = false;
  state.pageCapabilities = {};
  state.latestCognitionGraph = null;
  state.debugCognitionGraph = null;
  if (state.eventSource) state.eventSource.close();
  closeLogStream();
  state.eventSource = null;
  state.streamUrl = "";
  setAuthState(false);
  setText("#session-state", "signed out");
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
  if (!grid) return;
  setHtml(grid, "");
  const runningCount = payload.services.filter((service) => service.actual_state === "running").length;
  const visibleWorkflowCount = qsa("[data-page-link]").length;
  const cards = [
    ["Brain service", serviceStatus("brain"), "local child process"],
    ["Managed services", `${runningCount} / ${payload.services.length}`, "registry declared"],
    ["Event stream", payload.ui_capabilities.event_stream ? "ready" : "off", payload.stream_url],
    ["Visible workflows", visibleWorkflowCount, "primary navigation"],
  ];
  cards.forEach(([label, value, note]) => {
    appendHtml(grid, "beforeend", `<article class="metric" data-component="Card"><div class="metric-label">${escapeHtml(label)}</div><div class="metric-value">${escapeHtml(value)}</div><div class="metric-label">${escapeHtml(note)}</div></article>`);
  });
  setHtml("#overview-runtime-table", [
    ["Services", payload.services.length],
    ["Audit events", payload.recent_audit_events.length],
    ["CSRF header", payload.csrf_header_name],
    ["Stream URL", payload.stream_url],
  ].map(([key, value]) => `<tr><td>${escapeHtml(key)}</td><td>${escapeHtml(value)}</td></tr>`).join(""));
  setHtml("#overview-audit-table", auditRows(payload.recent_audit_events));
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
    setHtml(container, `<p class="graph-empty">${escapeHtml(emptyMessage)}</p>`);
    return;
  }

  const lanes = cognitionGraphLanes(nodes);
  const maxColumn = nodes.reduce((maximum, node) => Math.max(maximum, Number(node.column) || 1), 1);
  const model = cognitionGraphModel({graph, nodes, edges, lanes, maxColumn});
  setHtml(container, `
    <div class="cognition-graph-shell" data-graph-source="${escapeHtml(model.source)}" data-graph-run-id="${escapeHtml(model.runId)}" data-graph-current-node-id="${escapeHtml(model.currentNode?.id || "")}" data-graph-selected-node-id="${escapeHtml(model.selectedNode?.id || "")}">
      ${cognitionGraphSummaryMarkup(model)}
      <div class="graph-body">
        ${cognitionGraphStageMarkup(model)}
        ${cognitionGraphInspectorMarkup(model)}
      </div>
    </div>
  `);
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
    users: "Users",
    groups: "Groups",
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

  setHtml("#overview-capability-table", visibleRows.length
    ? visibleRows.join("")
    : "<tr><td>Status</td><td>No product-ready workflows loaded.</td></tr>");
  setHtml("#overview-unavailable-table", unavailableRows.length
    ? unavailableRows.join("")
    : "<tr><td>Status</td><td>No disabled workflows.</td></tr>");
}

function renderHealth(overview) {
  const health = overview.brain_health || {};
  const runtime = overview.runtime_status || {};
  const cache2 = overview.cache2 || {};
  const agents = Array.isArray(cache2.agents) ? cache2.agents : [];
  const healthStatus = health.status || "unavailable";
  setText("#health-brain-status", healthStatus);
  setText("#health-brain-detail", health.reason || (health.db === true ? "database reachable" : "health loaded"));
  setText("#health-cache-status", agents.length ? `${agents.length} agents` : healthStatus);
  setHtml("#health-cache-table", agents.length
    ? agents.map((agent) => `<tr><td>${escapeHtml(agent.agent_name || "agent")}</td><td>hits ${escapeHtml(agent.hit_count || 0)} / misses ${escapeHtml(agent.miss_count || 0)}</td></tr>`).join("")
    : `<tr><td>Status</td><td>${escapeHtml(health.reason || "no Cache2 agent stats reported")}</td></tr>`);
  setText("#health-runtime-status", runtime.worker_error_level || runtime.status || healthStatus);
  const runtimeRows = Object.entries(runtime)
    .filter(([, value]) => ["string", "number", "boolean"].includes(typeof value))
    .map(([key, value]) => `<tr><td>${escapeHtml(key)}</td><td>${escapeHtml(value)}</td></tr>`);
  setHtml("#health-runtime-table", runtimeRows.length
    ? runtimeRows.join("")
    : `<tr><td>Status</td><td>${escapeHtml(runtime.reason || "runtime status not available")}</td></tr>`);
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

async function refreshBrainModelRoutes(options = {}) {
  try {
    const payload = await api("/api/services/brain/model-routes");
    state.brainModelRoutes = payload.routes || [];
    state.brainModelServiceState = payload.service_state || payload.service || {};
    if (!state.selectedBrainRouteKey && state.brainModelRoutes.length) {
      state.selectedBrainRouteKey = state.brainModelRoutes[0].route_key;
    }
    renderServices();
  } catch (error) {
    state.brainModelRoutes = [];
    state.brainModelServiceState = {};
    if (!options.silent) throw error;
  }
}

function renderBrainServiceCard(service) {
  const routes = state.brainModelRoutes || [];
  const selectedRoute = selectedBrainRoute();
  const routeSummary = brainRouteSummary(routes);
  const startButton = serviceActionButton(service, "start", "Start", "primary");
  const restartButton = serviceActionButton(service, "restart", "Restart");
  const stopButton = serviceActionButton(service, "stop", "Stop", "danger");
  const logsButton = serviceLogsButton(service);
  const configBadge = serviceConfigBadge(service);
  const serviceError = service.last_error_preview ? `<div class="service-error">${escapeHtml(service.last_error_preview)}</div>` : "";
  return `
    <article class="service-card brain-service-card" data-component="Card" data-service-card="${escapeHtml(service.id)}">
      <div class="service-card-header">
        <div><strong>${escapeHtml(service.display_name)}</strong><br><code>${escapeHtml(service.id)}</code></div>
        <div class="badge-stack">
          <span class="${badgeClass(service.actual_state)}">${escapeHtml(service.actual_state)}</span>
          ${configBadge}
          <span class="badge">${escapeHtml(routes.length)} routes</span>
        </div>
      </div>
      <div class="brain-service-layout">
        <section class="brain-runtime-panel">
          <div class="brain-runtime-grid">
            <div class="kv"><span>desired</span><strong>${escapeHtml(service.desired_state)}</strong></div>
            <div class="kv"><span>version</span><strong>${escapeHtml(service.version)}</strong></div>
            <div class="kv"><span>pid</span><strong>${escapeHtml(service.pid || "-")}</strong></div>
            <div class="kv"><span>depends</span><code>${escapeHtml((service.dependencies || []).join(", ") || "-")}</code></div>
            <div class="kv"><span>override routes</span><strong>${escapeHtml(routeSummary.overrideCount)}</strong></div>
            <div class="kv"><span>families</span><strong>${escapeHtml(routeSummary.familyCount)}</strong></div>
          </div>
          ${serviceError}
          <div class="service-card-actions brain-runtime-actions">
            ${startButton}
            ${restartButton}
            ${stopButton}
            ${logsButton}
            <button class="btn" data-brain-route-refresh-all type="button">Refresh routes</button>
          </div>
        </section>
        <section class="brain-routes-panel">
          ${renderBrainRouteMatrix(routes)}
          ${renderBrainRouteEditor(selectedRoute, service)}
        </section>
      </div>
    </article>
  `;
}

function renderBrainRouteMatrix(routes) {
  const filteredRoutes = filteredBrainRoutes(routes);
  const groups = uniqueRouteValues(routes, "group");
  const sources = uniqueRouteValues(routes, "effective", "source");
  const families = uniqueRouteValues(routes, "diagnostics", "model_family");
  return `
    <div class="brain-route-toolbar">
      <label class="field">
        Search
        <input class="input" data-brain-route-filter="search" value="${escapeHtml(state.brainRouteFilters.search)}" placeholder="route or model" />
      </label>
      <label class="field">
        Group
        <select class="input" data-brain-route-filter="group">${brainFilterOptions(groups, state.brainRouteFilters.group)}</select>
      </label>
      <label class="field">
        Source
        <select class="input" data-brain-route-filter="source">${brainFilterOptions(sources, state.brainRouteFilters.source)}</select>
      </label>
      <label class="field">
        Family
        <select class="input" data-brain-route-filter="family">${brainFilterOptions(families, state.brainRouteFilters.family)}</select>
      </label>
    </div>
    <div class="brain-route-matrix" role="list">
      ${filteredRoutes.length ? filteredRoutes.map(renderBrainRouteTile).join("") : "<p class=\"panel-empty\">No model routes match the selected filters.</p>"}
    </div>
  `;
}

function renderBrainRouteTile(route) {
  const selected = route.route_key === state.selectedBrainRouteKey ? " selected" : "";
  const source = route.effective?.source || "default";
  const sourceClass = source === "override" ? "badge warn" : "badge";
  const family = route.diagnostics?.model_family || "unknown";
  const thinking = route.effective?.thinking_enabled ? "thinking" : "standard";
  return `
    <button class="brain-route-tile${selected}" data-brain-route-key="${escapeHtml(route.route_key)}" type="button" role="listitem">
      <span class="brain-route-name">${escapeHtml(route.label || route.route_key)}</span>
      <code>${escapeHtml(route.effective?.model || "not configured")}</code>
      <span class="brain-route-meta">
        <span class="${sourceClass}">${escapeHtml(source)}</span>
        <span class="badge">${escapeHtml(family)}</span>
        <span class="badge">${escapeHtml(thinking)}</span>
      </span>
    </button>
  `;
}

function renderBrainRouteEditor(route, service) {
  if (!route) {
    return `<section class="brain-route-editor"><p class="panel-empty">Select a route to configure its model override.</p></section>`;
  }
  const dirty = brainRouteDirtyValues(route);
  const modelValue = dirty.model ?? route.override?.model ?? route.effective?.model ?? "";
  const tokensValue = dirty.max_completion_tokens ?? route.override?.max_completion_tokens ?? route.effective?.max_completion_tokens ?? "";
  const thinkingValue = dirty.thinking_enabled ?? route.override?.thinking_enabled ?? route.effective?.thinking_enabled ?? false;
  const modelsState = state.availableModelCache[route.route_key] || {status: "not_loaded", models: []};
  const modelPicker = renderBrainModelPicker(route, modelsState, modelValue);
  const applyDisabled = state.brainRouteActionInFlight || !brainRouteHasDirty(route) ? " disabled aria-disabled=\"true\"" : "";
  const loadingDisabled = state.brainRouteActionInFlight ? " disabled aria-disabled=\"true\"" : "";
  const refreshLabel = brainModelRefreshLabel(modelsState);
  const runningText = service.actual_state === "running" ? "apply and restart" : "store for next start";
  return `
    <section class="brain-route-editor" data-selected-brain-route="${escapeHtml(route.route_key)}">
      <div class="brain-route-editor-header">
        <div>
          <strong>${escapeHtml(route.label)}</strong>
          <span>${escapeHtml(route.group)} · ${escapeHtml(route.env_prefix)}</span>
        </div>
        <div class="badge-stack">
          <span class="${route.required ? "badge warn" : "badge"}">${route.required ? "required" : "fallback backed"}</span>
          <span class="badge">${escapeHtml(route.diagnostics?.base_url_label || "provider unknown")}</span>
        </div>
      </div>
      <div class="brain-route-current">
        <div class="kv"><span>effective</span><strong>${escapeHtml(route.effective?.model || "not configured")}</strong></div>
        <div class="kv"><span>default</span><strong>${escapeHtml(route.default?.model || "empty")}</strong></div>
        <div class="kv"><span>source</span><strong>${escapeHtml(route.effective?.source || "default")}</strong></div>
      </div>
      <div class="brain-route-form">
        ${modelPicker}
        <label class="field">
          Max completion tokens
          <input class="input" data-brain-route-input="max_completion_tokens" type="number" min="1" max="65536" value="${escapeHtml(tokensValue)}" />
        </label>
        <label class="check-field brain-thinking-toggle">
          <input type="checkbox" data-brain-route-input="thinking_enabled"${thinkingValue ? " checked" : ""} />
          Thinking enabled
        </label>
      </div>
      <div class="brain-model-picker-state">${availableModelStatus(modelsState)}</div>
      <div class="service-card-actions brain-route-actions">
        <button class="btn" data-brain-route-refresh="${escapeHtml(route.route_key)}"${loadingDisabled} type="button">${refreshLabel}</button>
        <button class="btn" data-brain-route-reset="${escapeHtml(route.route_key)}"${loadingDisabled} type="button">Reset route</button>
        <button class="btn primary" data-brain-route-apply="${escapeHtml(route.route_key)}"${applyDisabled} type="button">${escapeHtml(runningText)}</button>
      </div>
    </section>
  `;
}

function ensureBrainRouteModelsLoaded(routeKey) {
  const cache = state.availableModelCache[routeKey] || {status: "not_loaded"};
  if (cache.status !== "not_loaded") return;
  refreshBrainAvailableModels(routeKey).catch(reportActionError);
}

function refreshBrainAvailableModels(routeKey) {
  const cache = state.availableModelCache[routeKey] || {};
  if (cache.status === "loading") return Promise.resolve();
  state.availableModelCache[routeKey] = {...cache, status: "loading", models: []};
  renderServices();
  return api(`/api/services/brain/model-routes/${encodeURIComponent(routeKey)}/available-models`)
    .then((payload) => {
      state.availableModelCache[routeKey] = {
        status: payload.status || "unavailable",
        models: payload.models || [],
        message: payload.message || "",
      };
      renderServices();
    })
    .catch((error) => {
      state.availableModelCache[routeKey] = {
        status: "unavailable",
        models: [],
        message: error.message,
      };
      renderServices();
    });
}

function selectedBrainRoute() {
  const routes = state.brainModelRoutes || [];
  return routes.find((route) => route.route_key === state.selectedBrainRouteKey) || routes[0] || null;
}

function brainRouteDirtyValues(route) {
  return state.dirtyBrainRouteValues[route.route_key] || {};
}

function brainRouteHasDirty(route) {
  return Object.keys(brainRouteDirtyValues(route)).length > 0;
}

function setBrainRouteDirtyValue(route, fieldName, value) {
  const dirtyValues = {...brainRouteDirtyValues(route)};
  if (route.effective?.[fieldName] === value) {
    delete dirtyValues[fieldName];
  } else {
    dirtyValues[fieldName] = value;
  }
  if (Object.keys(dirtyValues).length) {
    state.dirtyBrainRouteValues[route.route_key] = dirtyValues;
  } else {
    delete state.dirtyBrainRouteValues[route.route_key];
  }
}

function renderBrainModelPicker(route, modelsState, selectedModel) {
  const status = modelsState.status || "not_loaded";
  const models = modelsState.models || [];
  if (status === "available" && models.length === 1) {
    return singleBrainModelState(route, models[0]);
  }
  if (status === "available" && models.length > 1) {
    const selected = selectedAvailableModel(models, selectedModel);
    if (selected !== selectedModel) setBrainRouteDirtyValue(route, "model", selected);
    return `
      <label class="field">
        Available model
        <select class="input" data-brain-route-input="model">
          ${availableModelOptions(models, selected)}
        </select>
      </label>
    `;
  }
  if (status === "loading") {
    return brainModelStateMarkup("Available model", "loading", "Loading provider model list...");
  }
  if (status === "empty") {
    return brainModelStateMarkup("Available model", "empty", modelsState.message || "Provider returned no valid model ids.");
  }
  if (status === "unavailable") {
    return brainModelStateMarkup("Available model", "unavailable", modelsState.message || "Provider model list unavailable.");
  }
  return brainModelStateMarkup("Available model", "not loaded", "Provider model discovery will start for this route.");
}

function singleBrainModelState(route, model) {
  const modelId = model.id || "";
  if (modelId) setBrainRouteDirtyValue(route, "model", modelId);
  return `
    <div class="field">
      Available model
      <div class="brain-discovered-model">
        <span class="badge success">single discovered model</span>
        <code>${escapeHtml(modelId || "no valid model id")}</code>
        <span>${escapeHtml(model.family || "unknown")}</span>
      </div>
    </div>
  `;
}

function brainModelStateMarkup(label, status, message) {
  const tone = status === "loading" ? "badge warn" : "badge";
  return `
    <div class="field">
      ${escapeHtml(label)}
      <div class="brain-discovered-model">
        <span class="${tone}">${escapeHtml(status)}</span>
        <span>${escapeHtml(message)}</span>
      </div>
    </div>
  `;
}

function availableModelOptions(models, selectedModel) {
  const selected = selectedAvailableModel(models, selectedModel);
  return models.map((model) => {
    const value = model.id || "";
    const selectedText = value === selected ? " selected" : "";
    return `<option value="${escapeHtml(value)}"${selectedText}>${escapeHtml(value)} · ${escapeHtml(model.family || "unknown")}</option>`;
  }).join("");
}

function selectedAvailableModel(models, selectedModel) {
  const modelIds = models.map((model) => model.id || "").filter(Boolean);
  return modelIds.includes(selectedModel) ? selectedModel : modelIds[0] || "";
}

function availableModelStatus(modelsState) {
  const status = modelsState.status || "not_loaded";
  if (status === "loading") return "Loading provider model list...";
  if (status === "available") {
    const count = (modelsState.models || []).length;
    if (count === 1) return "One discovered provider model.";
    return `${count} provider models available.`;
  }
  if (status === "empty") return modelsState.message || "Provider returned no valid model ids.";
  if (status === "unavailable") return modelsState.message || "Provider model list unavailable.";
  return "Provider model discovery has not loaded for this route.";
}

function brainModelRefreshLabel(modelsState) {
  const status = modelsState.status || "not_loaded";
  if (status === "empty" || status === "unavailable") return "Retry discovery";
  return "Refresh models";
}

function brainRouteSummary(routes) {
  const overrideCount = routes.filter((route) => route.effective?.source === "override").length;
  const families = new Set(routes.map((route) => route.diagnostics?.model_family || "unknown"));
  return {overrideCount, familyCount: families.size};
}

function filteredBrainRoutes(routes) {
  const filters = state.brainRouteFilters;
  const search = filters.search.trim().toLowerCase();
  return routes.filter((route) => {
    const source = route.effective?.source || "default";
    const family = route.diagnostics?.model_family || "unknown";
    const haystack = `${route.label} ${route.route_key} ${route.effective?.model || ""}`.toLowerCase();
    return (!search || haystack.includes(search))
      && (filters.group === "all" || route.group === filters.group)
      && (filters.source === "all" || source === filters.source)
      && (filters.family === "all" || family === filters.family);
  });
}

function uniqueRouteValues(routes, primaryKey, secondaryKey = "") {
  const values = new Set();
  routes.forEach((route) => {
    const source = secondaryKey ? route[primaryKey]?.[secondaryKey] : route[primaryKey];
    if (source) values.add(source);
  });
  return Array.from(values).sort((a, b) => String(a).localeCompare(String(b)));
}

function brainFilterOptions(values, selected) {
  const options = [`<option value="all"${selected === "all" ? " selected" : ""}>all</option>`];
  values.forEach((value) => {
    const selectedText = value === selected ? " selected" : "";
    options.push(`<option value="${escapeHtml(value)}"${selectedText}>${escapeHtml(value)}</option>`);
  });
  return options.join("");
}

function renderGenericServiceCard(service) {
  const startButton = serviceActionButton(service, "start", "Start", "primary");
  const restartButton = serviceActionButton(service, "restart", "Restart");
  const stopButton = serviceActionButton(service, "stop", "Stop", "danger");
  const configButton = serviceConfigButton(service);
  const logsButton = serviceLogsButton(service);
  const configBadge = serviceConfigBadge(service);
  const serviceError = service.last_error_preview ? `<div class="service-error">${escapeHtml(service.last_error_preview)}</div>` : "";
  return `
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
  `;
}

function renderServices() {
  const grid = qs("#service-grid");
  if (!grid) return;
  setHtml(grid, "");
  state.services.forEach((service) => {
    const markup = service.id === "brain"
      ? renderBrainServiceCard(service)
      : renderGenericServiceCard(service);
    appendHtml(grid, "beforeend", markup);
  });
  const route = selectedBrainRoute();
  if (route) ensureBrainRouteModelsLoaded(route.route_key);
}

function auditRows(events) {
  if (!events.length) {
    return "<tr><td>No audit events</td><td>local JSONL is ready</td><td>redacted</td></tr>";
  }
  return events.map((event) => `<tr><td><code>${escapeHtml(event.created_at)}</code></td><td>${escapeHtml(event.operator_id)}</td><td>${escapeHtml(event.event_type)}</td><td>${escapeHtml(event.service_id || "-")}</td><td>${escapeHtml(event.reason || "-")}</td></tr>`).join("");
}

function renderAudit(events) {
  const rows = auditRows(events);
  setHtml("#audit-table", rows);
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
  const routeButton = event.target.closest("[data-brain-route-key]");
  if (routeButton) {
    state.selectedBrainRouteKey = routeButton.dataset.brainRouteKey || "";
    renderServices();
    return;
  }
  const routeRefreshAllButton = event.target.closest("[data-brain-route-refresh-all]");
  if (routeRefreshAllButton) {
    refreshBrainModelRoutes().catch(reportActionError);
    return;
  }
  const routeRefreshButton = event.target.closest("[data-brain-route-refresh]");
  if (routeRefreshButton) {
    refreshBrainAvailableModels(routeRefreshButton.dataset.brainRouteRefresh).catch(reportActionError);
    return;
  }
  const routeApplyButton = event.target.closest("[data-brain-route-apply]");
  if (routeApplyButton) {
    applyBrainRoute(routeApplyButton.dataset.brainRouteApply).catch(reportActionError);
    return;
  }
  const routeResetButton = event.target.closest("[data-brain-route-reset]");
  if (routeResetButton) {
    resetBrainRoute(routeResetButton.dataset.brainRouteReset).catch(reportActionError);
    return;
  }
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

function handleServiceGridInput(event) {
  const filter = event.target.closest("[data-brain-route-filter]");
  if (filter) {
    state.brainRouteFilters[filter.dataset.brainRouteFilter] = filter.value;
    renderServices();
    return;
  }
  const input = event.target.closest("[data-brain-route-input]");
  if (!input) return;
  const route = selectedBrainRoute();
  if (!route) return;
  const fieldName = input.dataset.brainRouteInput;
  let value = input.type === "checkbox" ? input.checked : input.value;
  if (fieldName === "max_completion_tokens") {
    value = Number(input.value);
  }
  if (fieldName === "model") {
    const modelsState = state.availableModelCache[route.route_key] || {};
    const discoveredIds = (modelsState.models || []).map((model) => model.id || "");
    if (!discoveredIds.includes(value)) return;
  }
  setBrainRouteDirtyValue(route, fieldName, value);
  if (event.type === "change") renderServices();
  else updateBrainRouteApplyButtons();
}

function updateBrainRouteApplyButtons() {
  const route = selectedBrainRoute();
  qsa("[data-brain-route-apply]").forEach((button) => {
    button.disabled = !route || state.brainRouteActionInFlight || !brainRouteHasDirty(route);
  });
}

async function applyBrainRoute(routeKey) {
  const route = (state.brainModelRoutes || []).find((item) => item.route_key === routeKey);
  if (!route) return;
  const dirtyValues = brainRouteDirtyValues(route);
  if (!Object.keys(dirtyValues).length) return;
  const service = serviceById("brain") || {};
  state.brainRouteActionInFlight = true;
  renderServices();
  try {
    const payload = await api(`/api/services/brain/model-routes/${encodeURIComponent(routeKey)}`, {
      method: "PUT",
      csrf: true,
      body: JSON.stringify({
        reason: "operator console model route change",
        expected_version: service.version,
        values: dirtyValues,
      }),
    });
    state.brainModelRoutes = payload.routes || [];
    state.brainModelServiceState = payload.service_state || payload.service || {};
    delete state.dirtyBrainRouteValues[routeKey];
    await bootstrap();
    showNotice(payload.restart?.attempted ? "Brain model route saved; restart attempted." : "Brain model route saved for next start.", "success");
  } finally {
    state.brainRouteActionInFlight = false;
    renderServices();
  }
}

async function resetBrainRoute(routeKey) {
  const service = serviceById("brain") || {};
  state.brainRouteActionInFlight = true;
  renderServices();
  try {
    const payload = await api(`/api/services/brain/model-routes/${encodeURIComponent(routeKey)}/reset`, {
      method: "POST",
      csrf: true,
      body: JSON.stringify({
        reason: "operator console model route reset",
        expected_version: service.version,
      }),
    });
    state.brainModelRoutes = payload.routes || [];
    state.brainModelServiceState = payload.service_state || payload.service || {};
    delete state.dirtyBrainRouteValues[routeKey];
    await bootstrap();
    showNotice("Brain model route reset.", "success");
  } finally {
    state.brainRouteActionInFlight = false;
    renderServices();
  }
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
  setHidden("#service-config-dialog", false);
}

function closeServiceConfig() {
  setHidden("#service-config-dialog", true);
  state.currentServiceConfig = null;
}

function renderServiceConfigDialog(config) {
  const service = serviceById(config.service_id) || {};
  const serviceLabel = service.display_name || config.service_id;
  setText("#service-config-title", config.title || serviceLabel);
  setText("#service-config-description", config.description || "Service runtime override.");
  setText("#service-config-state", (config.state || "default").replaceAll("_", " "));
  setClassName("#service-config-state", config.state === "override_active" ? "badge warn" : "badge");
  const running = service.actual_state === "running";
  setText("#service-config-restart-note", running
    ? "Apply and restart"
    : "Applies on next start");
  setText("#service-config-apply", running ? "Apply and restart" : "Apply override");
  setHtml("#service-config-fields", (config.fields || []).map((field) => renderConfigField(field)).join(""));
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
  setDisabled(sendButton, true);
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
  appendHtml("#chat-history", "beforeend", `<article class="message"><div class="meta">${escapeHtml(label)}</div><p>${escapeHtml(body)}</p><div class="meta">${escapeHtml(meta)}</div></article>`);
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

function panelItems(panel) {
  return panel && Array.isArray(panel.items) ? panel.items : [];
}

function panelEmptyText(panel, fallback) {
  if (!panel) return fallback;
  return panel.reason || panel.status || fallback;
}

function setEntityStatus(selector, status) {
  const element = qs(selector);
  element.textContent = status || "unavailable";
  element.className = status === "available" ? "badge success" : "badge";
}

function renderPanelState(target, panel) {
  const element = typeof target === "string" ? qs(target) : target;
  if (!element) return;
  const status = panel?.status || "unavailable";
  const reason = panel?.reason || "No rows are available for this panel.";
  const generatedAt = panel?.generated_at || "";
  setHtml(element, `<tr><td>Status</td><td>${escapeHtml(status)}</td></tr><tr><td>Reason</td><td>${escapeHtml(reason)}</td></tr>${generatedAt ? `<tr><td>Generated</td><td>${escapeHtml(generatedAt)}</td></tr>` : ""}`);
}

function renderLookupTable(target, {items = [], emptyText = "No rows available.", redaction = {}} = {}) {
  const element = typeof target === "string" ? qs(target) : target;
  if (!element) return;
  if (!items.length) {
    const redactionNote = redaction.model_inputs ? ` Model inputs ${redaction.model_inputs}.` : "";
    setHtml(element, `<tr><td>Status</td><td>${escapeHtml(emptyText + redactionNote)}</td></tr>`);
    return;
  }
  if (isKeyValueItems(items)) {
    setHtml(element, items.map((item) => (
      `<tr><td>${escapeHtml(formatLookupLabel(item.key))}</td><td>${escapeHtml(formatLookupValue(item.value))}</td></tr>`
    )).join(""));
    return;
  }
  setHtml(element, items.map((item) => {
    const rows = Object.entries(item)
      .filter(([, value]) => value !== null && value !== undefined && value !== "")
      .map(([key, value]) => `<tr><td>${escapeHtml(formatLookupLabel(key))}</td><td>${escapeHtml(formatLookupValue(value))}</td></tr>`);
    return rows.join("");
  }).join(""));
}

function renderPromptPanel(target, panel, {emptyText = "No prompt context is available."} = {}) {
  const element = typeof target === "string" ? qs(target) : target;
  if (!element) return;
  if (!panel || panel.status === "needs_input" || panel.status === "unavailable") {
    renderPanelState(element, panel || {status: "needs_input", reason: emptyText});
    return;
  }
  const content = panel.content;
  const rows = [];
  if (content !== null && content !== undefined && content !== "") {
    if (typeof content === "object") {
      Object.entries(content).forEach(([key, value]) => {
        rows.push(`<tr><td>${escapeHtml(formatLookupLabel(key))}</td><td><pre class="prompt-content">${escapeHtml(formatLookupValue(value))}</pre></td></tr>`);
      });
    } else {
      rows.push(`<tr><td>content</td><td><pre class="prompt-content">${escapeHtml(content)}</pre></td></tr>`);
    }
  }
  panelItems(panel).forEach((item) => {
    Object.entries(item)
      .filter(([, value]) => value !== null && value !== undefined && value !== "")
      .forEach(([key, value]) => {
        rows.push(`<tr><td>${escapeHtml(formatLookupLabel(key))}</td><td>${escapeHtml(formatLookupValue(value))}</td></tr>`);
      });
  });
  ["panel_contract", "source", "turn_count", "continuity", "selected_count", "candidate_count", "scope_order", "scope_summary", "projection_owner"].forEach((key) => {
    const value = panel[key];
    if (value !== null && value !== undefined && value !== "") {
      rows.push(`<tr><td>${escapeHtml(formatLookupLabel(key))}</td><td>${escapeHtml(formatLookupValue(value))}</td></tr>`);
    }
  });
  if (!rows.length) {
    renderPanelState(element, {status: panel.status || "empty", reason: panel.reason || emptyText, generated_at: panel.generated_at || ""});
    return;
  }
  setHtml(element, rows.join(""));
}

function renderOperationalPanel(target, panel, {emptyText = "No backing rows are available."} = {}) {
  const element = typeof target === "string" ? qs(target) : target;
  if (!element) return;
  const items = panelItems(panel);
  const metadataRows = ["panel_contract", "projection_owner"].map((key) => {
    const value = panel ? panel[key] : "";
    if (value === null || value === undefined || value === "") return "";
    return `<tr><td>${escapeHtml(formatLookupLabel(key))}</td><td>${escapeHtml(formatLookupValue(value))}</td></tr>`;
  }).filter(Boolean);
  if (!items.length) {
    const reason = panelEmptyText(panel, emptyText);
    setHtml(element, metadataRows.concat(
      `<tr><td>Status</td><td>${escapeHtml(reason)}</td></tr>`,
    ).join(""));
    return;
  }
  const rowHtml = items.map((item) => {
    const rows = Object.entries(item)
      .filter(([, value]) => value !== null && value !== undefined && value !== "")
      .map(([key, value]) => `<tr><td>${escapeHtml(formatLookupLabel(key))}</td><td>${escapeHtml(formatLookupValue(value))}</td></tr>`);
    return rows.join("");
  });
  setHtml(element, metadataRows.concat(rowHtml).join(""));
}

function renderReadableLookupValue(value) {
  return `<span class="table-primary">${escapeHtml(formatLookupValue(value))}</span>`;
}

function renderReadableLookupTable(target, {items = [], emptyText = "No rows available.", redaction = {}} = {}) {
  const element = typeof target === "string" ? qs(target) : target;
  if (!element) return;
  if (!items.length) {
    renderLookupTable(element, {items, emptyText, redaction});
    return;
  }
  if (isKeyValueItems(items)) {
    setHtml(element, items.map((item) => (
      `<tr><td>${escapeHtml(formatLookupLabel(item.key))}</td><td>${renderReadableLookupValue(item.value)}</td></tr>`
    )).join(""));
    return;
  }
  setHtml(element, items.map((item) => {
    const rows = Object.entries(item)
      .filter(([, value]) => value !== null && value !== undefined && value !== "")
      .map(([key, value]) => `<tr><td>${escapeHtml(formatLookupLabel(key))}</td><td>${renderReadableLookupValue(value)}</td></tr>`);
    return rows.join("");
  }).join(""));
}

function renderPanelEmptyContent(target, {emptyText = "No rows available.", redaction = {}} = {}) {
  const element = typeof target === "string" ? qs(target) : target;
  if (!element) return;
  const redactionNote = redaction.model_inputs ? ` Model inputs ${redaction.model_inputs}.` : "";
  setHtml(element, `<p class="panel-empty">${escapeHtml(emptyText + redactionNote)}</p>`);
}

function firstObjectItem(items) {
  return items.find((item) => item && typeof item === "object" && !Array.isArray(item)) || {};
}

function formatCharacterProse(value) {
  return formatLookupValue(value).replace(/;\s+/g, "\n");
}

function detailChip(label, value) {
  if (value === null || value === undefined || value === "") return "";
  return `<span class="detail-chip"><span>${escapeHtml(formatLookupLabel(label))}</span>${escapeHtml(formatLookupValue(value))}</span>`;
}

function detailChipRow(entries) {
  const chips = entries
    .map(([label, value]) => detailChip(label, value))
    .filter(Boolean)
    .join("");
  return chips ? `<div class="detail-chip-row">${chips}</div>` : "";
}

function renderDetailGrid(entries) {
  const rows = entries
    .filter(([, value]) => value !== null && value !== undefined && value !== "")
    .map(([label, value]) => `
      <div class="detail-kv">
        <span class="detail-label">${escapeHtml(formatLookupLabel(label))}</span>
        <span class="detail-value">${escapeHtml(formatCharacterProse(value))}</span>
      </div>
    `)
    .join("");
  return rows ? `<div class="detail-grid">${rows}</div>` : "";
}

function renderCharacterProfilePanel(target, {items = [], emptyText = "No character profile rows.", redaction = {}} = {}) {
  const element = typeof target === "string" ? qs(target) : target;
  if (!element) return;
  if (!items.length) {
    renderPanelEmptyContent(element, {emptyText, redaction});
    return;
  }
  const item = firstObjectItem(items);
  const knownFields = new Set(["name", "description", "gender", "age", "birthday", "personality_brief", "updated_at"]);
  const name = formatLookupValue(item.name || "Character");
  const description = item.description ? formatCharacterProse(item.description) : "";
  const personality = item.personality_brief && typeof item.personality_brief === "object"
    ? Object.entries(item.personality_brief)
    : [];
  const extraDetails = Object.entries(item).filter(([key, value]) => (
    !knownFields.has(key) && value !== null && value !== undefined && value !== ""
  ));
  setHtml(element, `
    <section class="character-summary">
      <div class="character-heading">
        <div>
          <h4 class="character-title">${escapeHtml(name)}</h4>
          ${item.updated_at ? `<span class="detail-muted">updated ${escapeHtml(formatLookupValue(item.updated_at))}</span>` : ""}
        </div>
        ${detailChipRow([
          ["gender", item.gender],
          ["age", item.age],
          ["birthday", item.birthday],
        ])}
      </div>
      ${description ? `<p class="character-prose">${escapeHtml(description)}</p>` : ""}
      ${personality.length ? `
        <section class="detail-section">
          <h5>Personality brief</h5>
          ${renderDetailGrid(personality)}
        </section>
      ` : ""}
      ${extraDetails.length ? `
        <section class="detail-section">
          <h5>Additional profile</h5>
          ${renderDetailGrid(extraDetails)}
        </section>
      ` : ""}
    </section>
  `);
}

function recentWindowEntries(value) {
  if (!value) return [];
  if (Array.isArray(value)) {
    return value.slice(0, 8).map((item) => {
      if (item && typeof item === "object" && !Array.isArray(item)) {
        const timestamp = item.timestamp || item.updated_at || item.date || item.year || "";
        const summary = item.summary || (item.title ? `title: ${formatLookupValue(item.title)}` : "");
        if (summary) return {timestamp, summary};
        const fallbackEntries = Object.entries(item).filter(([key]) => !["timestamp", "updated_at", "date", "year"].includes(key));
        return {timestamp, summary: formatLookupValue(Object.fromEntries(fallbackEntries))};
      }
      return {timestamp: "", summary: formatLookupValue(item)};
    }).filter((entry) => entry.summary);
  }
  if (typeof value === "object") {
    return recentWindowEntries([value]);
  }
  const text = String(value).trim();
  const entries = [];
  const pattern = /timestamp:\s*([^;]+);\s*summary:\s*([^;]+)/gi;
  let match = pattern.exec(text);
  while (match) {
    entries.push({timestamp: match[1].trim(), summary: match[2].trim()});
    match = pattern.exec(text);
  }
  if (entries.length) return entries;
  return text
    .split(/\s*;\s*/)
    .map((part) => part.trim())
    .filter(Boolean)
    .map((summary) => ({timestamp: "", summary}));
}

function renderTimeline(entries) {
  if (!entries.length) return "";
  return `
    <div class="timeline-list">
      ${entries.map((entry) => `
        <article class="timeline-item">
          ${entry.timestamp ? `<span class="detail-muted">${escapeHtml(formatLookupValue(entry.timestamp))}</span>` : ""}
          <p>${escapeHtml(formatCharacterProse(entry.summary))}</p>
        </article>
      `).join("")}
    </div>
  `;
}

function formatTraitStrength(value) {
  const numberValue = Number(value);
  if (!Number.isFinite(numberValue)) return value;
  return String(Math.round(numberValue * 1000) / 1000);
}

function renderCharacterSelfImagePanel(target, {items = [], emptyText = "No self-image rows.", redaction = {}} = {}) {
  const element = typeof target === "string" ? qs(target) : target;
  if (!element) return;
  if (!items.length) {
    renderPanelEmptyContent(element, {emptyText, redaction});
    return;
  }
  const item = firstObjectItem(items);
  const historicalSummary = item.historical_summary || item.current_self_concept || item.summary || "";
  const recentEntries = recentWindowEntries(item.recent_window);
  const milestoneEntries = recentWindowEntries(item.milestones);
  setHtml(element, `
    <section class="character-summary">
      ${recentEntries.length ? `
        <section class="detail-section">
          <h5>Recent window</h5>
          ${renderTimeline(recentEntries)}
        </section>
      ` : ""}
      ${milestoneEntries.length ? `
        <section class="detail-section">
          <h5>Milestones</h5>
          ${renderTimeline(milestoneEntries)}
        </section>
      ` : ""}
      ${historicalSummary ? `
        <section class="detail-section">
          <h5>Long-term self-image</h5>
          <p class="character-prose">${escapeHtml(formatCharacterProse(historicalSummary))}</p>
        </section>
      ` : ""}
      ${detailChipRow([
        ["last updated", item.last_updated || item.updated_at],
        ["synthesis count", item.synthesis_count],
      ])}
    </section>
  `);
}

function renderCharacterGrowthPanel(target, {items = [], emptyText = "No growth traits.", redaction = {}} = {}) {
  const element = typeof target === "string" ? qs(target) : target;
  if (!element) return;
  if (!items.length) {
    renderPanelEmptyContent(element, {emptyText, redaction});
    return;
  }
  setHtml(element, items.map((item) => {
    const title = item.trait_name || item.growth_axis || "Growth trait";
    const guidance = item.guidance || item.summary || "";
    return `
      <article class="trait-card">
        <div class="trait-header">
          <div>
            <h4>${escapeHtml(formatLookupValue(title))}</h4>
            ${item.updated_at ? `<span class="detail-muted">updated ${escapeHtml(formatLookupValue(item.updated_at))}</span>` : ""}
          </div>
          ${item.status ? `<span class="badge success">${escapeHtml(formatLookupValue(item.status))}</span>` : ""}
        </div>
        ${detailChipRow([
          ["axis", item.growth_axis],
          ["maturity", item.maturity_band],
          ["evidence", item.evidence_count],
          ["strength", formatTraitStrength(item.strength)],
        ])}
        ${guidance ? `<p class="character-prose">${escapeHtml(formatCharacterProse(guidance))}</p>` : ""}
      </article>
    `;
  }).join(""));
}

function renderMemoryUnitRows(target, {items = [], emptyText = "No memory rows available.", redaction = {}} = {}) {
  const element = typeof target === "string" ? qs(target) : target;
  if (!element) return;
  if (!items.length) {
    renderLookupTable(element, {items, emptyText, redaction});
    return;
  }
  setHtml(element, items.map((item) => {
    const typeText = formatLookupLabel(item.unit_type || "memory");
    const statusText = item.status ? formatLookupLabel(item.status) : "";
    const factText = formatLookupValue(item.fact || item.subjective_appraisal || item.relationship_signal);
    const primaryMeta = memoryMeta([
      statusText,
      item.updated_at ? `updated: ${formatLookupValue(item.updated_at)}` : "",
      item.last_seen_at && !item.updated_at ? `last seen: ${formatLookupValue(item.last_seen_at)}` : "",
    ]);
    const detailMeta = memoryMeta([
      item.relationship_signal ? `relationship: ${formatLookupValue(item.relationship_signal)}` : "",
      item.subjective_appraisal ? `appraisal: ${formatLookupValue(item.subjective_appraisal)}` : "",
      item.due_at ? `due: ${formatLookupValue(item.due_at)}` : "",
    ]);
    return `
      <tr>
        <td>
          <span class="table-primary">${escapeHtml(typeText || "memory")}</span>
          ${primaryMeta ? `<span class="table-meta">${escapeHtml(primaryMeta)}</span>` : ""}
        </td>
        <td>
          <span class="table-primary">${escapeHtml(factText)}</span>
          ${detailMeta ? `<span class="table-meta">${escapeHtml(detailMeta)}</span>` : ""}
        </td>
      </tr>
    `;
  }).join(""));
}

function renderStyleOverlayRows(target, {items = [], scopeLabel = "style"} = {}) {
  const element = typeof target === "string" ? qs(target) : target;
  if (!element) return;
  if (!items.length) {
    setHtml(element, `<tr><td>Status</td><td>No ${escapeHtml(scopeLabel)} guidance rows are available.</td></tr>`);
    return;
  }
  setHtml(element, items.map((item, index) => {
    const separator = index < items.length - 1 ? `<tr class="table-row-separator"><td colspan="2"></td></tr>` : "";
    const meta = memoryMeta([
      item.field ? formatLookupLabel(item.field) : "",
      item.scope ? `scope: ${formatLookupValue(item.scope)}` : "",
      item.confidence ? `confidence: ${formatLookupValue(item.confidence)}` : "",
    ]);
    const rows = `
      <tr>
        <td>Guidance</td>
        <td>
          <span class="table-primary">${escapeHtml(formatLookupValue(item.guidelines || []))}</span>
          ${meta ? `<span class="table-meta">${escapeHtml(meta)}</span>` : ""}
        </td>
      </tr>
      ${separator}
    `;
    return rows;
  }).join(""));
}

function renderStyleOverlayPanel(target, panel, {scopeLabel = "style"} = {}) {
  const items = panelItems(panel);
  if (!items.length && (panel?.status || panel?.reason)) {
    renderPanelState(target, {
      status: panel?.status || "empty",
      reason: panel?.reason || `No ${scopeLabel} guidance rows are available.`,
      generated_at: panel?.generated_at || "",
    });
    return;
  }
  renderStyleOverlayRows(target, {items, scopeLabel});
}

async function refreshCharacter() {
  const payload = await api("/api/entities/character?limit=25");
  setEntityStatus("#character-status", payload.status || "unavailable");
  const panels = payload.panels || {};
  renderPromptPanel("#character-growth-prompt-table", panels.promoted_global_growth_prompt, {
    emptyText: "No promoted global-growth prompt context.",
  });
  renderPromptPanel("#character-carry-over-table", panels.current_carry_over, {
    emptyText: "No character-global carry-over.",
  });
  renderOperationalPanel("#character-growth-runs-table", panels.growth_runs_audit, {
    emptyText: "No global-growth run audit rows.",
  });
  renderCharacterProfilePanel("#character-profile-table", {
    items: panelItems(panels.profile),
    emptyText: panelEmptyText(panels.profile, "No character profile rows."),
    redaction: payload.redaction || {},
  });
  renderLookupTable("#character-state-table", {
    items: panelItems(panels.state),
    emptyText: panelEmptyText(panels.state, "No character state rows."),
    redaction: payload.redaction || {},
  });
  renderCharacterSelfImagePanel("#character-self-image-table", {
    items: panelItems(panels.self_image),
    emptyText: panelEmptyText(panels.self_image, "No self-image rows."),
    redaction: payload.redaction || {},
  });
  renderCharacterGrowthPanel("#character-growth-table", {
    items: panelItems(panels.growth),
    emptyText: panelEmptyText(panels.growth, "No growth traits."),
    redaction: payload.redaction || {},
  });
  renderLookupTable("#character-learning-table", {
    items: panelItems(panels.learning),
    emptyText: panelEmptyText(panels.learning, "No background learning rows."),
    redaction: payload.redaction || {},
  });
}

async function refreshUsers(showNeedsInput = true) {
  const platform = getValue("#user-platform").trim();
  const platformUserId = getValue("#user-platform-user-id").trim();
  const platformChannelId = getValue("#user-platform-channel-id").trim();
  const channelType = getValue("#user-channel-type").trim();
  const query = getValue("#user-query").trim();
  if (!platform || !platformUserId) {
    setEntityStatus("#users-status", "needs input");
    if (showNeedsInput) {
      renderPanelState("#user-profile-table", {status: "needs_input", reason: "Enter platform and platform user ID to load user profile and relationship."});
      renderPanelState("#user-memory-table", {status: "needs_input", reason: "Enter platform and platform user ID to load user memory."});
      renderPanelState("#user-style-table", {status: "needs_input", reason: "Enter platform and platform user ID to load user style."});
      renderPanelState("#user-conversation-progress-table", {status: "needs_input", reason: "Enter platform, platform user ID, channel ID, and channel type to load conversation progress."});
      renderPanelState("#user-carry-over-table", {status: "needs_input", reason: "Enter platform, platform user ID, channel ID, and channel type to load carry-over."});
    }
    return;
  }

  const params = new URLSearchParams({platform, platform_user_id: platformUserId, limit: "25"});
  if (platformChannelId) params.set("platform_channel_id", platformChannelId);
  if (channelType) params.set("channel_type", channelType);
  if (query) params.set("query", query);
  const payload = await api(`/api/entities/user?${params.toString()}`);
  setEntityStatus("#users-status", payload.status || "unavailable");
  const panels = payload.panels || {};
  renderPromptPanel("#user-conversation-progress-table", panels.conversation_progress_prompt, {
    emptyText: "No conversation-progress prompt context.",
  });
  renderPromptPanel("#user-carry-over-table", panels.current_carry_over, {
    emptyText: "No current carry-over.",
  });
  const relationshipRows = panelItems(panels.relationship);
  const relationshipItem = relationshipRows.reduce((row, item) => {
    if (item && item.key) row[item.key] = item.value;
    return row;
  }, {});
  const profileItems = panelItems(panels.profile);
  const mergedProfileItems = Object.keys(relationshipItem).length
    ? [...profileItems, relationshipItem]
    : profileItems;
  renderReadableLookupTable("#user-profile-table", {
    items: mergedProfileItems,
    emptyText: panelEmptyText(panels.profile, panelEmptyText(panels.relationship, "No user profile rows.")),
    redaction: payload.redaction || {},
  });
  renderMemoryUnitRows("#user-memory-table", {
    items: panelItems(panels.memory),
    emptyText: panelEmptyText(panels.memory, "No user memory rows."),
    redaction: payload.redaction || {},
  });
  renderStyleOverlayPanel("#user-style-table", panels.style, {
    scopeLabel: "user style",
  });
}

async function refreshGroups(showNeedsInput = true) {
  const platform = getValue("#group-platform").trim();
  const groupId = getValue("#group-id").trim();
  const participantPlatformUserId = getValue("#group-participant-platform-user-id").trim();
  if (!platform || !groupId) {
    setEntityStatus("#groups-status", "needs input");
    if (showNeedsInput) {
      renderPanelState("#group-style-table", {status: "needs_input", reason: "Enter platform and group ID to load group style."});
      renderPanelState("#group-carry-over-table", {status: "needs_input", reason: "Enter platform and group ID to load group carry-over."});
      renderPanelState("#group-participant-progress-table", {status: "needs_input", reason: "Enter participant platform user ID to load participant progress."});
    }
    return;
  }

  const params = new URLSearchParams({platform, group_id: groupId, limit: "25"});
  if (participantPlatformUserId) {
    params.set("participant_platform_user_id", participantPlatformUserId);
  }
  const payload = await api(`/api/entities/group?${params.toString()}`);
  setEntityStatus("#groups-status", payload.status || "unavailable");
  const panels = payload.panels || {};
  renderPromptPanel("#group-carry-over-table", panels.group_carry_over, {
    emptyText: "No group carry-over.",
  });
  renderPromptPanel("#group-participant-progress-table", panels.participant_conversation_progress_prompt, {
    emptyText: "No participant conversation-progress prompt context.",
  });
  renderStyleOverlayPanel("#group-style-table", panels.style, {
    scopeLabel: "group style",
  });
}

async function refreshCalendar() {
  const platform = getValue("#calendar-platform").trim();
  const platformChannelId = getValue("#calendar-platform-channel-id").trim();
  const platformUserId = getValue("#calendar-platform-user-id").trim();
  const channelType = getValue("#calendar-channel-type").trim();
  const params = new URLSearchParams({limit: "25"});
  if (platform) params.set("platform", platform);
  if (platformChannelId) params.set("platform_channel_id", platformChannelId);
  if (platformUserId) params.set("platform_user_id", platformUserId);
  if (channelType) params.set("channel_type", channelType);
  const payload = await api(`/api/lookups/calendar?${params.toString()}`);
  const status = payload.status || "unavailable";
  setText("#calendar-status", status);
  setClassName("#calendar-status", status === "available" ? "badge success" : "badge");
  const panels = payload.panels || {};
  renderPromptPanel("#calendar-prompt-runs-table", panels.cognition_pending_runs, {
    emptyText: "No pending calendar prompt candidates.",
  });
  renderOperationalPanel("#calendar-schedules-table", panels.schedule_definitions, {
    emptyText: "No schedule definitions.",
  });
  renderOperationalPanel("#calendar-due-runs-table", panels.due_runs, {
    emptyText: "No due calendar runs.",
  });
}

async function refreshBackground() {
  const payload = await api("/api/lookups/background?limit=25");
  const status = payload.status || "unavailable";
  setText("#background-status", status);
  setClassName("#background-status", status === "available" ? "badge success" : "badge");
  const panels = payload.panels || {};
  renderPromptPanel("#background-result-ready-table", panels.result_ready_cognition_deliveries, {
    emptyText: "No result-ready cognition deliveries.",
  });
  renderOperationalPanel("#background-job-queue-table", panels.job_queue, {
    emptyText: "No background-work jobs.",
  });
  renderOperationalPanel("#background-worker-events-table", panels.worker_events, {
    emptyText: "No background worker events.",
  });
}

async function refreshEvents() {
  const source = getValue("#event-source", "console") || "console";
  const params = new URLSearchParams({source, limit: "25"});
  const requestId = getValue("#event-request-id").trim();
  const trackingId = getValue("#event-tracking-id").trim();
  if (requestId) params.set("request_id", requestId);
  if (trackingId) params.set("tracking_id", trackingId);
  const payload = await api(`/api/events?${params.toString()}`);
  if (!payload.items.length) {
    setHtml("#event-table", "<tr><td>No events</td><td>no rows for the selected source and filters</td><td>redacted</td></tr>");
    return;
  }
  setHtml("#event-table", payload.items.map((event) => `
    <tr>
      <td>${escapeHtml(event.source || "-")}</td>
      <td>${escapeHtml(event.component || event.service_id || "-")}</td>
      <td>${escapeHtml(event.event_type || "-")}</td>
      <td>${escapeHtml(event.status || event.level || "-")}</td>
      <td>${escapeHtml(event.created_at || "-")}</td>
    </tr>
  `).join(""));
}

function renderLogControls() {
  const serviceFilter = qs("#log-service-filter");
  if (!serviceFilter) return;
  const selected = serviceFilter.value || "all";
  const options = ['<option value="all">all services</option>'].concat(
    state.services.map((service) => `<option value="${escapeHtml(service.id)}">${escapeHtml(service.display_name || service.id)}</option>`),
  );
  setHtml(serviceFilter, options.join(""));
  serviceFilter.value = state.services.some((service) => service.id === selected) ? selected : "all";
  updateLogBufferStatus();
}

function logStreamUrl() {
  const params = new URLSearchParams({
    service_id: getValue("#log-service-filter", "all") || "all",
    streams: getValue("#log-stream-filter", "stdout,stderr,supervisor") || "stdout,stderr,supervisor",
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
  if (!table) return;
  const rows = state.logRows.filter(logRowMatches);
  if (!rows.length) {
    renderLogPlaceholder(emptyLogMessage());
    return;
  }
  setHtml(table, rows.map(renderLogRow).join(""));
  if (isChecked("#log-autoscroll")) scrollToBottom("#log-viewport");
  updateLogBufferStatus();
}

function emptyLogMessage() {
  const filter = getValue("#log-text-filter").trim();
  if (filter) return "No retained rows match this filter. Watching live logs...";
  return "No retained rows for this selection. Watching live logs...";
}

function logRowMatches(row) {
  const filter = getValue("#log-text-filter").trim().toLowerCase();
  const line = String(row.line || "");
  const matches = !filter || line.toLowerCase().includes(filter);
  return matches;
}

function renderLogRow(row) {
  const wrap = isChecked("#log-wrap-lines") ? " wrap" : "";
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
  if (!table) return;
  setHtml(table, `<tr class="log-row log-placeholder wrap"><td>Status</td><td>${escapeHtml(message)}</td><td></td></tr>`);
  updateLogBufferStatus();
}

function highlightLogLine(line) {
  const highlight = getValue("#log-highlight-filter").trim();
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
  setText("#log-pause", state.logPaused ? "Resume" : "Pause");
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
bind("#login", "click", () => runButtonAction(
  optionalElement("#login"),
  "Signing in...",
  "Signed in.",
  login,
));
bind("#token", "keydown", (event) => {
  if (event.key === "Enter") {
    runButtonAction(optionalElement("#login"), "Signing in...", "Signed in.", login);
  }
});
bind("#service-grid", "click", handleServiceGridClick);
bind("#service-grid", "input", handleServiceGridInput);
bind("#service-grid", "change", handleServiceGridInput);
bind("#log-service-filter", "change", refreshLogStream);
bind("#log-stream-filter", "change", refreshLogStream);
bind("#log-text-filter", "input", renderBufferedLogRows);
bind("#log-highlight-filter", "input", renderBufferedLogRows);
bind("#log-pause", "click", toggleLogPause);
bind("#log-clear", "click", clearLogRows);
bind("#log-wrap-lines", "change", () => {
  qsa(".log-row").forEach((row) => row.classList.toggle("wrap", isChecked("#log-wrap-lines")));
});
bind("#log-table", "click", copyLogRow);
bind("#service-config-close", "click", closeServiceConfig);
bind("#service-config-apply", "click", () => runButtonAction(
  optionalElement("#service-config-apply"),
  "Saving service configuration...",
  "",
  applyServiceConfig,
));
bind("#service-config-reset", "click", () => runButtonAction(
  optionalElement("#service-config-reset"),
  "Resetting service configuration...",
  "",
  resetServiceConfig,
));
bind("#service-config-dialog", "click", (event) => {
  if (event.target === optionalElement("#service-config-dialog")) closeServiceConfig();
});
bind("#debug-form", "submit", (event) => sendDebug(event).catch(reportActionError));
bind("#refresh-events", "click", () => runButtonAction(
  optionalElement("#refresh-events"),
  "Loading events...",
  "Events updated.",
  refreshEvents,
));
bind("#refresh-users", "click", () => runButtonAction(
  optionalElement("#refresh-users"),
  "Searching user...",
  "User search complete.",
  refreshUsers,
));
bind("#refresh-groups", "click", () => runButtonAction(
  optionalElement("#refresh-groups"),
  "Searching group...",
  "Group search complete.",
  refreshGroups,
));
bind("#refresh-calendar", "click", () => runButtonAction(
  optionalElement("#refresh-calendar"),
  "Loading calendar...",
  "Calendar updated.",
  refreshCalendar,
));
bind("#refresh-background", "click", () => runButtonAction(
  optionalElement("#refresh-background"),
  "Loading background work...",
  "Background work updated.",
  refreshBackground,
));
window.addEventListener("resize", () => {
  renderOverviewCognitionGraph(state.latestCognitionGraph);
  renderDebugCognitionGraph(state.debugCognitionGraph);
});
resumeSession();
