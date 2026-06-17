const state = {
  csrfHeaderName: "",
  csrfToken: "",
  services: [],
  pageCapabilities: {},
  applicationIdentity: {},
  latestCognitionGraph: null,
  debugCognitionGraph: null,
  eventSource: null,
  isAuthenticated: false,
};

const THEME_STORAGE_KEY = "kazusa-control-theme";
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
  if (name === "audit" && state.csrfHeaderName) refreshAudit().catch((error) => alert(error.message));
  if (name === "character" && state.csrfHeaderName) refreshCharacter().catch((error) => alert(error.message));
  if (name === "memory" && state.csrfHeaderName) refreshMemory(false).catch((error) => alert(error.message));
  if (name === "style" && state.csrfHeaderName) refreshStyle(false).catch((error) => alert(error.message));
  if (name === "calendar" && state.csrfHeaderName) refreshCalendar().catch((error) => alert(error.message));
  if (name === "background" && state.csrfHeaderName) refreshBackground().catch((error) => alert(error.message));
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
  const brainState = serviceStatus("brain");
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
  if (brainState === "conflict") {
    statusText.textContent = "Brain endpoint already running outside the console; lifecycle is unmanaged.";
    return;
  }
  if (brainState === "unavailable") {
    statusText.textContent = "Brain unavailable; check the service registry.";
    return;
  }
  statusText.textContent = `Brain ${brainState}; lifecycle controls available.`;
}

function isBrainHttpAvailable(brainState) {
  return ["running", "conflict"].includes(brainState);
}

function renderDebugAvailability() {
  const brainState = serviceStatus("brain");
  const available = state.isAuthenticated && isBrainHttpAvailable(brainState);
  const statusBadge = qs("#debug-brain-status");
  statusBadge.textContent = brainState === "conflict" ? "brain unmanaged" : `brain ${brainState}`;
  statusBadge.className = available ? "badge success" : "badge";
  qsa("[data-debug-input]").forEach((control) => {
    control.disabled = !available;
  });
  qs("[name='message_text']").placeholder = available
    ? "Send a debug message through /chat"
    : "Start or connect the brain service before sending a debug message";
  qs("#debug-send").disabled = !available;
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
}

async function bootstrap() {
  const payload = await api("/api/bootstrap");
  state.csrfHeaderName = payload.csrf_header_name || "";
  state.csrfToken = payload.csrf_token || "";
  state.services = payload.services;
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
  renderAudit(payload.recent_audit_events || []);
  openStream(payload.stream_url);
}

function lockSession() {
  state.csrfHeaderName = "";
  state.csrfToken = "";
  state.services = [];
  state.pageCapabilities = {};
  state.latestCognitionGraph = null;
  state.debugCognitionGraph = null;
  setAuthState(false);
  qs("#session-state").textContent = "signed out";
  renderBrand({status: "unavailable", character_name: "not connected"});
  renderDebugAvailability();
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
  status.className = graphStatus === "completed" || graphStatus === "running"
    ? "badge success"
    : "badge";

  if (!nodes.length) {
    container.innerHTML = `<p class="graph-empty">${escapeHtml(emptyMessage)}</p>`;
    return;
  }

  const lanes = cognitionGraphLanes(nodes);
  const maxColumn = nodes.reduce((maximum, node) => Math.max(maximum, Number(node.column) || 1), 1);
  const nodeMarkup = nodes.map((node) => cognitionGraphNodeMarkup(node, lanes)).join("");
  container.innerHTML = `
    <div class="cognition-graph-stage" style="--graph-columns: ${maxColumn}; --graph-lanes: ${lanes.length};">
      <svg class="graph-edge-layer" aria-hidden="true"></svg>
      ${nodeMarkup}
    </div>
    <div class="graph-legend">${escapeHtml(graph.run_id || "run id not reported")}</div>
  `;
  window.requestAnimationFrame(() => drawCognitionGraphEdges(container, edges));
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

function cognitionGraphNodeMarkup(node, lanes) {
  const column = Math.max(1, Number(node.column) || 1);
  const lane = node.lane || "cognition";
  const row = Math.max(1, lanes.indexOf(lane) + 1);
  const detail = cognitionGraphDetail(node.detail || {});
  const branch = node.branch ? `<span>${escapeHtml(node.branch)}</span>` : "";
  const status = node.status || "not_reported";
  return `
    <button class="graph-node status-${escapeHtml(status)}" type="button" data-node-id="${escapeHtml(node.id)}" style="grid-column: ${column}; grid-row: ${row};">
      <span class="node-stage">${escapeHtml(node.stage || "stage")}</span>
      <strong>${escapeHtml(node.label || node.id)}</strong>
      <span class="node-meta">${escapeHtml(lane)}${branch}</span>
      <span class="node-detail">${detail}</span>
    </button>
  `;
}

function cognitionGraphDetail(detail) {
  const rows = Object.entries(detail)
    .filter(([, value]) => value !== null && value !== undefined && value !== "")
    .slice(0, 6)
    .map(([key, value]) => `<span><strong>${escapeHtml(key)}</strong>${escapeHtml(cognitionGraphValue(value))}</span>`);
  return rows.length ? rows.join("") : "<span>No reasoning detail reported.</span>";
}

function cognitionGraphValue(value) {
  if (Array.isArray(value)) return value.join(", ");
  if (value && typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function drawCognitionGraphEdges(container, edges) {
  const stage = container.querySelector(".cognition-graph-stage");
  const svg = container.querySelector(".graph-edge-layer");
  if (!stage || !svg) return;

  const bounds = stage.getBoundingClientRect();
  svg.setAttribute("viewBox", `0 0 ${bounds.width} ${bounds.height}`);
  svg.innerHTML = "";
  edges.forEach((edge) => {
    const source = stage.querySelector(`[data-node-id="${CSS.escape(edge.source)}"]`);
    const target = stage.querySelector(`[data-node-id="${CSS.escape(edge.target)}"]`);
    if (!source || !target) return;

    const sourceBounds = source.getBoundingClientRect();
    const targetBounds = target.getBoundingClientRect();
    const x1 = sourceBounds.left - bounds.left + sourceBounds.width;
    const y1 = sourceBounds.top - bounds.top + sourceBounds.height / 2;
    const x2 = targetBounds.left - bounds.left;
    const y2 = targetBounds.top - bounds.top + targetBounds.height / 2;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", x1);
    line.setAttribute("y1", y1);
    line.setAttribute("x2", x2);
    line.setAttribute("y2", y2);
    line.setAttribute("class", `graph-edge edge-${edge.kind || "sequence"}`);
    svg.appendChild(line);
  });
}

function renderCapabilitySummary() {
  const labels = {
    overview: "Overview",
    services: "Services",
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

function serviceStatus(serviceId) {
  const service = state.services.find((item) => item.id === serviceId);
  return service ? service.actual_state : "unavailable";
}

function dependenciesAvailable(service) {
  return (service.dependencies || []).every((serviceId) => ["running", "conflict"].includes(serviceStatus(serviceId)));
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

function renderServices() {
  const grid = qs("#service-grid");
  grid.innerHTML = "";
  state.services.forEach((service) => {
    const startButton = serviceActionButton(service, "start", "Start", "primary");
    const restartButton = serviceActionButton(service, "restart", "Restart");
    const stopButton = serviceActionButton(service, "stop", "Stop", "danger");
    const serviceError = service.last_error_preview ? `<div class="service-error">${escapeHtml(service.last_error_preview)}</div>` : "";
    grid.insertAdjacentHTML("beforeend", `
      <article class="service-card" data-component="Card" data-service-card="${escapeHtml(service.id)}">
        <div class="service-card-header">
          <div><strong>${escapeHtml(service.display_name)}</strong><br><code>${escapeHtml(service.id)}</code></div>
          <span class="${badgeClass(service.actual_state)}">${escapeHtml(service.actual_state)}</span>
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
  button.disabled = true;
  try {
    await api(`/api/services/${serviceId}/${action}`, {
      method: "POST",
      csrf: true,
      body: JSON.stringify({reason: "operator console action", expected_version: expectedVersion}),
    });
    await bootstrap();
  } catch (error) {
    await bootstrap();
    throw error;
  } finally {
    button.disabled = false;
  }
}

async function sendDebug(event) {
  event.preventDefault();
  const form = new FormData(event.target);
  const payload = Object.fromEntries(form.entries());
  const selectedMode = form.get("debug_mode");
  const debugModes = form.getAll("debug_modes");
  if (selectedMode && selectedMode !== "visible_reply") debugModes.push(selectedMode);
  payload.debug_modes = debugModes;
  delete payload.debug_mode;
  const result = await api("/api/debug-chat", {method: "POST", csrf: true, body: JSON.stringify(payload)});
  state.debugCognitionGraph = result.cognition_graph || null;
  const label = result.brain_available ? "brain" : "unavailable";
  const body = debugResponseBody(result);
  const meta = debugResponseMeta(result);
  qs("#chat-history").insertAdjacentHTML("beforeend", `<article class="message"><div class="meta">${escapeHtml(label)}</div><p>${escapeHtml(body)}</p><div class="meta">${escapeHtml(meta)}</div></article>`);
  renderDebugCognitionGraph(state.debugCognitionGraph);
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
  const globalUserId = qs("#memory-global-user-id").value.trim();
  const query = qs("#memory-query").value.trim();
  if (!globalUserId) {
    qs("#memory-status").textContent = "needs input";
    qs("#memory-status").className = "badge";
    if (showNeedsInput) {
      qs("#memory-table").innerHTML = "<tr><td>Status</td><td>Enter a global user id to load scoped memory units.</td></tr>";
    }
    return;
  }

  const params = new URLSearchParams({global_user_id: globalUserId, limit: "25"});
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
  const globalUserId = qs("#style-global-user-id").value.trim();
  const platform = qs("#style-platform").value.trim();
  const groupId = qs("#style-channel-id").value.trim();
  if (!globalUserId && !groupId) {
    qs("#style-status").textContent = "needs input";
    qs("#style-status").className = "badge";
    if (showNeedsInput) {
      qs("#style-table").innerHTML = "<tr><td>Status</td><td>Enter a global user id or group scope to load interaction-style guidance.</td></tr>";
    }
    return;
  }

  const params = new URLSearchParams({limit: "25"});
  if (globalUserId) params.set("global_user_id", globalUserId);
  if (platform) params.set("platform", platform);
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

function openStream(url) {
  if (state.eventSource) state.eventSource.close();
  state.eventSource = new EventSource(url);
  state.eventSource.addEventListener("control.gap", () => bootstrap());
  state.eventSource.addEventListener("control.cognition_graph_invalidated", () => bootstrap());
}

initializeTheme();
qsa("[data-page-link]").forEach((link) => link.addEventListener("click", () => setPage(link.dataset.pageLink)));
qsa("[data-theme-choice]").forEach((button) => button.addEventListener("click", () => setTheme(button.dataset.themeChoice)));
qs("#login").addEventListener("click", () => login().catch((error) => alert(error.message)));
qs("#token").addEventListener("keydown", (event) => {
  if (event.key === "Enter") login().catch((error) => alert(error.message));
});
qs("#service-grid").addEventListener("click", (event) => serviceAction(event).catch((error) => alert(error.message)));
qs("#debug-form").addEventListener("submit", (event) => sendDebug(event).catch((error) => alert(error.message)));
qs("#refresh-events").addEventListener("click", () => refreshEvents().catch((error) => alert(error.message)));
qs("#refresh-memory").addEventListener("click", () => refreshMemory().catch((error) => alert(error.message)));
qs("#refresh-style").addEventListener("click", () => refreshStyle().catch((error) => alert(error.message)));
qs("#refresh-calendar").addEventListener("click", () => refreshCalendar().catch((error) => alert(error.message)));
qs("#refresh-background").addEventListener("click", () => refreshBackground().catch((error) => alert(error.message)));
window.addEventListener("resize", () => {
  renderOverviewCognitionGraph(state.latestCognitionGraph);
  renderDebugCognitionGraph(state.debugCognitionGraph);
});
resumeSession();
