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
  latestSelfCognitionGraph: null,
  latestCognitionChainRun: null,
  latestSelfCognitionChainRun: null,
  debugCognitionGraph: null,
  debugCognitionChainRun: null,
  userDirectory: [],
  groupDirectory: [],
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
const COGNITION_ENGINE_DESCRIPTOR_SCHEMA = "cognition_engine_descriptor.v1";
const COGNITION_ENGINE_DESCRIPTOR_FIELDS = [
  "engine_id",
  "chain_model_name",
  "sidecar_model_name",
  "sidecar_enabled",
  "subconscious_enabled",
  "appraisal_group_count",
  "chain_context_window_tokens",
  "normal_budget_tokens",
  "extended_budget_tokens",
  "turn_deadline_seconds",
];

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

function formatOperationalLabel(value) {
  return formatLookupLabel(value).replaceAll(".", " · ");
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

function booleanStatus(value) {
  if (value === true) return "yes";
  if (value === false) return "no";
  return "not reported";
}

function formatPercent(value) {
  const numberValue = Number(value);
  if (!Number.isFinite(numberValue)) return "not reported";
  return `${Math.round(numberValue * 1000) / 10}%`;
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
  clearNotice();
  qsa("[data-page]").forEach((page) => page.classList.toggle("active", page.dataset.page === name));
  qsa("[data-page-link]").forEach((link) => link.classList.toggle("active", link.dataset.pageLink === name));
  if (name === "logs" && state.csrfHeaderName) {
    renderLogControls();
    openLogStream();
  } else {
    closeLogStream();
  }
  if (name === "overview" && state.csrfHeaderName) refreshOverview().catch(reportActionError);
  if (name === "audit" && state.csrfHeaderName) refreshAudit().catch(reportActionError);
  if (name === "character" && state.csrfHeaderName) refreshCharacter().catch(reportActionError);
  if (name === "users" && state.csrfHeaderName) refreshUsers(false).catch(reportActionError);
  if (name === "groups" && state.csrfHeaderName) refreshGroups(false).catch(reportActionError);
  if (name === "calendar" && state.csrfHeaderName) refreshCalendar().catch(reportActionError);
  if (name === "background" && state.csrfHeaderName) refreshBackground().catch(reportActionError);
  if (name === "health" && state.csrfHeaderName) refreshHealth().catch(reportActionError);
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
  if (["available", "completed", "healthy", "ok", "running", "succeeded"].includes(status)) return "badge success";
  if (["conflict", "crashed", "unhealthy"].includes(status) || ["failed", "unavailable"].includes(status)) return "badge danger";
  if (["partial", "requested", "starting", "stopping"].includes(status)) return "badge warn";
  return "badge";
}

function renderShellStatus(payload) {
  const dot = qs(".status-dot");
  const statusText = qs("#shell-status-text");
  if (!dot || !statusText) return;
  if (!state.isAuthenticated) {
    dot.dataset.state = "locked";
    setText(statusText, "Sign in to inspect local services.");
    return;
  }

  dot.dataset.state = "authenticated";
  const operatorId = payload.operator?.operator_id || "local operator";
  setText(statusText, `Signed in as ${operatorId}.`);
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
  state.latestCognitionGraph = payload.latest_cognition_graph || null;
  state.latestSelfCognitionGraph = payload.latest_self_cognition_graph || null;
  state.latestCognitionChainRun = payload.latest_cognition_chain_run || notReportedCognitionChainRun();
  state.latestSelfCognitionChainRun = payload.latest_self_cognition_chain_run || notReportedCognitionChainRun();
  setAuthState(true);
  setText("#session-state", payload.operator ? payload.operator.operator_id : "signed in");
  renderBrand(payload.application_identity || {});
  renderPageCapabilities();
  renderShellStatus(payload);
  renderDebugAvailability();
  renderDebugCognitionChain(state.debugCognitionChainRun || notReportedCognitionChainRun());
  renderOverview(payload);
  renderHealth(payload.health || {});
  renderServices();
  await refreshBrainModelRoutes({silent: true});
  renderLogControls();
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
  state.latestSelfCognitionGraph = null;
  state.latestCognitionChainRun = null;
  state.latestSelfCognitionChainRun = null;
  state.debugCognitionGraph = null;
  state.debugCognitionChainRun = null;
  state.userDirectory = [];
  state.groupDirectory = [];
  if (state.eventSource) state.eventSource.close();
  closeLogStream();
  state.eventSource = null;
  state.streamUrl = "";
  setAuthState(false);
  setText("#session-state", "signed out");
  renderBrand({status: "unavailable", character_name: "not connected"});
  renderDebugAvailability();
  renderDebugCognitionChain(notReportedCognitionChainRun());
  renderCognitionEngineDescriptor(null);
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
  const overview = payload.overview || payload;
  const panels = overview.panels || {};
  state.latestCognitionGraph = overview.latest_cognition_graph || null;
  state.latestSelfCognitionGraph = overview.latest_self_cognition_graph || null;
  state.latestCognitionChainRun = overview.latest_cognition_chain_run || notReportedCognitionChainRun();
  state.latestSelfCognitionChainRun = overview.latest_self_cognition_chain_run || notReportedCognitionChainRun();
  const graphItems = panelItems(panels.cognition_graphs);
  const conversationGraph = graphItems.find((item) => item.graph_kind === "conversation");
  const selfCognitionGraph = graphItems.find((item) => item.graph_kind === "self_cognition");
  if (conversationGraph?.graph) state.latestCognitionGraph = conversationGraph.graph;
  if (selfCognitionGraph?.graph) state.latestSelfCognitionGraph = selfCognitionGraph.graph;
  if (conversationGraph?.chain_run) state.latestCognitionChainRun = conversationGraph.chain_run;
  if (selfCognitionGraph?.chain_run) state.latestSelfCognitionChainRun = selfCognitionGraph.chain_run;
  const servicePanel = panels.service_summary || {};
  const serviceSummary = firstObjectItem(panelItems(servicePanel));
  setEntityStatus("#overview-service-status", servicePanel.status || "unavailable");
  setHtml("#overview-service-summary-table", overviewServiceSummaryRows(serviceSummary, servicePanel));

  const readinessPanel = panels.internal_readiness || {};
  const readiness = firstObjectItem(panelItems(readinessPanel));
  setEntityStatus("#overview-readiness-status", readinessPanel.status || "unavailable");
  setHtml("#overview-readiness-table", overviewReadinessRows(readiness, readinessPanel));

  const failuresPanel = panels.recent_failures || {};
  setEntityStatus("#overview-failures-status", failuresPanel.status || "empty");
  setHtml("#overview-failures-table", overviewFailureRows(failuresPanel));

  const changesPanel = panels.recent_changes || {};
  setEntityStatus("#overview-changes-status", changesPanel.status || "empty");
  setHtml("#overview-changes-table", overviewChangeRows(changesPanel));
  renderOverviewCognitionGraph(state.latestCognitionGraph);
  renderOverviewSelfCognitionGraph(state.latestSelfCognitionGraph);
  renderCognitionChainRun({
    containerSelector: "#overview-cognition-chain",
    statusSelector: "#overview-cognition-chain-status",
    snapshot: state.latestCognitionChainRun,
  });
  renderCognitionChainRun({
    containerSelector: "#overview-self-cognition-chain",
    statusSelector: "#overview-self-cognition-chain-status",
    snapshot: state.latestSelfCognitionChainRun,
  });
  renderCognitionEngineDescriptor(overview.cognition_engine);
}

async function refreshOverview() {
  const payload = await api("/api/overview");
  renderOverview(payload);
}

function overviewServiceSummaryRows(summary, panel) {
  if (!Object.keys(summary).length) {
    return `<tr><td>Status</td><td>${escapeHtml(panelEmptyText(panel, "Service totals are unavailable."))}</td></tr>`;
  }
  const rows = [
    ["Managed", summary.managed_services],
    ["Running", summary.running],
    ["Stopped", summary.stopped],
    ...[
      ["Starting", summary.starting],
      ["Stopping", summary.stopping],
      ["Unhealthy", summary.unhealthy],
      ["Crashed", summary.crashed],
      ["Conflict", summary.conflict],
      ["Unavailable", summary.unavailable],
    ].filter(([, value]) => Number(value) > 0),
  ];
  return rows.map(([label, value]) => `<tr><td>${escapeHtml(label)}</td><td>${escapeHtml(formatLookupValue(value))}</td></tr>`).join("");
}

function overviewReadinessRows(readiness, panel) {
  if (!Object.keys(readiness).length) {
    return `<tr><td>Status</td><td>${escapeHtml(panelEmptyText(panel, "Internal readiness is unavailable."))}</td></tr>`;
  }
  return [
    ["Overall", readiness.status],
    ["Database", booleanStatus(readiness.database)],
    ["Scheduler", booleanStatus(readiness.scheduler)],
    ["Worker error level", readiness.worker_error_level],
  ].map(([label, value]) => `<tr><td>${escapeHtml(label)}</td><td>${escapeHtml(formatLookupValue(value))}</td></tr>`).join("");
}

function overviewFailureRows(panel) {
  const items = panelItems(panel);
  if (!items.length) {
    return `<tr><td colspan="4">${escapeHtml(panelEmptyText(panel, "No recent failures."))}</td></tr>`;
  }
  return items.map((item) => `
    <tr>
      <td>${escapeHtml(formatLookupValue(item.created_at))}</td>
      <td>${escapeHtml(formatLookupValue(item.target))}</td>
      <td><span class="${badgeClass(item.outcome)}">${escapeHtml(formatLookupValue(item.outcome))}</span></td>
      <td>${escapeHtml(formatLookupValue(item.reason))}</td>
    </tr>
  `).join("");
}

function overviewChangeRows(panel) {
  const items = panelItems(panel);
  if (!items.length) {
    return `<tr><td colspan="4">${escapeHtml(panelEmptyText(panel, "No recent changes."))}</td></tr>`;
  }
  return items.map((item) => `
    <tr>
      <td>${escapeHtml(formatLookupValue(item.created_at))}</td>
      <td>${escapeHtml(formatLookupValue(item.action))}</td>
      <td>${escapeHtml(formatLookupValue(item.target))}</td>
      <td><span class="${badgeClass(item.outcome)}">${escapeHtml(formatLookupValue(item.outcome))}</span></td>
    </tr>
  `).join("");
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

function renderDebugCognitionChain(snapshot) {
  renderCognitionChainRun({
    containerSelector: "#debug-cognition-chain",
    statusSelector: "#debug-cognition-chain-status",
    snapshot,
  });
}

function notReportedCognitionChainRun() {
  return {
    status: "not_reported",
    chain_run_id: null,
    run_id: null,
    llm_trace_id: null,
    cognition_invocation_id: null,
    chain_model_name: "",
    sidecar_model_name: "",
    terminal_disposition: "",
    started_at: "",
    completed_at: "",
    step_count: 0,
    warning_codes: [],
  };
}

function renderCognitionChainRun({containerSelector, statusSelector, snapshot}) {
  const container = qs(containerSelector);
  const statusElement = qs(statusSelector);
  if (!container || !statusElement) return;

  const chain = snapshot && typeof snapshot === "object"
    ? snapshot
    : notReportedCognitionChainRun();
  const chainStatus = String(chain.status || "not_reported");
  statusElement.textContent = formatLookupLabel(chainStatus);
  statusElement.className = cognitionGraphStatusBadgeClass(chainStatus);
  const fields = [
    ["status", chainStatus],
    ["chain_run_id", chain.chain_run_id],
    ["run_id", chain.run_id],
    ["llm_trace_id", chain.llm_trace_id],
    ["cognition_invocation_id", chain.cognition_invocation_id],
    ["chain_model_name", chain.chain_model_name],
    ["sidecar_model_name", chain.sidecar_model_name],
    ["terminal_disposition", chain.terminal_disposition],
    ["started_at", chain.started_at],
    ["completed_at", chain.completed_at],
    ["step_count", chain.step_count],
    ["warning_codes", chain.warning_codes],
  ];
  const rows = fields.map(([fieldName, fieldValue]) => `
    <tr>
      <th scope="row">${escapeHtml(fieldName)}</th>
      <td>${escapeHtml(cognitionChainFieldValue(fieldName, fieldValue, chainStatus))}</td>
    </tr>
  `).join("");
  setHtml(container, `
    <div class="cognition-chain-table-wrap table-wrap flush" data-chain-status="${escapeHtml(chainStatus)}">
      <table class="cognition-chain-table">
        <tbody>${rows}</tbody>
      </table>
    </div>
  `);
}

function cognitionChainFieldValue(fieldName, value, chainStatus) {
  if (chainStatus === "not_reported") return "not_reported";
  if (fieldName === "warning_codes") {
    if (!Array.isArray(value) || !value.length) return "not_reported";
    const warningCodes = value.filter((item) => typeof item === "string" && item.trim());
    return warningCodes.length ? warningCodes.join(", ") : "not_reported";
  }
  if (fieldName === "step_count" && Number.isInteger(value)) return String(value);
  if (typeof value === "string" && value.trim()) return value;
  return "not_reported";
}

function projectCognitionEngineDescriptor(rawDescriptor) {
  if (!rawDescriptor || typeof rawDescriptor !== "object") {
    return {status: "not_reported"};
  }
  if (rawDescriptor.schema_version !== COGNITION_ENGINE_DESCRIPTOR_SCHEMA) {
    return {status: "not_reported"};
  }
  if (!COGNITION_ENGINE_DESCRIPTOR_FIELDS.every((fieldName) => (
    Object.prototype.hasOwnProperty.call(rawDescriptor, fieldName)
  ))) {
    return {status: "not_reported"};
  }
  const descriptor = {
    status: "available",
    schema_version: COGNITION_ENGINE_DESCRIPTOR_SCHEMA,
  };
  COGNITION_ENGINE_DESCRIPTOR_FIELDS.forEach((fieldName) => {
    descriptor[fieldName] = rawDescriptor[fieldName];
  });
  return descriptor;
}

function renderCognitionEngineDescriptor(rawDescriptor) {
  const container = qs("#overview-cognition-engine");
  const descriptor = projectCognitionEngineDescriptor(rawDescriptor);
  if (!container) return;
  const descriptorStatus = descriptor.status || "not_reported";
  setEntityStatus("#overview-cognition-engine-status", descriptorStatus);
  const fields = COGNITION_ENGINE_DESCRIPTOR_FIELDS.map((fieldName) => [
    fieldName,
    descriptor[fieldName],
  ]);
  const rows = fields.map(([fieldName, fieldValue]) => `
    <tr>
      <th scope="row">${escapeHtml(fieldName)}</th>
      <td>${escapeHtml(cognitionEngineFieldValue(fieldValue, descriptorStatus))}</td>
    </tr>
  `).join("");
  setHtml(container, `
    <div class="cognition-chain-table-wrap table-wrap flush" data-engine-status="${escapeHtml(descriptorStatus)}">
      <table class="cognition-chain-table cognition-engine-table">
        <tbody>${rows}</tbody>
      </table>
    </div>
  `);
}

function cognitionEngineFieldValue(value, descriptorStatus) {
  if (descriptorStatus !== "available") return "not_reported";
  if (value === null || value === undefined || value === "") {
    return "not_reported";
  }
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number") {
    return Number.isFinite(value) ? String(value) : "not_reported";
  }
  if (typeof value === "string" && value.trim()) return value;
  return "not_reported";
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
      ${cognitionGraphParallelSummaryMarkup(model)}
      ${cognitionGraphDependencyMarkup(model)}
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

function renderOverviewSelfCognitionGraph(snapshot) {
  const card = qs("#overview-self-cognition-card");
  if (!card) return;
  const hasReportedGraph = Boolean(
    snapshot && snapshot.status && snapshot.status !== "not_reported",
  );
  card.hidden = !hasReportedGraph;
  if (!hasReportedGraph) return;
  renderCognitionGraph({
    containerSelector: "#overview-self-cognition-graph",
    statusSelector: "#overview-self-cognition-status",
    snapshot,
    emptyMessage: "No latest self-cognition graph has been reported by the brain.",
  });
}

function cognitionGraphSummaryMarkup(model) {
  const current = model.currentNode;
  const status = cognitionGraphStatusLabel(model.graph.status || "not_reported");
  const sourceLabel = cognitionGraphSourceLabel(model.source);
  const currentLabel = current
    ? `${model.focusKind} · ${current.stage || "stage"} · ${current.label || current.id}`
    : "no current node";
  return `
    <div class="graph-run-summary">
      <div class="graph-run-title">
        <strong>${escapeHtml(sourceLabel)}</strong>
        ${renderReferenceDisclosure(
          "Run reference",
          cognitionGraphReferenceEntries(model),
        )}
      </div>
      <div class="badge-stack">
        <span class="${escapeHtml(cognitionGraphStatusBadgeClass(model.graph.status || "not_reported"))}" data-component="Badge">${escapeHtml(status)}</span>
        <span class="badge${model.stale ? " warn" : ""}" data-component="Badge">${escapeHtml(model.freshness)}</span>
        <span class="badge" data-component="Badge">${escapeHtml(currentLabel)}</span>
      </div>
    </div>
  `;
}

function cognitionGraphReferenceEntries(model) {
  const entries = [["run_id", model.graph.run_id]];
  if (model.source === "self_latest") {
    entries.push(
      ["child_llm_trace_id", model.graph.llm_trace_id],
      ["source_calendar_run_id", model.graph.source_calendar_run_id],
    );
  } else {
    entries.push(
      ["llm_trace_id", model.graph.llm_trace_id],
      ["cognition_invocation_id", model.graph.cognition_invocation_id],
    );
  }
  return entries;
}

function cognitionGraphSourceLabel(source) {
  const labels = {
    overview_latest: "Latest conversation cognition",
    debug_latest: "Current debug cognition",
    self_latest: "Latest self-cognition",
    historical: "Historical cognition",
  };
  return labels[source] || "Cognition run";
}

function cognitionGraphParallelSummaryMarkup(model) {
  const executionNode = model.nodes.find((node) => node.id === "v2.parallel");
  const branchNodes = model.nodes.filter((node) => node.id.startsWith("v2.branch."));
  if (!executionNode && !branchNodes.length) return "";
  const execution = executionNode?.detail?.parallel_execution || {};
  const metrics = [
    ["max concurrency", execution.maximum_concurrency],
    ["completed", execution.completed_branch_count],
    ["failed", execution.failed_branch_count],
    ["overlap", execution.overlap_ms === undefined ? null : `${execution.overlap_ms} ms`],
  ].filter(([, value]) => value !== null && value !== undefined && value !== "");
  const metricMarkup = metrics.map(([label, value]) => (
    `<span class="graph-parallel-metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong></span>`
  )).join("");
  const branchMarkup = branchNodes.map((node) => {
    const detail = node.detail || {};
    const selection = detail.selection || "unselected";
    const status = cognitionGraphStatusLabel(node.status || "not_reported");
    const semantic = detail.intention || detail.desired_outcome || detail.reason || "No branch result reported.";
    return `
      <button class="graph-parallel-result status-${escapeHtml(node.status || "not_reported")}" type="button" data-graph-node data-node-id="${escapeHtml(node.id)}" aria-label="${escapeHtml(node.label || node.id)}">
        <span class="graph-parallel-result-header">
          <strong>${escapeHtml(node.label || node.id)}</strong>
          <span class="badge" data-component="Badge">${escapeHtml(selection)} · ${escapeHtml(status)}</span>
        </span>
        <span>${escapeHtml(cognitionGraphPreview(cognitionGraphValue(semantic), 180))}</span>
      </button>
    `;
  }).join("");
  return `
    <section class="graph-parallel-summary" aria-label="Parallel cognition results" data-component="Parallel cognition results">
      <div class="graph-parallel-header">
        <div>
          <span>Native V2</span>
          <strong>Parallel cognition results</strong>
        </div>
        <div class="graph-parallel-metrics">${metricMarkup}</div>
      </div>
      <div class="graph-parallel-results">
        ${branchMarkup || `<p class="graph-parallel-empty">No branch result was reported.</p>`}
      </div>
    </section>
  `;
}

function cognitionGraphDependencyMarkup(model) {
  if (!model.edges.length) return "";
  const labelById = new Map(model.nodes.map((node) => [node.id, node.label || node.id]));
  const edges = [...model.edges].sort((left, right) => {
    const leftNative = left.source?.startsWith("v2.") || left.target?.startsWith("v2.");
    const rightNative = right.source?.startsWith("v2.") || right.target?.startsWith("v2.");
    return Number(rightNative) - Number(leftNative);
  });
  const edgeMarkup = edges.map((edge) => {
    const source = labelById.get(edge.source) || edge.source;
    const target = labelById.get(edge.target) || edge.target;
    const kind = String(edge.kind || "sequence").replaceAll("_", " ");
    const label = edge.label ? ` · ${edge.label}` : "";
    return `<div class="graph-dependency-row"><strong>${escapeHtml(source)} → ${escapeHtml(target)}</strong><span>${escapeHtml(`${kind}${label}`)}</span></div>`;
  }).join("");
  return `
    <section class="graph-dependency-panel" aria-label="Cognition graph dependencies">
      <div class="graph-dependency-header">
        <span>Recorded relationships</span>
        <strong>Fork / join dependencies</strong>
      </div>
      <div class="graph-dependency-list">${edgeMarkup}</div>
    </section>
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
  const detailMarkup = rows
    ? `<div class="graph-inspector-rows">${rows}</div>`
    : `<p class="graph-inspector-empty">No approved semantic detail was reported for this node.</p>`;
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
      ${detailMarkup}
      <div class="graph-inspector-actions">
        ${showReturn ? `<button class="btn" type="button" data-graph-return-current>Return to current</button>` : ""}
      </div>
    </aside>
  `;
}

function cognitionGraphInspectorRows(node) {
  if (!node) return [];
  const detail = node.detail || {};
  const rows = [];
  const fieldOrder = [
    ["input", "Input"],
    ["summary", "Summary"],
    ["reply_context", "Reply context"],
    ["decision", "Decision"],
    ["reasoning", "Reasoning"],
    ["parallel_execution", "Parallel execution"],
    ["failure_code", "Failure code"],
    ["appraisal_results", "Appraisal results"],
    ["branch_results", "Branch results"],
    ["phase", "Phase"],
    ["goal_kind", "Goal kind"],
    ["selection", "Selection"],
    ["intention", "Intention"],
    ["desired_outcome", "Desired outcome"],
    ["concrete_detail", "Concrete detail"],
    ["reason", "Reason"],
    ["internal_monologue", "Internal monologue"],
    ["private_monologue", "Private monologue"],
    ["expected_consequences", "Expected consequences"],
    ["confidence", "Confidence"],
    ["collapse", "Collapse"],
    ["failure", "Failure"],
    ["selected_intention", "Selected intention"],
    ["selected_bid_reason", "Selected bid reason"],
    ["affect_projection", "Affect projection"],
    ["expression_policy", "Expression policy"],
    ["goal_resolution", "Goal resolution"],
    ["logical_stance", "Logical stance"],
    ["character_intent", "Character intent"],
    ["judgment_note", "Judgment note"],
    ["retrieval_answer", "Retrieval answer"],
    ["memory_evidence", "Memory evidence"],
    ["conversation_evidence", "Conversation evidence"],
    ["external_evidence", "External evidence"],
    ["recall_evidence", "Recall evidence"],
    ["media_evidence", "Media evidence"],
    ["user_continuity", "User continuity"],
    ["conversation_progress", "Conversation progress"],
    ["public_group_scene", "Public group scene"],
    ["active_commitments", "Active commitments"],
    ["selected_actions", "Selected actions"],
    ["action_results", "Action results"],
    ["action_continuation", "Action continuation"],
    ["context_consumption", "Context consumption"],
    ["facial_expression", "Facial expression"],
    ["body_language", "Body language"],
    ["gaze_direction", "Gaze direction"],
    ["visual_vibe", "Visual vibe"],
    ["messages", "Messages"],
    ["empty_state", "State"],
  ];
  fieldOrder.forEach(([key, label]) => {
    if (cognitionGraphValuePresent(detail[key])) rows.push([label, detail[key]]);
  });
  return rows;
}

function cognitionGraphNodeSummary(node) {
  const detail = node.detail || {};
  const value = cognitionGraphFirstSemanticValue(detail);
  if (value === null) return "No approved semantic detail reported.";
  return cognitionGraphPreview(cognitionGraphValue(value));
}

function cognitionGraphValue(value) {
  if (Array.isArray(value)) {
    return value.map((item) => {
      const rendered = cognitionGraphValue(item);
      return rendered.split("\n").map((line, index) => (
        `${index === 0 ? "• " : "  "}${line}`
      )).join("\n");
    }).join("\n");
  }
  if (value && typeof value === "object") {
    return Object.entries(value).map(([key, nested]) => {
      const rendered = cognitionGraphValue(nested);
      const indented = rendered.split("\n").map((line, index) => (
        `${index === 0 ? "" : "  "}${line}`
      )).join("\n");
      return `${key.replaceAll("_", " ")}: ${indented}`;
    }).join("\n");
  }
  if (value === null || value === undefined) return "";
  return String(value);
}

function cognitionGraphValuePresent(value) {
  if (value === null || value === undefined || value === "") return false;
  if (Array.isArray(value) || (value && typeof value === "object")) {
    return Object.keys(value).length > 0;
  }
  return true;
}

function cognitionGraphFirstSemanticValue(detail) {
  const fieldOrder = [
    "input",
    "reply_context",
    "summary",
    "decision",
    "reasoning",
    "parallel_execution",
    "appraisal_results",
    "branch_results",
    "phase",
    "goal_kind",
    "selection",
    "intention",
    "desired_outcome",
    "concrete_detail",
    "reason",
    "internal_monologue",
    "private_monologue",
    "expected_consequences",
    "confidence",
    "collapse",
    "selected_intention",
    "selected_bid_reason",
    "affect_projection",
    "expression_policy",
    "goal_resolution",
    "logical_stance",
    "character_intent",
    "judgment_note",
    "retrieval_answer",
    "memory_evidence",
    "conversation_evidence",
    "external_evidence",
    "recall_evidence",
    "media_evidence",
    "user_continuity",
    "conversation_progress",
    "public_group_scene",
    "active_commitments",
    "selected_actions",
    "action_results",
  "action_continuation",
    "context_consumption",
    "facial_expression",
    "body_language",
    "gaze_direction",
    "visual_vibe",
    "messages",
    "empty_state",
  ];
  const key = fieldOrder.find((candidate) => cognitionGraphValuePresent(detail[candidate]));
  return key ? detail[key] : null;
}

function cognitionGraphPreview(value, maxLength = 180) {
  const compact = String(value || "").replace(/\s+/g, " ").trim();
  if (compact.length <= maxLength) return compact;
  return `${compact.slice(0, maxLength)}…`;
}

function renderHealth(payload) {
  const panels = payload.panels || {};
  setEntityStatus("#health-status", payload.status || "unavailable");

  const readinessPanel = panels.readiness || {};
  const readiness = firstObjectItem(panelItems(readinessPanel));
  setEntityStatus("#health-readiness-status", readinessPanel.status || "unavailable");
  setHtml("#health-readiness-table", healthReadinessRows(readiness, readinessPanel));

  const workersPanel = panels.workers || {};
  const workers = panelItems(workersPanel);
  setEntityStatus("#health-workers-status", workersPanel.status || "empty");
  setHtml("#health-workers-table", workers.length
    ? workers.map((worker) => `
      <tr>
        <td>${escapeHtml(formatOperationalLabel(worker.worker_name))}</td>
        <td>${escapeHtml(booleanStatus(worker.enabled))}</td>
        <td>${escapeHtml(booleanStatus(worker.task_alive))}</td>
        <td>${escapeHtml(formatOperationalLabel(worker.last_status))}</td>
        <td>${escapeHtml(formatLookupValue(worker.last_event_at))}</td>
      </tr>
    `).join("")
    : `<tr><td colspan="5">${escapeHtml(panelEmptyText(workersPanel, "No worker state was reported."))}</td></tr>`);

  const cachePanel = panels.cache_agents || {};
  const agents = panelItems(cachePanel);
  setEntityStatus("#health-cache-status", cachePanel.status || "empty");
  setHtml("#health-cache-table", agents.length
    ? agents.map((agent) => `
      <tr>
        <td>${escapeHtml(formatOperationalLabel(agent.agent_name))}</td>
        <td>${escapeHtml(formatLookupValue(agent.hits))}</td>
        <td>${escapeHtml(formatLookupValue(agent.misses))}</td>
        <td>${escapeHtml(formatLookupValue(agent.total))}</td>
        <td>${escapeHtml(formatPercent(agent.hit_rate))}</td>
      </tr>
    `).join("")
    : `<tr><td colspan="5">${escapeHtml(panelEmptyText(cachePanel, "No Cache2 agent statistics were reported."))}</td></tr>`);
}

function healthReadinessRows(readiness, panel) {
  if (!Object.keys(readiness).length) {
    return `<tr><td>Status</td><td>${escapeHtml(panelEmptyText(panel, "Readiness is unavailable."))}</td></tr>`;
  }
  return [
    ["Overall", readiness.status],
    ["Database", booleanStatus(readiness.database)],
    ["Scheduler", booleanStatus(readiness.scheduler)],
    ["Worker error level", readiness.worker_error_level],
  ].map(([label, value]) => `<tr><td>${escapeHtml(label)}</td><td>${escapeHtml(formatLookupValue(value))}</td></tr>`).join("");
}

async function refreshHealth() {
  const payload = await api("/api/health");
  renderHealth(payload);
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

function serviceActionBlockReason(service) {
  if (service.actual_state === "starting" || service.actual_state === "stopping") {
    return `Lifecycle action blocked while the service is ${service.actual_state}.`;
  }
  const unavailableDependencies = (service.dependencies || []).filter((serviceId) => {
    const dependency = serviceById(serviceId);
    return !dependency || (dependency.actual_state !== "running" && !isEndpointConflict(dependency));
  });
  if (service.actual_state !== "running" && unavailableDependencies.length) {
    return `Start blocked by ${unavailableDependencies.join(", ")}.`;
  }
  return "";
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
  const configButton = serviceConfigButton(service);
  const configBadge = serviceConfigBadge(service);
  const stateExplanation = brainServiceStateExplanation(service);
  const serviceErrorText = service.last_error_preview || stateExplanation;
  const serviceError = serviceErrorText ? `<div class="service-error">${escapeHtml(serviceErrorText)}</div>` : "";
  const actionBlockReason = serviceActionBlockReason(service);
  return `
    <article class="service-card brain-service-card" data-component="Card" data-service-card="${escapeHtml(service.id)}">
      <div class="service-card-header">
        <div><strong>${escapeHtml(service.display_name)}</strong><br><code>${escapeHtml(service.id)}</code></div>
        <div class="badge-stack">
          <span class="${badgeClass(service.actual_state)}" data-service-status-badge>${escapeHtml(service.actual_state)}</span>
          ${configBadge}
          <span class="badge">${escapeHtml(routes.length)} routes</span>
        </div>
      </div>
      <div class="brain-service-layout">
        <section class="brain-runtime-panel">
          <div class="brain-runtime-grid">
            <div class="kv"><span>desired</span><strong>${escapeHtml(service.desired_state)}</strong></div>
            <div class="kv"><span>override routes</span><strong>${escapeHtml(routeSummary.overrideCount)}</strong></div>
            <div class="kv"><span>families</span><strong>${escapeHtml(routeSummary.familyCount)}</strong></div>
          </div>
          <details>
            <summary>Process detail</summary>
            <div class="brain-runtime-grid">
              <div class="kv"><span>version</span><strong>${escapeHtml(service.version)}</strong></div>
              <div class="kv"><span>pid</span><strong>${escapeHtml(service.pid || "-")}</strong></div>
              <div class="kv"><span>depends</span><code>${escapeHtml((service.dependencies || []).join(", ") || "-")}</code></div>
            </div>
          </details>
          ${serviceError}
          ${actionBlockReason ? `<div class="service-error">${escapeHtml(actionBlockReason)}</div>` : ""}
          <div class="service-card-actions brain-runtime-actions">
            ${startButton}
            ${restartButton}
            ${stopButton}
            ${logsButton}
            ${configButton}
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

function brainServiceStateExplanation(service) {
  if (isEndpointConflict(service)) {
    return "Brain endpoint already running outside the console; lifecycle is unmanaged.";
  }
  if (service.actual_state === "conflict") {
    return "Brain has a stale lifecycle conflict; inspect the recorded process detail.";
  }
  return "";
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
  const isSelected = route.route_key === state.selectedBrainRouteKey;
  const selected = isSelected ? " selected" : "";
  const source = route.effective?.source || "default";
  const sourceClass = source === "override" ? "badge warn" : "badge";
  const family = route.diagnostics?.model_family || "unknown";
  const thinking = route.effective?.thinking_enabled ? "thinking" : "standard";
  const currentValue = `
      <code>${escapeHtml(route.effective?.model || "not configured")}</code>
      <span class="brain-route-meta">
        <span class="${sourceClass}">${escapeHtml(source)}</span>
        <span class="badge">${escapeHtml(family)}</span>
        <span class="badge">${escapeHtml(thinking)}</span>
      </span>
  `;
  return `
    <button class="brain-route-tile${selected}" data-brain-route-key="${escapeHtml(route.route_key)}" type="button" role="listitem">
      <span class="brain-route-name">${escapeHtml(route.label || route.route_key)}</span>
      ${currentValue}
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
  const actionBlockReason = serviceActionBlockReason(service);
  return `
    <article class="service-card" data-component="Card" data-service-card="${escapeHtml(service.id)}">
      <div class="service-card-header">
        <div><strong>${escapeHtml(service.display_name)}</strong><br><code>${escapeHtml(service.id)}</code></div>
        <div class="badge-stack">
          <span class="${badgeClass(service.actual_state)}" data-service-status-badge>${escapeHtml(service.actual_state)}</span>
          ${configBadge}
        </div>
      </div>
      <div class="service-card-body">
        <div class="kv"><span>desired</span><strong>${escapeHtml(service.desired_state)}</strong></div>
        <details>
          <summary>Process detail</summary>
          <div class="kv"><span>version</span><strong>${escapeHtml(service.version)}</strong></div>
          <div class="kv"><span>pid</span><strong>${escapeHtml(service.pid || "-")}</strong></div>
          <div class="kv"><span>depends</span><code>${escapeHtml((service.dependencies || []).join(", ") || "-")}</code></div>
        </details>
      </div>
      ${serviceError}
      ${actionBlockReason ? `<div class="service-error">${escapeHtml(actionBlockReason)}</div>` : ""}
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

function renderAudit(payload) {
  const actions = Array.isArray(payload.actions) ? payload.actions : [];
  setText("#audit-action-count", `${actions.length} actions`);
  setHtml("#audit-table", actions.length
    ? actions.map((action) => `
      <tr>
        <td>${escapeHtml(formatLookupValue(action.created_at))}</td>
        <td>${escapeHtml(formatLookupValue(action.action))}</td>
        <td>${escapeHtml(formatLookupValue(action.target_label))}</td>
        <td><span class="${badgeClass(action.outcome)}">${escapeHtml(formatLookupValue(action.outcome))}</span></td>
        <td>${escapeHtml(formatLookupValue(action.operator_id))}</td>
        <td>${escapeHtml(formatLookupValue(action.reason))}</td>
      </tr>
    `).join("")
    : "<tr><td colspan=\"6\">No state-changing actions matched the selected filters.</td></tr>");

  const outcomes = payload.facets?.outcomes || {};
  setHtml("#audit-outcome-facets", facetRows(outcomes, "No action outcomes matched."));
  const views = Array.isArray(payload.view_summary) ? payload.view_summary : [];
  setHtml("#audit-view-summary", views.length
    ? views.map((item) => `<tr><td>${escapeHtml(formatLookupValue(item.view))}</td><td>${escapeHtml(formatLookupValue(item.count))}</td></tr>`).join("")
    : "<tr><td>Status</td><td>No page views were recorded in this window.</td></tr>");
}

async function refreshAudit() {
  const params = new URLSearchParams({limit: "25"});
  const filters = [
    ["category", "#audit-category"],
    ["event_type", "#audit-event-type"],
    ["service_id", "#audit-service-id"],
    ["operator_id", "#audit-operator-id"],
    ["outcome", "#audit-outcome"],
    ["request_id", "#audit-request-id"],
  ];
  filters.forEach(([key, selector]) => {
    const value = getValue(selector).trim();
    if (value) params.set(key, value);
  });
  const since = getValue("#audit-since").trim();
  if (since) params.set("since", new Date(since).toISOString());
  const payload = await api(`/api/audit?${params.toString()}`);
  renderAudit(payload);
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
  if (validation.min_value !== undefined) parts.push(`minimum ${validation.min_value}`);
  if (validation.max_value !== undefined) parts.push(`maximum ${validation.max_value}`);
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
  state.debugCognitionChainRun = notReportedCognitionChainRun();
  appendChatMessage({
    label: "operator",
    body: messageText || "Debug message sent.",
    meta: "awaiting brain response",
  });
  renderDebugCognitionGraph(state.debugCognitionGraph);
  renderDebugCognitionChain(state.debugCognitionChainRun);
  try {
    const result = await api("/api/debug-chat", {method: "POST", csrf: true, body: JSON.stringify(payload)});
    state.debugCognitionGraph = result.cognition_graph || null;
    state.debugCognitionChainRun = result.cognition_chain_run || notReportedCognitionChainRun();
    const label = result.brain_available ? "brain" : "unavailable";
    const body = debugResponseBody(result);
    const meta = debugResponseMeta(result);
    appendChatMessage({label, body, meta});
    renderDebugCognitionGraph(state.debugCognitionGraph);
    renderDebugCognitionChain(state.debugCognitionChainRun);
  } catch (error) {
    state.debugCognitionGraph = failedDebugCognitionGraph(error);
    state.debugCognitionChainRun = notReportedCognitionChainRun();
    renderDebugCognitionGraph(state.debugCognitionGraph);
    renderDebugCognitionChain(state.debugCognitionChainRun);
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
        detail: {input: messageText || "message submitted"},
      },
      {
        id: "debug.cognition",
        label: "Cognition",
        stage: "Brain",
        lane: "cognition",
        column: 2,
        branch: "live",
        status: "running",
        detail: {empty_state: "Waiting for the debug cognition result."},
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
        detail: {empty_state: error.message},
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
  const traceId = firstPresentValue(result, ["llm_trace_id", "trace_id"]);
  const trackingId = firstPresentValue(result, [
    "delivery_tracking_id",
    "tracking_id",
  ]);
  if (traceId) parts.push(`trace ${traceId}`);
  else parts.push("trace unavailable");
  if (trackingId) parts.push(`tracking ${trackingId}`);
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
  if (!element) return;
  element.textContent = formatLookupLabel(status || "unavailable");
  element.className = badgeClass(status);
}

function renderPanelState(target, panel) {
  const element = typeof target === "string" ? qs(target) : target;
  if (!element) return;
  const status = panel?.status || "unavailable";
  const reason = panel?.reason || "No rows are available for this panel.";
  const generatedAt = panel?.generated_at || "";
  if (isTableBody(element)) {
    setHtml(element, `<tr><td>Status</td><td>${escapeHtml(formatLookupLabel(status))}</td></tr><tr><td>Reason</td><td>${escapeHtml(reason)}</td></tr>${generatedAt ? `<tr><td>Generated</td><td>${escapeHtml(generatedAt)}</td></tr>` : ""}`);
    return;
  }
  const generated = generatedAt ? ` Generated ${formatLookupValue(generatedAt)}.` : "";
  setHtml(element, `<p class="panel-empty"><strong>${escapeHtml(formatLookupLabel(status))}</strong>: ${escapeHtml(reason)}${escapeHtml(generated)}</p>`);
}

function renderLookupTable(target, {items = [], emptyText = "No rows available."} = {}) {
  const element = typeof target === "string" ? qs(target) : target;
  if (!element) return;
  if (!items.length) {
    if (!isTableBody(element)) {
      setHtml(element, `<p class="panel-empty">${escapeHtml(emptyText)}</p>`);
      return;
    }
    setHtml(element, `<tr><td>Status</td><td>${escapeHtml(emptyText)}</td></tr>`);
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

function isTableBody(element) {
  return element?.tagName === "TBODY";
}

function setSummaryMetric(selector, value, note) {
  const element = optionalElement(selector);
  if (!element) return;
  const valueElement = element.querySelector(".metric-value");
  const labelElements = element.querySelectorAll(".metric-label");
  if (valueElement) valueElement.textContent = formatLookupValue(value);
  if (labelElements.length > 1) labelElements[labelElements.length - 1].textContent = formatLookupValue(note);
}

function firstPresentValue(item, keys) {
  for (const key of keys) {
    const value = item?.[key];
    if (value !== null && value !== undefined && value !== "") return value;
  }
  return "";
}

function recordTitle(item, fallback = "record") {
  const title = firstPresentValue(item, [
    "run_id",
    "calendar_run_id",
    "calendar_schedule_id",
    "background_work_job_id",
    "job_id",
    "schedule_id",
    "event_id",
    "episode_id",
    "trigger_kind",
    "event_type",
    "unit_type",
    "source",
  ]);
  return formatLookupValue(title || fallback);
}

function recordDetailEntries(item, hiddenKeys = []) {
  const hidden = new Set(hiddenKeys);
  return Object.entries(item || {}).filter(([key, value]) => (
    !hidden.has(key) && value !== null && value !== undefined && value !== ""
  ));
}

function renderReferenceDisclosure(summary, entries) {
  const visibleEntries = entries.filter(([, value]) => (
    value !== null && value !== undefined && value !== ""
  ));
  if (!visibleEntries.length) return "";
  return `
    <details class="graph-run-reference">
      <summary>${escapeHtml(summary)}</summary>
      ${renderDetailGrid(visibleEntries)}
    </details>
  `;
}

function renderRecordCard(item, {title = "", status = "", hiddenKeys = [], body = "", chips = [], reference = "", referenceLabel = "Job reference", references = []} = {}) {
  const cardTitle = title || recordTitle(item);
  const statusText = status || item?.status || item?.delivery_state || "";
  const details = renderDetailGrid(recordDetailEntries(item, hiddenKeys));
  const chipRow = chips.length ? detailChipRow(chips) : "";
  const referenceEntries = references.length
    ? references
    : [[referenceLabel, reference]];
  const referenceMarkup = renderReferenceDisclosure(
    referenceLabel,
    referenceEntries,
  );
  return `
    <article class="record-card">
      <div class="record-card-header">
        <div><h4>${escapeHtml(formatLookupValue(cardTitle))}</h4></div>
        ${statusText ? `<span class="${badgeClass(statusText)}">${escapeHtml(formatLookupLabel(statusText))}</span>` : ""}
      </div>
      ${body ? `<p class="character-prose">${escapeHtml(formatCharacterProse(body))}</p>` : ""}
      ${chipRow}
      ${referenceMarkup}
      ${details}
    </article>
  `;
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

function renderPanelEmptyContent(target, {emptyText = "No rows available."} = {}) {
  const element = typeof target === "string" ? qs(target) : target;
  if (!element) return;
  setHtml(element, `<p class="panel-empty">${escapeHtml(emptyText)}</p>`);
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
  const selfConcept = item.self_concept || "";
  const growthEdges = Array.isArray(item.current_growth_edges)
    ? item.current_growth_edges
    : [];
  setHtml(element, `
    <section class="character-summary">
      ${selfConcept ? `
        <section class="detail-section">
          <h5>Current self-concept</h5>
          <p class="character-prose">${escapeHtml(formatCharacterProse(selfConcept))}</p>
        </section>
      ` : ""}
      <section class="detail-section">
        <h5>Current growth edges</h5>
        ${growthEdges.length
          ? `<ul class="semantic-list">${growthEdges.map((edge) => `<li>${escapeHtml(formatCharacterProse(edge))}</li>`).join("")}</ul>`
          : '<p class="detail-muted">No current growth edges.</p>'}
      </section>
    </section>
  `);
}

function renderIdentityGrowthPanel(target, {items = [], emptyText = "No growth activity.", redaction = {}} = {}) {
  const element = typeof target === "string" ? qs(target) : target;
  if (!element) return;
  if (!items.length) {
    renderPanelEmptyContent(element, {emptyText, redaction});
    return;
  }
  setHtml(element, items.map((item) => {
    if (item.kind === "identity_candidate") {
      return renderRecordCard(item, {
        title: `${formatLookupLabel(item.change_kind || "identity")} candidate`,
        status: item.status || "",
        hiddenKeys: [
          "kind",
          "status",
          "change_kind",
          "proposed_paths",
          "root_count",
          "local_date_count",
          "updated_at",
        ],
        body: (item.proposed_paths || []).map(formatLookupLabel).join(", "),
        chips: [
          ["base revision", item.base_revision_number],
          ["roots", item.root_count],
          ["local dates", item.local_date_count],
          ["updated", item.updated_at],
        ],
      });
    }
    return renderRecordCard(item, {
      title: `${formatLookupLabel(item.run_kind || "identity")} growth run`,
      status: item.disposition || item.lifecycle_state || "",
      hiddenKeys: [
        "kind",
        "run_kind",
        "disposition",
        "lifecycle_state",
        "latest_reason_code",
        "started_at",
        "completed_at",
      ],
      body: item.latest_reason_code
        ? `Latest reason: ${formatLookupLabel(item.latest_reason_code)}`
        : "",
      chips: [
        ["lifecycle", item.lifecycle_state],
        ["base revision", item.base_revision_number],
        ["roots", item.root_count],
        ["started", item.started_at],
        ["completed", item.completed_at],
      ],
    });
  }).join(""));
}

const IDENTITY_HEALTH_LABELS = {
  healthy_idle: "healthy idle",
  waiting_for_evidence: "waiting for evidence",
  semantic_rejection: "semantic rejection",
  promotion_ready: "promotion ready",
  awaiting_consumption: "awaiting consumption",
  healthy_active: "healthy active",
  pipeline_error: "pipeline error",
  consumption_error: "consumption error",
};

function identityHealthLabel(value) {
  return IDENTITY_HEALTH_LABELS[value] || formatLookupLabel(value || "unknown");
}

function renderIdentityHealth(item) {
  return `
    <article class="identity-health-card">
      <div class="trait-header">
        <div>
          <h4>Growth pipeline health</h4>
          <span class="detail-muted">Latest reason: ${escapeHtml(formatLookupLabel(item.latest_reason_code || "not routed"))}</span>
        </div>
        <span class="${badgeClass(item.state || "")}">${escapeHtml(identityHealthLabel(item.state))}</span>
      </div>
      ${detailChipRow([
        ["latest revision", item.latest_revision_number],
        ["latest consumed", item.latest_consumed_revision_number],
        ["roots", item.root_count],
        ["local dates", item.local_date_count],
      ])}
      ${renderDetailGrid([
        ["routed", item.routed_count],
        ["no change", item.no_change_count],
        ["emerging", item.emerging_candidate_count],
        ["ready", item.ready_candidate_count],
        ["rejected", item.rejected_count],
        ["failed", item.failed_count],
        ["promoted", item.promoted_count],
        ["consumed", item.consumed_count],
      ])}
    </article>
  `;
}

function renderIdentityRevision(item) {
  const diffRows = Array.isArray(item.change_diff) ? item.change_diff : [];
  const currentLabel = item.is_current ? "current" : `revision ${item.revision_number}`;
  const diffContent = diffRows.length
    ? diffRows.map((row) => `
        <li>
          <strong>${escapeHtml(formatLookupLabel(row.path || ""))}</strong>
          <span>${escapeHtml(formatLookupLabel(row.value_kind || "value"))}</span>
        </li>
      `).join("")
    : '<li class="detail-muted">Seed revision; no changed paths.</li>';
  return `
    <details class="identity-revision-card"${item.is_current ? " open" : ""}>
      <summary>
        <span>${escapeHtml(formatLookupLabel(item.revision_kind || "identity revision"))}</span>
        <span class="${item.is_current ? "badge success" : "badge"}">${escapeHtml(currentLabel)}</span>
      </summary>
      <div class="identity-revision-content">
        ${detailChipRow([
          ["base", item.base_revision_number],
          ["roots", item.evidence_root_count],
          ["local dates", item.evidence_local_date_count],
          ["created", item.created_at],
        ])}
        ${item.evidence_summary
          ? `<p class="character-prose">${escapeHtml(formatCharacterProse(item.evidence_summary))}</p>`
          : ""}
        <section class="detail-section">
          <h5>Redacted diff</h5>
          <ul class="identity-diff-list">${diffContent}</ul>
        </section>
        ${renderDetailGrid([
          ["source scopes", item.source_scope_kinds],
          ["proposal confidence", item.proposal_confidence],
          ["review confidence", item.review_confidence],
        ])}
      </div>
    </details>
  `;
}

function renderIdentityLineagePanel(target, {items = [], emptyText = "No identity lineage.", redaction = {}} = {}) {
  const element = typeof target === "string" ? qs(target) : target;
  if (!element) return;
  if (!items.length) {
    renderPanelEmptyContent(element, {emptyText, redaction});
    return;
  }
  const health = items.find((item) => item.kind === "identity_growth_health");
  const revisions = items.filter((item) => item.kind === "identity_revision");
  setHtml(element, `
    ${health ? renderIdentityHealth(health) : '<p class="panel-empty">Identity health is unavailable.</p>'}
    <div class="identity-revision-list">
      ${revisions.length
        ? revisions.map(renderIdentityRevision).join("")
        : '<p class="panel-empty">No identity revisions are available.</p>'}
    </div>
  `);
}

function renderMemoryUnitRows(target, {items = [], emptyText = "No memory rows available.", redaction = {}} = {}) {
  const element = typeof target === "string" ? qs(target) : target;
  if (!element) return;
  if (!items.length) {
    if (isTableBody(element)) {
      renderLookupTable(element, {items, emptyText, redaction});
    } else {
      renderPanelEmptyContent(element, {emptyText, redaction});
    }
    return;
  }
  if (!isTableBody(element)) {
    setHtml(element, items.map((item) => {
      const typeText = formatLookupLabel(item.unit_type || "memory");
      const bodyKey = item.fact
        ? "fact"
        : item.subjective_appraisal
          ? "subjective_appraisal"
          : "relationship_signal";
      const factText = formatLookupValue(item[bodyKey]);
      const chips = [
        ["status", item.status],
        ["updated", item.updated_at],
        ["last seen", item.last_seen_at && !item.updated_at ? item.last_seen_at : ""],
        ["relationship", item.relationship_signal ? "present" : ""],
        ["appraisal", item.subjective_appraisal ? "present" : ""],
        ["due", item.due_at],
      ];
      const details = {
        relationship_signal: bodyKey === "relationship_signal" ? "" : item.relationship_signal,
        subjective_appraisal: bodyKey === "subjective_appraisal" ? "" : item.subjective_appraisal,
      };
      return renderRecordCard(details, {
        title: typeText || "memory",
        status: item.status || "",
        hiddenKeys: [],
        body: factText,
        chips,
      });
    }).join(""));
    return;
  }
  setHtml(element, items.map((item) => {
    const typeText = formatLookupLabel(item.unit_type || "memory");
    const statusText = item.status ? formatLookupLabel(item.status) : "";
    const bodyKey = item.fact
      ? "fact"
      : item.subjective_appraisal
        ? "subjective_appraisal"
        : "relationship_signal";
    const factText = formatLookupValue(item[bodyKey]);
    const primaryMeta = memoryMeta([
      statusText,
      item.updated_at ? `updated: ${formatLookupValue(item.updated_at)}` : "",
      item.last_seen_at && !item.updated_at ? `last seen: ${formatLookupValue(item.last_seen_at)}` : "",
    ]);
    const detailMeta = memoryMeta([
      item.relationship_signal && bodyKey !== "relationship_signal"
        ? `relationship: ${formatLookupValue(item.relationship_signal)}`
        : "",
      item.subjective_appraisal && bodyKey !== "subjective_appraisal"
        ? `appraisal: ${formatLookupValue(item.subjective_appraisal)}`
        : "",
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

function renderPanelCards(target, panel, cards, emptyText) {
  const items = panelItems(panel);
  if (!items.length) {
    renderPanelState(target, {
      status: panel?.status || "empty",
      reason: panelEmptyText(panel, emptyText),
    });
    return;
  }
  setHtml(target, cards(items).join(""));
}

function cognitionValueMarkup(value) {
  if (Array.isArray(value)) {
    if (!value.length) return "";
    return value.map((item) => {
      if (item && typeof item === "object") {
        return `<article class="record-card">${renderDetailGrid(Object.entries(item))}</article>`;
      }
      return `<p class="character-prose">${escapeHtml(formatLookupValue(item))}</p>`;
    }).join("");
  }
  if (value && typeof value === "object") {
    return renderDetailGrid(Object.entries(value));
  }
  return `<p class="character-prose">${escapeHtml(formatLookupValue(value))}</p>`;
}

function renderCognitionStatePanel(target, panel, emptyText) {
  const items = panelItems(panel);
  if (!items.length) {
    renderPanelState(target, {status: panel?.status || "empty", reason: panelEmptyText(panel, emptyText)});
    return;
  }
  const emptyItems = items.filter((item) => cognitionValueIsEmpty(item.value));
  const populatedItems = items.filter((item) => !cognitionValueIsEmpty(item.value));
  const populatedMarkup = populatedItems.map((item) => `
    <section class="detail-section">
      <h5>${escapeHtml(formatLookupLabel(item.key))}</h5>
      ${cognitionValueMarkup(item.value)}
    </section>
  `).join("");
  const emptyMarkup = emptyItems.length ? `
    <section class="detail-section">
      <h5>Currently empty</h5>
      <p class="panel-empty">${escapeHtml(emptyItems.map((item) => formatLookupLabel(item.key)).join(", "))}</p>
    </section>
  ` : "";
  setHtml(target, populatedMarkup + emptyMarkup);
}

function renderCharacterOperationalPosturePanel(target, panel) {
  const items = panelItems(panel);
  if (!items.length) {
    renderPanelState(target, {
      status: panel?.status || "empty",
      reason: panelEmptyText(panel, "No native character operational posture is available."),
    });
    return;
  }
  const posture = items[0] || {};
  const latest = posture.latest_context || {};
  const context = latest.context || {};
  const health = context.health || {};
  const sections = [
    renderOperationalPostureView("Persisted posture", posture.persisted || {}, false),
    renderOperationalPostureView("Elapsed-effective posture", posture.effective || {}, Boolean(posture.fading_changed)),
    renderOperationalConsumption(latest, context),
    renderOperationalHealth(health),
  ].filter(Boolean);
  setHtml(target, sections.join(""));
}

function renderOperationalPostureView(title, view, fadingChanged) {
  if (!view || typeof view !== "object" || !Object.keys(view).length) return "";
  const affect = Array.isArray(view.affect) ? view.affect : [];
  const pressures = Array.isArray(view.pressures) ? view.pressures : [];
  return `
    <section class="operational-posture-section">
      <div class="trait-header">
        <div><h4>${escapeHtml(title)}</h4><span class="detail-muted">source ${escapeHtml(formatLookupValue(view.source_updated_at))} · effective ${escapeHtml(formatLookupValue(view.effective_at))}</span></div>
        <span class="${badgeClass(fadingChanged ? "partial" : "available")}">${escapeHtml(fadingChanged ? "ordinary fading changed" : "unchanged")}</span>
      </div>
      ${detailChipRow([["source digest", view.source_digest], ["view digest", view.view_digest]])}
      ${renderOperationalRows("Native affect", affect, "No active or fading affect rows.")}
      ${renderOperationalRows("Pressure", pressures, "No bounded pressure rows.")}
    </section>
  `;
}

function renderOperationalRows(title, rows, emptyText) {
  if (!rows.length) {
    return `<section class="detail-section"><h5>${escapeHtml(title)}</h5><p class="detail-muted">${escapeHtml(emptyText)}</p></section>`;
  }
  const fields = [...new Set(rows.flatMap((row) => Object.keys(row || {})))];
  return `
    <section class="detail-section">
      <h5>${escapeHtml(title)}</h5>
      <div class="table-wrap"><table><thead><tr>${fields.map((field) => `<th>${escapeHtml(formatLookupLabel(field))}</th>`).join("")}</tr></thead><tbody>
        ${rows.map((row) => `<tr>${fields.map((field) => `<td>${escapeHtml(formatLookupValue(row?.[field]))}</td>`).join("")}</tr>`).join("")}
      </tbody></table></div>
    </section>
  `;
}

function renderOperationalConsumption(latest, context) {
  const status = latest.status || context.status || "not_reported";
  const stages = [
    ["Settled relevance", context.settled_relevance],
    ["Cognition", context.cognition],
    ["Surface", context.surface],
  ].filter(([, stage]) => stage && typeof stage === "object" && Object.keys(stage).length);
  return `
    <section class="operational-posture-section">
      <div class="trait-header"><div><h4>Latest consumed context</h4><span class="detail-muted">Source-owned graph projection; no console reconstruction.</span></div><span class="${badgeClass(status)}">${escapeHtml(formatLookupLabel(status))}</span></div>
      ${detailChipRow([["run", latest.run_id], ["generated", latest.generated_at], ["reason", latest.reason_code]])}
      ${stages.length ? stages.map(([label, stage]) => `
        <section class="detail-section"><h5>${escapeHtml(label)}</h5>${renderDetailGrid(Object.entries(stage))}</section>
      `).join("") : `<p class="detail-muted">${escapeHtml(latest.reason_code ? formatLookupLabel(latest.reason_code) : "The latest graph has not reported consumed context.")}</p>`}
    </section>
  `;
}

function renderOperationalHealth(health) {
  if (!health || typeof health !== "object" || !Object.keys(health).length) return "";
  return `
    <section class="operational-posture-section">
      <h4>Predecessor and stage health</h4>
      ${renderDetailGrid(Object.entries(health))}
    </section>
  `;
}

function cognitionValueIsEmpty(value) {
  if (Array.isArray(value)) return value.length === 0;
  if (value && typeof value === "object") return Object.keys(value).length === 0;
  return value === null || value === undefined || value === "";
}

function renderContinuityPanel(target, panel, emptyText) {
  renderPanelCards(target, panel, (items) => items.map((item, index) => {
    const titleKey = ["title", "source"].find((key) => item[key]) || "";
    const bodyKey = ["summary", "content", "claim", "context"].find(
      (key) => item[key],
    ) || "";
    return renderRecordCard(item, {
      title: titleKey ? item[titleKey] : `Continuity ${index + 1}`,
      status: item.status || "",
      hiddenKeys: ["status", titleKey, bodyKey].filter(Boolean),
      body: bodyKey ? item[bodyKey] : "",
    });
  }), emptyText);
}

function renderUserProfilePanel(panel) {
  renderPanelCards("#user-profile-table", panel, (items) => items.map((item) => {
    const accounts = Array.isArray(item.accounts) ? item.accounts : [];
    return `
      <article class="record-card">
        <div class="record-card-header"><h4>${escapeHtml(formatLookupValue(item.display_name || "User accounts"))}</h4></div>
        ${detailChipRow([
          ["accounts", item.account_count],
          ["aliases", item.alias_count],
          ["updated", item.updated_at],
          ["global_user_id", item.global_user_id],
        ])}
        ${accounts.map((account) => `
          <div class="detail-kv">
            <span class="detail-label">${escapeHtml(formatLookupValue(account.platform))}</span>
            <span class="detail-value">${escapeHtml(
              account.display_name && account.display_name !== account.platform_user_id
                ? `${formatLookupValue(account.display_name)} · ${formatLookupValue(account.platform_user_id)}`
                : formatLookupValue(account.platform_user_id),
            )}</span>
          </div>
        `).join("")}
      </article>
    `;
  }), "No user profile fields are available.");
}

function renderRelationshipPanel(panel) {
  const items = panelItems(panel);
  if (!items.length) {
    setHtml("#user-relationship-table", `<tr><td colspan="3">${escapeHtml(panelEmptyText(panel, "No native V2 relationship state is available."))}</td></tr>`);
    return;
  }
  const axisRows = items.map((item) => `
    <tr>
      <td>${escapeHtml(formatLookupLabel(item.axis))}</td>
      <td>${escapeHtml(formatLookupValue(item.value))}</td>
      <td>${escapeHtml(formatLookupValue(item.band))}</td>
    </tr>
  `).join("");
  const meta = `Evidence ${formatLookupValue(panel.evidence_count)} · updated ${formatLookupValue(panel.updated_at)}`;
  setHtml("#user-relationship-table", `${axisRows}<tr><td colspan="3"><span class="table-meta">${escapeHtml(meta)}</span></td></tr>`);
}

function renderRelationshipOperationalPanel(panel) {
  const items = panelItems(panel);
  if (!items.length) {
    renderPanelState("#user-relationship-operational-table", {
      status: panel?.status || "empty",
      reason: panelEmptyText(panel, "No causal relationship context is available."),
    });
    return;
  }
  const context = items[0] || {};
  const axes = context.axes && typeof context.axes === "object"
    ? Object.entries(context.axes)
    : [];
  const causalRows = Array.isArray(context.causal_context) ? context.causal_context : [];
  const affectRows = Array.isArray(context.affect) ? context.affect : [];
  setHtml("#user-relationship-operational-table", `
    <section class="operational-posture-section">
      <div class="detail-section"><h5>Relationship axes</h5>${axes.length ? renderDetailGrid(axes) : '<p class="detail-muted">No projected axes.</p>'}</div>
      ${renderOperationalRows("Causal rows", causalRows, "No causal rows are active.")}
      ${renderOperationalRows("Relationship affect", affectRows, "No relationship-rooted affect rows are active.")}
      ${detailChipRow([["relationship freshness", context.relationship_freshness], ["evidence freshness", context.evidence_freshness]])}
    </section>
  `);
}

function renderStylePanel(target, panel, scopeLabel) {
  renderPanelCards(target, panel, (items) => items.map((item, index) => {
    if (item.consumer_role) {
      const guidance = item.guidance && typeof item.guidance === "object"
        ? Object.entries(item.guidance)
        : [];
      return `
        <article class="record-card">
          <div class="record-card-header"><h4>${escapeHtml(`${formatLookupLabel(item.consumer_role)} · ${formatLookupLabel(item.source || scopeLabel)}`)}</h4><span class="${badgeClass(item.status || "")}">${escapeHtml(formatLookupLabel(item.status || "not reported"))}</span></div>
          ${detailChipRow([["revision", item.revision], ["confidence", item.confidence]])}
          ${guidance.length ? guidance.map(([field, values]) => `<section class="detail-section"><h5>${escapeHtml(formatLookupLabel(field))}</h5><p class="character-prose">${escapeHtml(formatLookupValue(values))}</p></section>`).join("") : '<p class="detail-muted">No declared guidance for this projection.</p>'}
        </article>
      `;
    }
    return renderRecordCard(item, {
      title: item.field || item.scope || `${scopeLabel} ${index + 1}`,
      hiddenKeys: ["field", "scope", "guidelines", "confidence"],
      body: item.guidelines || "",
      chips: [["scope", item.scope], ["confidence", item.confidence]],
    });
  }), `No ${scopeLabel} guidance is available.`);
}

function renderUserDirectory(payload) {
  const items = Array.isArray(payload.items) ? payload.items : [];
  state.userDirectory = items;
  setEntityStatus("#user-directory-status", payload.status || "empty");
  const rows = items.flatMap((item) => {
    const accounts = Array.isArray(item.accounts) ? item.accounts : [];
    return accounts.map((account) => `
      <tr>
        <td>
          <span class="table-primary">${escapeHtml(formatLookupValue(`${account.platform}:${account.platform_user_id}`))}</span>
          ${item.global_user_id ? `<span class="table-meta">${escapeHtml(formatLookupValue(`global_user_id: ${item.global_user_id}`))}</span>` : ""}
        </td>
        <td>${escapeHtml(formatLookupValue(account.display_name || item.display_name))}</td>
        <td>${escapeHtml(formatLookupValue(item.alias_count))}</td>
        <td>${escapeHtml(formatLookupValue(item.updated_at))}</td>
        <td><button class="btn" type="button" data-user-platform="${escapeHtml(account.platform)}" data-user-id="${escapeHtml(account.platform_user_id)}">Inspect</button></td>
      </tr>
    `);
  });
  setHtml("#user-directory-table", rows.length
    ? rows.join("")
    : `<tr><td colspan="5">${escapeHtml(payload.reason || "No recent user profiles are available.")}</td></tr>`);
}

function renderGroupDirectory(payload) {
  const items = Array.isArray(payload.items) ? payload.items : [];
  state.groupDirectory = items;
  setEntityStatus("#group-directory-status", payload.status || "empty");
  setHtml("#group-directory-table", items.length
    ? items.map((item) => `
      <tr>
        <td>${escapeHtml(formatLookupValue(item.channel_name || item.group_id))}</td>
        <td>${escapeHtml(formatLookupValue(item.platform))}</td>
        <td>${escapeHtml(formatLookupValue(item.message_count))}</td>
        <td>${escapeHtml(formatLookupValue(item.participant_count))}</td>
        <td>${escapeHtml(formatLookupValue(item.last_activity_at))}</td>
        <td><button class="btn" type="button" data-group-platform="${escapeHtml(item.platform)}" data-group-id="${escapeHtml(item.group_id)}">Inspect</button></td>
      </tr>
    `).join("")
    : `<tr><td colspan="6">${escapeHtml(payload.reason || "No recent group activity is available.")}</td></tr>`);
}

function renderGroupActivityPanel(panel) {
  renderPanelCards("#group-activity-table", panel, (items) => items.map((item) => renderRecordCard(item, {
    title: item.channel_name || item.group_id || "Group activity",
    hiddenKeys: ["channel_name", "group_id", "platform", "last_activity_at", "message_count", "participant_count"],
    chips: [
      ["platform", item.platform],
      ["messages", item.message_count],
      ["participants", item.participant_count],
      ["last activity", item.last_activity_at],
    ],
  })), "No activity matched the selected group.");
}

function renderGroupReviewPanel(panel) {
  renderPanelCards("#group-review-table", panel, (items) => items.map((item) => {
    const readableItem = {
      ...item,
      skip_reason: item.skip_reason ? formatLookupLabel(item.skip_reason) : "",
    };
    return renderRecordCard(readableItem, {
      title: "Latest review",
      status: item.status || "",
      hiddenKeys: ["status", "window_start", "window_end", "reviewed_at"],
      chips: [
        ["from", item.window_start],
        ["to", item.window_end],
        ["reviewed", item.reviewed_at],
      ],
    });
  }), "No terminal review windows matched this group.");
}

function renderCalendarSummary(panel) {
  const summary = firstObjectItem(panelItems(panel));
  setEntityStatus("#calendar-summary-status", panel?.status || "unavailable");
  if (!Object.keys(summary).length) {
    setHtml("#calendar-summary-table", `<tr><td>Status</td><td>${escapeHtml(panelEmptyText(panel, "Calendar counts are unavailable."))}</td></tr>`);
    return;
  }
  const rows = [
    ["Active schedules", summary.active_schedules],
    ["Upcoming", summary.upcoming],
    ["Overdue", summary.overdue],
    ...[
      ["Recent running", summary.running],
      ["Recent completed", summary.completed],
      ["Recent failed", summary.failed],
      ["Recent skipped", summary.skipped],
    ].filter(([, value]) => Number(value) > 0),
  ];
  setHtml("#calendar-summary-table", rows.map(([label, value]) => `<tr><td>${escapeHtml(label)}</td><td>${escapeHtml(formatLookupValue(value))}</td></tr>`).join(""));
}

function renderCalendarSchedules(panel) {
  setEntityStatus("#calendar-schedules-status", panel?.status || "empty");
  renderPanelCards("#calendar-schedules-table", panel, (items) => items.map((item) => renderRecordCard(item, {
    title: item.trigger_kind ? formatLookupLabel(item.trigger_kind) : "Schedule",
    status: item.status || "",
    chips: [
      ["starts", item.start_at],
      ["next run", item.next_run_at],
      ["updated", item.updated_at],
    ],
    hiddenKeys: [
      "calendar_schedule_id",
      "trigger_kind",
      "status",
      "start_at",
      "next_run_at",
      "updated_at",
    ],
    referenceLabel: "Schedule reference",
    references: [["Schedule reference", item.calendar_schedule_id]],
  })), "No schedule definitions are available.");
}

function renderCalendarRuns(panel) {
  setEntityStatus("#calendar-runs-status", panel?.status || "empty");
  renderPanelCards("#calendar-runs-table", panel, (items) => items.map((item) => renderRecordCard(item, {
    title: item.trigger_kind ? formatLookupLabel(item.trigger_kind) : "Calendar run",
    status: item.status || "",
    hiddenKeys: [
      "calendar_run_id",
      "trigger_kind",
      "status",
      "due_at",
      "completed_at",
      "failed_at",
      "skipped_at",
      "updated_at",
    ],
    chips: [
      ["due", item.due_at],
      ["completed", item.completed_at],
      ["failed", item.failed_at],
      ["skipped", item.skipped_at],
      [
        "updated",
        [item.completed_at, item.failed_at, item.skipped_at].includes(
          item.updated_at,
        ) ? "" : item.updated_at,
      ],
    ],
    referenceLabel: "Run reference",
    references: [["Run reference", item.calendar_run_id]],
  })), "No recent calendar runs are available.");
}

function renderBackgroundSummary(panel) {
  const summary = firstObjectItem(panelItems(panel));
  setEntityStatus("#background-summary-status", panel?.status || "unavailable");
  if (!Object.keys(summary).length) {
    setHtml("#background-summary-table", `<tr><td>Status</td><td>${escapeHtml(panelEmptyText(panel, "Background counts are unavailable."))}</td></tr>`);
    return;
  }
  const rows = [
    ["Queued", summary.queued],
    ["Running", summary.running],
    ...[
      ["Completed", summary.completed],
      ["Failed", summary.failed],
      ["Delivery ready", summary.delivery_ready],
      ["Deferred", summary.deferred],
    ].filter(([, value]) => Number(value) > 0),
  ];
  setHtml("#background-summary-table", rows.map(([label, value]) => `<tr><td>${escapeHtml(label)}</td><td>${escapeHtml(formatLookupValue(value))}</td></tr>`).join(""));
}

function renderBackgroundJobs(target, panel, emptyText, showJobReference = false) {
  renderPanelCards(target, panel, (items) => items.map((item, index) => renderRecordCard(item, {
    title: item.requester_display_name || item.worker || `Job ${index + 1}`,
    status: item.status || item.delivery_state || "",
    hiddenKeys: [
      "requester_display_name",
      "worker",
      "status",
      "delivery_state",
      "created_at",
      "updated_at",
      "completed_at",
      ...(showJobReference ? ["background_work_job_id", "job_id"] : []),
    ],
    chips: [
      ["worker", item.worker],
      ["delivery", item.delivery_state === item.status ? "" : item.delivery_state],
      ...(target === "#background-jobs-table" ? [] : [
        ["created", item.created_at],
        ["completed", item.completed_at],
        ["updated", item.updated_at === item.completed_at ? "" : item.updated_at],
      ]),
    ],
    references: showJobReference
      ? [["Job reference", item.background_work_job_id]]
      : [],
  })), emptyText);
}

function renderBackgroundWorkers(panel) {
  const items = panelItems(panel);
  setEntityStatus("#background-worker-status", panel?.status || "empty");
  setHtml("#background-worker-table", items.length
    ? items.map((item) => `
      <tr>
        <td>${escapeHtml(formatOperationalLabel(item.worker_name))}</td>
        <td>${escapeHtml(formatLookupValue(item.event_count))}</td>
        <td>${escapeHtml(formatLookupValue(item.processed_count))}</td>
        <td>${escapeHtml(formatLookupValue(item.succeeded_count))}</td>
        <td>${escapeHtml(formatLookupValue(item.failed_count))}</td>
        <td>${escapeHtml(formatLookupValue(item.skipped_count))}</td>
        <td>${escapeHtml(formatLookupValue(item.deferred_count))}</td>
        <td>${escapeHtml(formatOperationalLabel(item.last_status))}</td>
      </tr>
    `).join("")
    : `<tr><td colspan="8">${escapeHtml(panelEmptyText(panel, "No background worker activity is available."))}</td></tr>`);
}

const CHARACTER_PANEL_TARGETS = [
  "#character-profile-table",
  "#character-cognition-state-table",
  "#character-operational-posture-table",
  "#character-self-image-table",
  "#character-growth-table",
  "#character-carry-over-table",
];

function renderCharacterLoadingState() {
  setEntityStatus("#character-status", "loading");
  CHARACTER_PANEL_TARGETS.forEach((target) => {
    setHtml(target, '<p class="panel-empty">Loading character identity...</p>');
  });
}

function renderCharacterErrorState(error) {
  const reason = error instanceof Error && error.message
    ? error.message
    : "request failed";
  setEntityStatus("#character-status", "unavailable");
  CHARACTER_PANEL_TARGETS.forEach((target) => {
    setHtml(
      target,
      `<p class="panel-empty">Character identity could not be loaded. ${escapeHtml(reason)}</p>`,
    );
  });
}

async function refreshCharacter() {
  renderCharacterLoadingState();
  let payload;
  try {
    payload = await api("/api/entities/character?limit=25");
  } catch (error) {
    renderCharacterErrorState(error);
    throw error;
  }
  setEntityStatus("#character-status", payload.status || "unavailable");
  const panels = payload.panels || {};
  renderCharacterProfilePanel("#character-profile-table", {
    items: panelItems(panels.profile),
    emptyText: panelEmptyText(panels.profile, "No character profile rows."),
    redaction: payload.redaction || {},
  });
  renderCognitionStatePanel("#character-cognition-state-table", panels.cognition_state, "No character cognition state is available.");
  renderCharacterOperationalPosturePanel(
    "#character-operational-posture-table",
    panels.operational_posture,
  );
  renderCharacterSelfImagePanel("#character-self-image-table", {
    items: panelItems(panels.self_image),
    emptyText: panelEmptyText(panels.self_image, "No self-image rows."),
    redaction: payload.redaction || {},
  });
  renderIdentityGrowthPanel("#character-growth-table", {
    items: panelItems(panels.growth),
    emptyText: panelEmptyText(panels.growth, "No identity growth activity."),
    redaction: payload.redaction || {},
  });
  renderIdentityLineagePanel("#character-carry-over-table", {
    items: panelItems(panels.carry_over),
    emptyText: panelEmptyText(panels.carry_over, "No identity lineage is available."),
    redaction: payload.redaction || {},
  });
}

async function refreshUsers(showNeedsInput = true) {
  const directory = await api("/api/entities/users?limit=25");
  renderUserDirectory(directory);
  const platform = getValue("#user-platform").trim();
  const platformUserId = getValue("#user-platform-user-id").trim();
  if (!platform || !platformUserId) {
    setEntityStatus("#users-status", directory.status || "needs_input");
    if (showNeedsInput) renderUserNeedsInput();
    return;
  }

  const params = new URLSearchParams({limit: "25"});
  const platformChannelId = getValue("#user-platform-channel-id").trim();
  const channelType = getValue("#user-channel-type").trim();
  const query = getValue("#user-query").trim();
  if (platformChannelId) params.set("platform_channel_id", platformChannelId);
  if (channelType) params.set("channel_type", channelType);
  if (query) params.set("query", query);
  const payload = await api(`/api/entities/users/${encodeURIComponent(platform)}/${encodeURIComponent(platformUserId)}?${params.toString()}`);
  setEntityStatus("#users-status", payload.status || "unavailable");
  const panels = payload.panels || {};
  renderUserProfilePanel(panels.profile);
  renderRelationshipPanel(panels.relationship);
  renderRelationshipOperationalPanel(panels.relationship_operational);
  renderCognitionStatePanel("#user-cognition-state-table", panels.cognition_state, "No user cognition state is available.");
  renderMemoryUnitRows("#user-memory-table", {
    items: panelItems(panels.memory),
    emptyText: panelEmptyText(panels.memory, "No user memory rows."),
    redaction: payload.redaction || {},
  });
  renderStylePanel("#user-style-table", panels.style, "user style");
  renderContinuityPanel("#user-conversation-progress-table", panels.conversation_progress, "No conversation progress is available for this scope.");
  renderContinuityPanel("#user-carry-over-table", panels.carry_over, "No carry-over is available for this scope.");
}

function renderUserNeedsInput() {
  const panel = {status: "needs_input", reason: "Select a known user or enter platform and account ID."};
  renderPanelState("#user-profile-table", panel);
  setHtml("#user-relationship-table", '<tr><td colspan="3">Select a known user or enter platform and account ID.</td></tr>');
  renderPanelState("#user-relationship-operational-table", panel);
  renderPanelState("#user-cognition-state-table", panel);
  renderPanelState("#user-memory-table", panel);
  renderPanelState("#user-style-table", panel);
  renderPanelState("#user-conversation-progress-table", panel);
  renderPanelState("#user-carry-over-table", panel);
}

function selectUserDirectoryRow(event) {
  const button = event.target.closest("[data-user-platform][data-user-id]");
  if (!button) return;
  setValue("#user-platform", button.dataset.userPlatform || "");
  setValue("#user-platform-user-id", button.dataset.userId || "");
  refreshUsers(true).catch(reportActionError);
}

async function refreshGroups(showNeedsInput = true) {
  const directory = await api("/api/entities/groups?limit=25");
  renderGroupDirectory(directory);
  const platform = getValue("#group-platform").trim();
  const groupId = getValue("#group-id").trim();
  if (!platform || !groupId) {
    setEntityStatus("#groups-status", directory.status || "needs_input");
    if (showNeedsInput) renderGroupNeedsInput();
    return;
  }

  const params = new URLSearchParams({limit: "25"});
  const participantPlatformUserId = getValue("#group-participant-platform-user-id").trim();
  if (participantPlatformUserId) params.set("participant_platform_user_id", participantPlatformUserId);
  const payload = await api(`/api/entities/groups/${encodeURIComponent(platform)}/${encodeURIComponent(groupId)}?${params.toString()}`);
  setEntityStatus("#groups-status", payload.status || "unavailable");
  const panels = payload.panels || {};
  renderGroupActivityPanel(panels.activity);
  renderGroupReviewPanel(panels.review);
  renderStylePanel("#group-style-table", panels.style, "group style");
  renderContinuityPanel("#group-carry-over-table", panels.carry_over, "No group-scene carry-over is available.");
  renderContinuityPanel("#group-participant-progress-table", panels.participant_progress, "No participant progress is available for this scope.");
}

function renderGroupNeedsInput() {
  const panel = {status: "needs_input", reason: "Select an active group or enter platform and group ID."};
  renderPanelState("#group-activity-table", panel);
  renderPanelState("#group-review-table", panel);
  renderPanelState("#group-style-table", panel);
  renderPanelState("#group-carry-over-table", panel);
  renderPanelState("#group-participant-progress-table", panel);
}

function selectGroupDirectoryRow(event) {
  const button = event.target.closest("[data-group-platform][data-group-id]");
  if (!button) return;
  setValue("#group-platform", button.dataset.groupPlatform || "");
  setValue("#group-id", button.dataset.groupId || "");
  refreshGroups(true).catch(reportActionError);
}

async function refreshCalendar() {
  const params = new URLSearchParams({limit: "6"});
  const filters = [
    ["platform", "#calendar-platform"],
    ["platform_channel_id", "#calendar-platform-channel-id"],
    ["platform_user_id", "#calendar-platform-user-id"],
    ["channel_type", "#calendar-channel-type"],
  ];
  filters.forEach(([key, selector]) => {
    const value = getValue(selector).trim();
    if (value) params.set(key, value);
  });
  const payload = await api(`/api/lookups/calendar?${params.toString()}`);
  setEntityStatus("#calendar-status", payload.status || "unavailable");
  const panels = payload.panels || {};
  renderCalendarSummary(panels.summary);
  renderCalendarSchedules(panels.schedules);
  renderCalendarRuns(panels.runs);
  setEntityStatus("#calendar-cognition-status", panels.cognition_visibility?.status || "needs_input");
  renderContinuityPanel("#calendar-cognition-visibility-table", panels.cognition_visibility, "No pending calendar candidates are visible to this scope.");
}

async function refreshBackground() {
  const payload = await api("/api/lookups/background-work?limit=25");
  setEntityStatus("#background-status", payload.status || "unavailable");
  const panels = payload.panels || {};
  renderBackgroundSummary(panels.summary);
  setEntityStatus("#background-jobs-status", panels.jobs?.status || "empty");
  renderBackgroundJobs("#background-jobs-table", panels.jobs, "No background-work jobs are available.", true);
  renderBackgroundWorkers(panels.worker_activity);
  setEntityStatus("#background-errors-status", panels.errors?.status || "empty");
  renderBackgroundJobs("#background-errors-table", panels.errors, "No recent background worker errors.", true);
  setEntityStatus("#background-delivery-status", panels.delivery_detail?.status || "empty");
  renderBackgroundJobs("#background-delivery-table", panels.delivery_detail, "No jobs are ready for delivery.");
}

function facetRows(facet, emptyText, labelFormatter = formatLookupLabel) {
  const entries = Object.entries(facet || {}).sort((left, right) => right[1] - left[1]);
  if (!entries.length) return `<tr><td>Status</td><td>${escapeHtml(emptyText)}</td></tr>`;
  return entries.map(([label, count]) => `<tr><td>${escapeHtml(labelFormatter(label))}</td><td>${escapeHtml(formatLookupValue(count))}</td></tr>`).join("");
}

function eventDetailMarkup(event) {
  const details = [
    ["source", event.source],
    ["request_id", event.request_id],
    ["correlation_id", event.correlation_id],
    ["tracking_id", event.tracking_id],
    ["run_id", event.run_id],
    ["trigger_id", event.trigger_id],
    ["attempt_id", event.attempt_id],
    ["processed", event.processed_count],
    ["succeeded", event.succeeded_count],
    ["failed", event.failed_count],
    ["skipped", event.skipped_count],
    ["deferred", event.deferred],
    ["defer reason", event.defer_reason ? formatOperationalLabel(event.defer_reason) : ""],
    ["run kind", event.run_kind ? formatOperationalLabel(event.run_kind) : ""],
    ["worker", event.worker_name ? formatOperationalLabel(event.worker_name) : ""],
  ].filter(([, value]) => value !== null && value !== undefined && value !== "");
  const eventLabel = formatOperationalLabel(event.event_type);
  if (!details.length) return escapeHtml(eventLabel);
  return `<details><summary>${escapeHtml(eventLabel)}</summary>${renderDetailGrid(details)}</details>`;
}

async function refreshEvents() {
  const source = getValue("#event-source", "all") || "all";
  const params = new URLSearchParams({source, limit: "25"});
  const filters = [
    ["service_id", "#event-service-id"],
    ["event_type", "#event-type"],
    ["level", "#event-level"],
    ["request_id", "#event-request-id"],
    ["tracking_id", "#event-tracking-id"],
  ];
  filters.forEach(([key, selector]) => {
    const value = getValue(selector).trim();
    if (value) params.set(key, value);
  });
  const since = getValue("#event-since").trim();
  if (since) params.set("since", new Date(since).toISOString());
  const payload = await api(`/api/events?${params.toString()}`);
  const events = Array.isArray(payload.items) ? payload.items : [];
  setText("#event-result-count", `${events.length} events`);
  setHtml("#event-severity-facets", facetRows(payload.facets?.levels, "No severity counts matched."));
  setHtml("#event-status-facets", facetRows(payload.facets?.statuses, "No outcome counts matched."));
  setHtml("#event-component-facets", facetRows(
    payload.facets?.components,
    "No component counts matched.",
    formatOperationalLabel,
  ));
  if (!events.length) {
    setHtml("#event-table", '<tr><td colspan="7">No structured events matched the selected filters.</td></tr>');
    return;
  }
  setHtml("#event-table", events.map((event) => {
    const errorText = [event.error_class, event.message].filter(Boolean).join(": ");
    return `
      <tr>
        <td>${escapeHtml(formatLookupValue(event.created_at))}</td>
        <td>${escapeHtml(formatOperationalLabel(event.level))}</td>
        <td>${escapeHtml(formatOperationalLabel(event.component || event.service_id))}</td>
        <td>${eventDetailMarkup(event)}</td>
        <td>${escapeHtml(formatOperationalLabel(event.status))}</td>
        <td>${escapeHtml(event.duration_ms === undefined ? "-" : `${event.duration_ms} ms`)}</td>
        <td>${escapeHtml(formatLookupValue(errorText))}</td>
      </tr>
    `;
  }).join(""));
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
bind("#user-directory-table", "click", selectUserDirectoryRow);
bind("#group-directory-table", "click", selectGroupDirectoryRow);
bind("#refresh-events", "click", () => runButtonAction(
  optionalElement("#refresh-events"),
  "Loading events...",
  "Events updated.",
  refreshEvents,
));
bind("#refresh-audit", "click", () => runButtonAction(
  optionalElement("#refresh-audit"),
  "Loading audit actions...",
  "Audit updated.",
  refreshAudit,
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
  renderOverviewSelfCognitionGraph(state.latestSelfCognitionGraph);
  renderDebugCognitionGraph(state.debugCognitionGraph);
});
resumeSession();
