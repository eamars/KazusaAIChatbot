"""Documentation harmonization contract tests."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ROUTE_KEYS = (
    "RELEVANCE_AGENT_LLM",
    "VISION_DESCRIPTOR_LLM",
    "MSG_DECONTEXTUALIZER_LLM",
    "RAG_PLANNER_LLM",
    "RAG_SUBAGENT_LLM",
    "WEB_SEARCH_LLM",
    "COGNITION_LLM",
    "COGNITION_LLM_CHARACTER_CARRYOVER",
    "COGNITION_LLM_APPRAISAL_EVENT_AGENCY",
    "COGNITION_LLM_APPRAISAL_RELATIONSHIP_SOCIAL",
    "COGNITION_LLM_APPRAISAL_MORAL_IDENTITY",
    "COGNITION_LLM_APPRAISAL_GOAL_THREAT_OUTCOME",
    "COGNITION_LLM_APPRAISAL_EPISTEMIC_COMPARISON_MEMORY",
    "COGNITION_LLM_APPRAISAL_EXISTENTIAL_DRIVE",
    "COGNITION_LLM_GOAL_ORDINARY_RESPONSE",
    "COGNITION_LLM_GOAL_ACTIVE_BRANCH",
    "COGNITION_LLM_WORKSPACE_COLLAPSE",
    "COGNITION_LLM_ACTION_PLANNING",
    "COGNITION_LLM_ACTION_AUTHORIZATION",
    "COGNITION_LLM_RESOLVER_AUTHORIZATION",
    "BACKGROUND_WORK_LLM",
    "DIALOG_GENERATOR_LLM",
    "CONSOLIDATION_LLM",
    "JSON_REPAIR_LLM",
    "EMBEDDING",
)


def _read_doc(relative_path: str) -> str:
    """Read one repository documentation file as UTF-8 text."""

    path = ROOT / relative_path
    text = path.read_text(encoding="utf-8")
    return text


def _assert_in_order(content: str, phrases: tuple[str, ...]) -> None:
    """Assert that each phrase appears after the previous phrase."""

    previous_index = -1
    for phrase in phrases:
        current_index = content.find(phrase)
        assert current_index >= 0, f"missing phrase: {phrase}"
        assert current_index > previous_index, f"out of order phrase: {phrase}"
        previous_index = current_index


def test_documentation_guide_defines_living_doc_contract() -> None:
    """The documentation guide should define document roles and update rules."""

    content = _read_doc("docs/DOCUMENTATION_GUIDE.md")

    required_phrases = (
        "Source-Of-Truth Hierarchy",
        "Module README Section Contract",
        "Bilingual Parity",
        "Historical Plan Policy",
        "Document Control",
        "Forbidden Paths",
    )
    for phrase in required_phrases:
        assert phrase in content


def test_subagent_interface_guide_covers_known_families() -> None:
    """The subagent guide should cover each current family without abstraction."""

    content = _read_doc("docs/SUBAGENT_INTERFACES.md")

    required_phrases = (
        "BaseRAGHelperAgent",
        "web_agent3",
        "SOURCE",
        "ComplexTaskSubagentV1",
        "SUBAGENT",
        "background_work",
        "task_orchestrator",
        "background_work_job.v2",
        "WORKER",
        "not a shared runtime base class",
    )
    for phrase in required_phrases:
        assert phrase in content


def test_top_level_readmes_include_current_route_families() -> None:
    """English and Chinese READMEs should both name every current model route."""

    english_readme = _read_doc("README.md")
    chinese_readme = _read_doc("README_CN.md")
    howto = _read_doc("docs/HOWTO.md")

    for route_key in ROUTE_KEYS:
        assert route_key in english_readme
        assert route_key in chinese_readme

        env_name = f"{route_key}_BASE_URL"
        if route_key == "EMBEDDING":
            env_name = "EMBEDDING_BASE_URL"
        assert env_name in howto

    removed_route = "BOUNDARY_CORE_" + "LLM"
    assert removed_route not in english_readme
    assert removed_route not in chinese_readme
    assert removed_route not in howto


def test_top_level_readmes_link_current_runtime_subsystems() -> None:
    """Both top-level docs should link the current major runtime ICDs."""

    english_readme = _read_doc("README.md")
    chinese_readme = _read_doc("README_CN.md")

    required_paths = (
        "src/control_console/README.md",
        "src/kazusa_ai_chatbot/accepted_task/README.md",
        "src/kazusa_ai_chatbot/background_work/README.md",
        "docs/DOCUMENTATION_GUIDE.md",
        "docs/SUBAGENT_INTERFACES.md",
    )
    for path in required_paths:
        assert path in english_readme
        assert path in chinese_readme


def test_howto_startup_order_matches_service_lifespan() -> None:
    """The runbook startup notes should match the service lifespan order."""

    content = _read_doc("docs/HOWTO.md")

    _assert_in_order(
        content,
        (
            "db_bootstrap()",
            "Requires and loads the latest manually seeded immutable identity revision",
            "Hydrates persistent media descriptor cache",
            "Compiles the top-level LangGraph pipeline",
            "Starts configured MCP servers",
            "Builds the runtime adapter registry and starts the chat input worker",
            "Starts the durable calendar worker",
            "Starts the self-cognition worker",
            "Starts the background-work runtime",
            "Starts the reflection worker",
        ),
    )

    _assert_in_order(
        content,
        (
            "## Character Profile",
            "kazusa-control-console",
            "Direct service startup remains available for development fallback",
            "uvicorn kazusa_ai_chatbot.service:app",
        ),
    )


def test_selected_compact_module_readmes_keep_icd_sections() -> None:
    """Compact module docs should expose enough interface detail for callers."""

    expected_sections = {
        "src/kazusa_ai_chatbot/accepted_task/README.md": (
            "Document Control",
            "Purpose",
            "Boundary",
            "Public Interfaces",
            "Persistence",
            "Failure Behavior",
            "Testing Contract",
            "Forbidden Paths",
        ),
        "src/kazusa_ai_chatbot/cognition_core_v2/README.md": (
            "Document Control",
            "Purpose",
            "Boundary",
            "Public Entrypoints",
            "Runtime Flow",
            "Failure Behavior",
            "Testing Contract",
            "Forbidden Paths",
        ),
        "src/kazusa_ai_chatbot/llm_tracing/README.md": (
            "Document Control",
            "Purpose",
            "Public Interfaces",
            "Storage Contract",
            "Failure Behavior",
            "Testing Contract",
            "Forbidden Paths",
        ),
        "src/scripts/README.md": (
            "Document Control",
            "Purpose",
            "Public Interfaces",
            "Testing Contract",
            "Forbidden Paths",
        ),
    }

    for relative_path, sections in expected_sections.items():
        content = _read_doc(relative_path)
        for section in sections:
            assert f"## {section}" in content


def test_current_doc_surfaces_are_explicit() -> None:
    """The documentation guide should expose only current tracked surfaces."""

    content = _read_doc("docs/DOCUMENTATION_GUIDE.md")

    required_phrases = (
        "docs/FUTURE_ARCHITECTURE.md",
        "docs/architecture/**/*.md",
        "development_plans/active/**/*.md",
        "development_plans/long_term/todo.md",
        "development_plans/archive/**/*.md",
        "Historical plan record",
    )
    for phrase in required_phrases:
        assert phrase in content

    assert "development_plans/reference/**/*.md" not in content
