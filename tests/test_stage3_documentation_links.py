"""Documentation contracts for the native Stage 3 runtime boundary."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGE3_DOCUMENTS = (
    "README.md",
    "README_CN.md",
    "docs/HOWTO.md",
    "src/control_console/README.md",
    "src/scripts/README.md",
    "src/kazusa_ai_chatbot/brain_service/README.md",
    "src/kazusa_ai_chatbot/cognition_core_v2/README.md",
    "src/kazusa_ai_chatbot/db/README.md",
    "src/kazusa_ai_chatbot/self_cognition/README.md",
)
LOCAL_MARKDOWN_LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


def _read(relative_path: str) -> str:
    """Read one Stage 3 documentation target as UTF-8."""

    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_stage3_documentation_targets_exist_and_link_local_files() -> None:
    """Changed documentation must retain resolvable local Markdown links."""

    for relative_path in STAGE3_DOCUMENTS:
        document_path = ROOT / relative_path
        content = _read(relative_path)
        for link_target in LOCAL_MARKDOWN_LINK.findall(content):
            if link_target.startswith(("http://", "https://", "mailto:")):
                continue
            target_path = (document_path.parent / link_target).resolve()
            assert target_path.is_relative_to(ROOT), link_target
            assert target_path.exists(), (
                f"{relative_path} links to missing {link_target}"
            )


def test_stage3_docs_describe_native_startup_and_runtime_ownership() -> None:
    """Operator docs must expose the fresh native startup boundary."""

    combined = "\n".join(_read(path) for path in STAGE3_DOCUMENTS)
    required_terms = (
        "CHARACTER_PROFILE_PATH",
        "_test_kazusa_core_v2",
        "cognitive_episode.v1",
        "episode_trace.v2",
        "internal_action_latches",
        "post_turn_lifecycle_records",
        "user_message",
        "internal_thought",
        "self_cognition",
        "scheduled_tick",
        "tool_result",
    )
    for term in required_terms:
        assert term in combined


def test_stage3_plan_companions_and_stage4_boundary_are_linked() -> None:
    """Lifecycle docs must expose the execution companions and handoff."""

    plan = _read(
        "development_plans/archive/completed/short_term/"
        "cognition_core_v2_stage_3_system_adoption_plan.md",
    )
    manifest = _read(
        "development_plans/archive/completed/short_term/"
        "cognition_core_v2_stage_3_execution_manifest.md",
    )
    radius = _read(
        "development_plans/archive/completed/short_term/"
        "cognition_core_v2_stage_3_change_radius.md",
    )

    for term in (
        "cognition_core_v2_stage_3_execution_manifest.md",
        "cognition_core_v2_stage_3_change_radius.md",
        "cognition_core_v2_stage_4_production_database_migration_plan.md",
        "Stage3FreshDatabaseEvidenceV1",
        "Stage3NativeSchemaManifestV1",
    ):
        assert term in plan or term in manifest or term in radius
