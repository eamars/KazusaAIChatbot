"""Validation tests for global character growth candidates."""

from __future__ import annotations

from kazusa_ai_chatbot.global_character_growth import validation


def test_accepts_stable_global_communication_growth() -> None:
    """Repeated global communication evidence should enter drift."""

    result = validation.validate_candidate_response(
        parsed_response={
            "candidate_deltas": [
                _candidate(
                    guidance=(
                        "亲密度上升时，保持关心可见，同时保留清晰同意和退出空间。"
                    ),
                    source_card_ids=["card-1", "card-2"],
                    supporting_dates=["2026-05-01", "2026-05-03"],
                )
            ],
            "summary": "one stable candidate",
        },
        memory_cards=[_card("card-1"), _card("card-2", date="2026-05-03")],
        current_trait_rows=[],
    )

    assert len(result["accepted_candidates"]) == 1
    accepted = result["accepted_candidates"][0]
    assert accepted["candidate_id"].startswith("gcc_")
    assert accepted["evidence_strength"] > 0.9
    assert result["rejected_candidates"] == []


def test_rejects_technology_and_domain_topic_candidates() -> None:
    """Technology skill must not become personality."""

    result = validation.validate_candidate_response(
        parsed_response={
            "candidate_deltas": [
                _candidate(
                    growth_axis="other_communication",
                    trait_name="Debugging competence",
                    guidance="Become more confident when explaining Python async bugs.",
                    scope_assessment="domain_topic",
                )
            ],
            "summary": "domain noise",
        },
        memory_cards=[_card("card-1"), _card("card-2")],
        current_trait_rows=[],
    )

    assert result["accepted_candidates"] == []
    assert "scope_assessment" in result["rejected_candidates"][0]["reason"]


def test_rejects_chinese_domain_topic_candidates() -> None:
    """Chinese technology wording must not bypass domain-topic validation."""

    result = validation.validate_candidate_response(
        parsed_response={
            "candidate_deltas": [
                _candidate(
                    growth_axis="other_communication",
                    trait_name="技术解释自信",
                    guidance="可以更稳定地展示技术调试和代码解释能力。",
                )
            ],
            "summary": "domain noise",
        },
        memory_cards=[_card("card-1"), _card("card-2")],
        current_trait_rows=[],
    )

    assert result["accepted_candidates"] == []
    assert result["rejected_candidates"][0]["reason"] == "domain_topic"


def test_rejects_non_chinese_trait_free_text() -> None:
    """Prompt free-text language policy should be enforced after parsing."""

    result = validation.validate_candidate_response(
        parsed_response={
            "candidate_deltas": [
                _candidate(
                    trait_name="validation-first boundary setting",
                    guidance="先确认状态，再清楚设边界，并给出下一步选择。",
                )
            ],
            "summary": "language mismatch",
        },
        memory_cards=[_card("card-1"), _card("card-2")],
        current_trait_rows=[],
    )

    assert result["accepted_candidates"] == []
    assert result["rejected_candidates"][0]["reason"] == "language"


def test_rejects_user_specific_style_candidates() -> None:
    """One user's preferred interaction loop cannot become global growth."""

    result = validation.validate_candidate_response(
        parsed_response={
            "candidate_deltas": [
                _candidate(
                    guidance="Use reward challenges for this user whenever he asks.",
                    scope_assessment="user_specific",
                )
            ],
            "summary": "user specific",
        },
        memory_cards=[_card("card-1"), _card("card-2")],
        current_trait_rows=[],
    )

    assert result["accepted_candidates"] == []
    assert "scope_assessment" in result["rejected_candidates"][0]["reason"]


def test_rejects_private_or_source_detail_leakage() -> None:
    """Prompt-facing guidance must not carry user ids or source ids."""

    result = validation.validate_candidate_response(
        parsed_response={
            "candidate_deltas": [
                _candidate(
                    guidance="For qq:673225019 after reflection-run-1, soften replies.",
                    private_detail_risk="high",
                )
            ],
            "summary": "private detail",
        },
        memory_cards=[_card("card-1"), _card("card-2")],
        current_trait_rows=[],
    )

    assert result["accepted_candidates"] == []
    assert "private_detail_risk" in result["rejected_candidates"][0]["reason"]


def test_rejects_unknown_source_cards_and_dates() -> None:
    """Candidates must cite only prompt input cards and derived dates."""

    result = validation.validate_candidate_response(
        parsed_response={
            "candidate_deltas": [
                _candidate(
                    source_card_ids=["card-1", "missing-card"],
                    supporting_dates=["2026-05-01", "2026-05-08"],
                )
            ],
            "summary": "bad refs",
        },
        memory_cards=[_card("card-1"), _card("card-2")],
        current_trait_rows=[],
    )

    assert result["accepted_candidates"] == []
    reason = result["rejected_candidates"][0]["reason"]
    assert "source_card_ids" in reason or "supporting_dates" in reason


def test_rejects_duplicate_candidates_against_active_traits() -> None:
    """Duplicate global guidance should not inflate the trait ledger."""

    current_traits = [{
        "trait_id": "trait-1",
        "growth_axis": "clarity",
        "guidance": (
            "亲密度上升时，保持关心可见，同时保留清晰同意和退出空间。"
        ),
        "maturity_band": "promoted",
        "status": "active",
    }]

    result = validation.validate_candidate_response(
        parsed_response={
            "candidate_deltas": [
                _candidate(
                    guidance=(
                        "亲密度上升时，保持关心可见，同时保留清晰同意和退出空间。"
                    )
                )
            ],
            "summary": "duplicate",
        },
        memory_cards=[_card("card-1"), _card("card-2")],
        current_trait_rows=current_traits,
    )

    assert result["accepted_candidates"] == []
    assert "duplicate" in result["rejected_candidates"][0]["reason"]


def test_caps_accepted_candidates_per_run() -> None:
    """The validator should keep only the bounded strongest candidates."""

    axes_and_guidance = [
        ("clarity", "解释模糊压力前，先问一个直接澄清问题。"),
        ("guarded_care", "表达关心时，同时保留明确的退出空间。"),
        ("recovery_style", "尴尬后先短暂承认，再自然推进。"),
        ("playful_challenge", "使用轻量挑战维持玩笑感，避免惩罚感。"),
        ("trust_calibration", "等待重复可靠证据后，再把亲近视为稳定。"),
        ("emotional_exposure", "支持持续稳定时，允许小幅表达真实感受。"),
    ]
    candidates = [
        _candidate(
            growth_axis=growth_axis,
            trait_name=f"成长候选{index}",
            guidance=guidance,
            source_card_ids=[f"card-{index}-a", f"card-{index}-b"],
            supporting_dates=["2026-05-01", "2026-05-02"],
        )
        for index, (growth_axis, guidance) in enumerate(axes_and_guidance)
    ]
    cards = [
        _card(f"card-{index}-{suffix}", date="2026-05-01" if suffix == "a" else "2026-05-02")
        for index in range(6)
        for suffix in ("a", "b")
    ]

    result = validation.validate_candidate_response(
        parsed_response={"candidate_deltas": candidates, "summary": "capped"},
        memory_cards=cards,
        current_trait_rows=[],
    )

    assert len(result["accepted_candidates"]) == 4
    assert len(result["rejected_candidates"]) == 2


def _candidate(
    *,
    growth_axis: str = "clarity",
    trait_name: str = "关心与边界",
    guidance: str = "保持关心可见，同时保留清晰同意和退出空间。",
    source_card_ids: list[str] | None = None,
    supporting_dates: list[str] | None = None,
    scope_assessment: str = "global",
    support_level: str = "stable",
    confidence: str = "high",
    private_detail_risk: str = "low",
) -> dict:
    """Build one LLM candidate fixture."""

    return {
        "candidate_action": "observe_trait",
        "growth_axis": growth_axis,
        "trait_name": trait_name,
        "guidance": guidance,
        "source_card_ids": source_card_ids or ["card-1", "card-2"],
        "supporting_dates": supporting_dates or ["2026-05-01", "2026-05-02"],
        "scope_assessment": scope_assessment,
        "support_level": support_level,
        "confidence": confidence,
        "private_detail_risk": private_detail_risk,
        "novelty_reason": "new repeated pattern",
        "stability_reason": "多日重复出现",
        "rejection_reason": "",
    }


def _card(card_id: str, *, date: str = "2026-05-01") -> dict:
    """Build one prompt memory card fixture."""

    return {
        "source_card_id": card_id,
        "memory_unit_id": f"memory-{card_id}",
        "memory_name": f"Memory {card_id}",
        "memory_type": "defense_rule",
        "content": "Repeated general communication pattern.",
        "character_local_dates": [date, "2026-05-02"],
        "source_reflection_run_ids": [f"run-{card_id}"],
        "confidence_note": "stable support",
    }
