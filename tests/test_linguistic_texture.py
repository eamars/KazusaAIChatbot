"""Unit tests for linguistic-texture translation helpers in L3.

Tests cover:
- _pick_level: clamping and rounding
- All 10 get_*_description functions: boundary values (0.0, 0.5, 1.0) and
  clamping behaviour (<0.0, >1.0)

No LLM calls are made; these tests run fully offline.
"""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.nodes.linguistic_texture import (
    _pick_level,
    get_abstraction_reframing_description,
    get_counter_questioning_description,
    get_direct_assertion_description,
    get_emotional_leakage_description,
    get_formalism_avoidance_description,
    get_fragmentation_description,
    get_hesitation_density_description,
    get_rhythmic_bounce_description,
    get_self_deprecation_description,
    get_softener_density_description,
)

# All 10 translation functions under test
ALL_DESCRIBERS = [
    get_fragmentation_description,
    get_hesitation_density_description,
    get_counter_questioning_description,
    get_softener_density_description,
    get_formalism_avoidance_description,
    get_abstraction_reframing_description,
    get_direct_assertion_description,
    get_emotional_leakage_description,
    get_rhythmic_bounce_description,
    get_self_deprecation_description,
]


# ---------------------------------------------------------------------------
# _pick_level
# ---------------------------------------------------------------------------

class TestPickLevel:
    def test_zero(self):
        assert _pick_level(0.0) == 0

    def test_one(self):
        assert _pick_level(1.0) == 10

    def test_midpoint(self):
        assert _pick_level(0.5) == 5

    def test_clamp_below_zero(self):
        assert _pick_level(-0.5) == 0

    def test_clamp_above_one(self):
        assert _pick_level(1.5) == 10

    def test_rounding_down(self):
        # 0.34 * 10 = 3.4 → rounds to 3
        assert _pick_level(0.34) == 3

    def test_rounding_up(self):
        # 0.35 * 10 = 3.5 → tolerate either rounding mode
        assert _pick_level(0.35) in (3, 4)

    def test_all_integer_tenths(self):
        for i in range(11):
            assert _pick_level(i / 10) == i


# ---------------------------------------------------------------------------
# Each translation function: return type, non-empty, boundary correctness
# ---------------------------------------------------------------------------

class TestTranslationFunctions:
    @pytest.mark.parametrize("fn", ALL_DESCRIBERS)
    def test_returns_non_empty_string_at_zero(self, fn):
        result = fn(0.0)
        assert isinstance(result, str) and result.strip()

    @pytest.mark.parametrize("fn", ALL_DESCRIBERS)
    def test_returns_non_empty_string_at_half(self, fn):
        result = fn(0.5)
        assert isinstance(result, str) and result.strip()

    @pytest.mark.parametrize("fn", ALL_DESCRIBERS)
    def test_returns_non_empty_string_at_one(self, fn):
        result = fn(1.0)
        assert isinstance(result, str) and result.strip()

    @pytest.mark.parametrize("fn", ALL_DESCRIBERS)
    def test_clamp_below_zero_returns_same_as_zero(self, fn):
        assert fn(-1.0) == fn(0.0)

    @pytest.mark.parametrize("fn", ALL_DESCRIBERS)
    def test_clamp_above_one_returns_same_as_one(self, fn):
        assert fn(2.0) == fn(1.0)

    @pytest.mark.parametrize("fn", ALL_DESCRIBERS)
    def test_low_differs_from_high(self, fn):
        """0.0 and 1.0 should map to distinct descriptions."""
        assert fn(0.0) != fn(1.0)

    @pytest.mark.parametrize("fn", ALL_DESCRIBERS)
    def test_has_eleven_distinct_levels(self, fn):
        """Each function must cover 11 distinct descriptions."""
        outputs = {fn(i / 10) for i in range(11)}
        assert len(outputs) == 11, (
            f"{fn.__name__} does not have 11 distinct descriptions"
        )


# ---------------------------------------------------------------------------
# Semantic spot-checks: descriptions must include expected markers
# ---------------------------------------------------------------------------

class TestSemanticContent:
    def test_fragmentation_low_is_smooth(self):
        assert "完整" in get_fragmentation_description(0.0)

    def test_fragmentation_high_mentions_fragments(self):
        desc = get_fragmentation_description(1.0)
        assert "断片" in desc or "碎片" in desc or "省略" in desc

    def test_hesitation_low_no_fillers(self):
        desc = get_hesitation_density_description(0.0)
        assert "那个" in desc or "嗯" in desc  # mentioned as forbidden markers

    def test_hesitation_high_has_fillers(self):
        assert "那个" in get_hesitation_density_description(1.0) or "嗯" in get_hesitation_density_description(1.0)

    def test_formalism_avoidance_low_allows_formal(self):
        desc = get_formalism_avoidance_description(0.0)
        assert "因为" in desc or "然而" in desc or "综上" in desc

    def test_formalism_avoidance_high_bans_formal(self):
        desc = get_formalism_avoidance_description(1.0)
        assert "禁" in desc or "拒绝" in desc or "严禁" in desc

    def test_self_deprecation_low_forbids_self_minimising(self):
        desc = get_self_deprecation_description(0.0)
        assert "禁" in desc or "从不" in desc

    def test_emotional_leakage_high_mentions_punctuation(self):
        desc = get_emotional_leakage_description(1.0)
        assert "颤" in desc or "破碎" in desc or "标点" in desc

    def test_direct_assertion_low_is_indirect(self):
        desc = get_direct_assertion_description(0.0)
        assert "绕" in desc or "侧面" in desc or "含糊" in desc

    def test_direct_assertion_high_is_direct(self):
        desc = get_direct_assertion_description(1.0)
        assert "直" in desc or "结论" in desc

    def test_kazusa_fragmentation_level(self):
        """fragmentation=0.3 → level 3 → '五六句' expected."""
        assert "五六句" in get_fragmentation_description(0.3)

    def test_kazusa_emotional_leakage_level(self):
        """emotional_leakage=0.7 → level 7 → visible leakage description."""
        desc = get_emotional_leakage_description(0.7)
        assert "省略号" in desc or "渗入" in desc or "语序" in desc

    def test_kazusa_self_deprecation_level(self):
        """self_deprecation=0.15 → level 2 → infrequent self-deprecation."""
        desc = get_self_deprecation_description(0.15)
        assert "很少" in desc or "偶尔" in desc
