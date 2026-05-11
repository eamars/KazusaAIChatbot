from __future__ import annotations

from scripts import profile_embedding_prefix_modes as module


def test_prefix_mode_definitions_include_researched_modes() -> None:
    """Prefix profiling should expose the three researched comparison modes."""

    modes = module.prefix_mode_definitions()
    mode_names = {mode["name"] for mode in modes}

    assert mode_names == {
        "no_prefix",
        "transformers_manual_prefix",
        "sentence_transformers_prompt_equivalent",
    }


def test_sentence_transformers_mode_uses_model_config_prompt_strings() -> None:
    """SentenceTransformers mode should mirror the checked model config."""

    mapping = module.sentence_transformers_prompt_config()

    assert mapping == {
        "query": "search_query: ",
        "passage": "search_document: ",
    }


def test_transformers_and_sentence_transformers_modes_match() -> None:
    """For this Nomic model, both prefixed modes should produce equal input text."""

    query = "gpu market share"
    document = "Steam says RTX 3060 is popular."

    transformers_query = module.effective_query_text(
        "transformers_manual_prefix",
        query,
    )
    sentence_query = module.effective_query_text(
        "sentence_transformers_prompt_equivalent",
        query,
    )
    transformers_document = module.effective_document_text(
        "transformers_manual_prefix",
        document,
    )
    sentence_document = module.effective_document_text(
        "sentence_transformers_prompt_equivalent",
        document,
    )

    assert sentence_query == transformers_query == "search_query: gpu market share"
    assert sentence_document == transformers_document == (
        "search_document: Steam says RTX 3060 is popular."
    )


def test_prefix_mode_metrics_track_hit_rank_and_false_positive_rank() -> None:
    """Profile metrics should preserve ranks for true and forbidden hits."""

    rows = [
        {"rank": 1, "body_text": "unrelated row"},
        {"rank": 2, "body_text": "answer mentions RTX 3060"},
        {"rank": 3, "body_text": "breakfast delivery false trail"},
    ]
    positive_case = {
        "case_id": "positive",
        "kind": "positive",
        "expected_any": ["RTX 3060"],
        "forbidden_any": ["breakfast"],
    }
    negative_case = {
        "case_id": "negative",
        "kind": "negative",
        "expected_any": [],
        "forbidden_any": ["breakfast"],
    }

    positive_metrics = module.evaluate_prefix_mode_case(
        positive_case,
        rows,
        top_ks=[1, 2, 3],
    )
    negative_metrics = module.evaluate_prefix_mode_case(
        negative_case,
        rows,
        top_ks=[1, 2, 3],
    )

    assert positive_metrics["first_expected_hit_rank"] == 2
    assert positive_metrics["first_forbidden_hit_rank"] == 3
    assert positive_metrics["hit_at"] == {"1": False, "2": True, "3": True}
    assert negative_metrics["first_forbidden_hit_rank"] == 3
    assert negative_metrics["false_positive_at"] == {
        "1": False,
        "2": False,
        "3": True,
    }
