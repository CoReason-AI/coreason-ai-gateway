# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_ai_gateway

import pytest
from fastapi import HTTPException

from coreason_ai_gateway.routing import resolve_provider_path


def test_routing_edge_cases_empty_and_whitespace() -> None:
    """Test that empty strings or whitespace raise 400."""
    invalid_inputs = ["", "   ", "\n", "\t"]
    for model in invalid_inputs:
        with pytest.raises(HTTPException) as exc:
            resolve_provider_path(model)
        assert exc.value.status_code == 400


def test_routing_edge_cases_case_sensitivity() -> None:
    """
    Test that routing is case-sensitive.
    'GPT-4o' should currently fail as we expect lowercase standard IDs.
    """
    invalid_cased = ["GPT-4o", "Claude-3-Opus", "O1-preview"]
    for model in invalid_cased:
        with pytest.raises(HTTPException) as exc:
            resolve_provider_path(model)
        assert exc.value.status_code == 400


def test_routing_edge_cases_partial_prefixes() -> None:
    """Test that prefixes must match exactly including the separator if defined."""
    # "gpt-" matches, but "gpt" does not
    with pytest.raises(HTTPException) as exc:
        resolve_provider_path("gpt")
    assert exc.value.status_code == 400

    # "claude" vs "claude-"
    with pytest.raises(HTTPException) as exc:
        resolve_provider_path("claude")
    assert exc.value.status_code == 400


def test_routing_edge_cases_unicode_and_special_chars() -> None:
    """
    Test unusual but validly prefixed models.
    The router is a 'dumb' prefix matcher, so as long as it starts with 'gpt-', it should route.
    """
    # These should conceptually route to OpenAI even if the model ID is nonsense
    assert resolve_provider_path("gpt-🚀") == "infrastructure/openai"
    assert resolve_provider_path("claude-@#$") == "infrastructure/anthropic"

    # These should fail
    with pytest.raises(HTTPException) as exc:
        resolve_provider_path("🚀-gpt")
    assert exc.value.status_code == 400


def test_routing_edge_cases_long_strings() -> None:
    """Test handling of extremely long strings to ensure no regex DOS or buffer issues."""
    long_model = "gpt-" + "a" * 10000
    assert resolve_provider_path(long_model) == "infrastructure/openai"

    long_invalid = "invalid-" + "a" * 10000
    with pytest.raises(HTTPException) as exc:
        resolve_provider_path(long_invalid)
    assert exc.value.status_code == 400


def test_routing_complex_workflow_bulk_mixed_processing() -> None:
    """
    Simulates a high-volume batch processing scenario.
    Iterates through a mix of valid OpenAI, valid Anthropic, and invalid models.
    Verifies that the router correctly segregates them without state leakage.
    """
    # Define a pattern of inputs
    # pattern: (input, expected_result or None for Exception)
    base_pattern = [
        ("gpt-4o", "infrastructure/openai"),
        ("claude-3-opus", "infrastructure/anthropic"),
        ("invalid-model", None),
        ("o1-preview", "infrastructure/openai"),
        ("   ", None),
        ("claude-instant", "infrastructure/anthropic"),
        ("GPT-4", None),  # Case sensitive failure
    ]

    # Scale up the dataset to simulate load/redundancy (e.g., 7 * 200 = 1400 items)
    dataset = base_pattern * 200

    results = {"openai": 0, "anthropic": 0, "errors": 0}

    for model, expected in dataset:
        try:
            path = resolve_provider_path(model)
            if expected is None:
                # Should have failed but didn't
                pytest.fail(f"Model '{model}' should have raised HTTPException but returned '{path}'")

            assert path == expected

            if "openai" in path:
                results["openai"] += 1
            elif "anthropic" in path:
                results["anthropic"] += 1

        except HTTPException:
            if expected is not None:
                # Should have passed but didn't
                pytest.fail(f"Model '{model}' should have returned '{expected}' but raised HTTPException")
            results["errors"] += 1

    # Verify distribution
    # 2 valid openai types in pattern * 200 = 400
    # 2 valid anthropic types in pattern * 200 = 400
    # 3 invalid types in pattern * 200 = 600
    assert results["openai"] == 400
    assert results["anthropic"] == 400
    assert results["errors"] == 600
