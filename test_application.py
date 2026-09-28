"""
Test verification script for Structured AI Email Generator (OpenRouter Integration).

Tests:
1. Pydantic model creation and validation rules.
2. Boundary checks for formality_score (ge=1, le=10).
3. Non-empty string validators for all text fields.
4. LangChain prompt formatting and variable composition.
5. Error handling for missing OPENROUTER_API_KEY.
6. OpenRouter pipeline construction with base_url and custom model.
"""

import sys
from pydantic import ValidationError
from structured_output import EmailResponse
from langchain_email import (
    EMAIL_PROMPT,
    OPENROUTER_BASE_URL,
    DEFAULT_OPENROUTER_MODEL,
    create_email_chain,
)


def test_pydantic_valid():
    """Test valid model instantiation."""
    data = {
        "subject": "Team Sync Follow-up",
        "body": "Hi Team,\n\nThanks for attending today's sync.\n\nBest,\nAlex",
        "tone": "Professional",
        "estimated_read_time": "30 seconds",
        "formality_score": 7,
    }
    email = EmailResponse(**data)
    assert email.subject == "Team Sync Follow-up"
    assert email.formality_score == 7
    print("[PASS] test_pydantic_valid passed")


def test_pydantic_invalid_formality_score():
    """Test that formality scores outside 1-10 raise validation errors."""
    data_low = {
        "subject": "Test",
        "body": "Test body",
        "tone": "Casual",
        "estimated_read_time": "15s",
        "formality_score": 0,
    }
    try:
        EmailResponse(**data_low)
        print("[FAIL] test_pydantic_invalid_formality_score (low) failed to raise")
        sys.exit(1)
    except ValidationError:
        print("[PASS] test_pydantic_invalid_formality_score (low) correctly rejected score=0")

    data_high = {
        "subject": "Test",
        "body": "Test body",
        "tone": "Formal",
        "estimated_read_time": "15s",
        "formality_score": 11,
    }
    try:
        EmailResponse(**data_high)
        print("[FAIL] test_pydantic_invalid_formality_score (high) failed to raise")
        sys.exit(1)
    except ValidationError:
        print("[PASS] test_pydantic_invalid_formality_score (high) correctly rejected score=11")


def test_pydantic_empty_string_validation():
    """Test that empty string fields are rejected."""
    data = {
        "subject": "   ",
        "body": "Valid body",
        "tone": "Formal",
        "estimated_read_time": "15s",
        "formality_score": 5,
    }
    try:
        EmailResponse(**data)
        print("[FAIL] test_pydantic_empty_string_validation failed to raise")
        sys.exit(1)
    except ValidationError:
        print("[PASS] test_pydantic_empty_string_validation correctly rejected empty subject")


def test_prompt_template_formatting():
    """Test prompt formatting with variables."""
    messages = EMAIL_PROMPT.format_messages(
        topic="Project status update",
        tone="Concise",
        recipient="VP of Engineering",
    )
    assert len(messages) == 2
    assert "Project status update" in messages[1].content
    assert "VP of Engineering" in messages[1].content
    print("[PASS] test_prompt_template_formatting passed")


def test_chain_creation_missing_openrouter_key():
    """Test chain creation fails cleanly without OPENROUTER_API_KEY."""
    import os
    original_key = os.environ.pop("OPENROUTER_API_KEY", None)
    try:
        create_email_chain(api_key=None)
        print("[FAIL] Expected ValueError for missing OpenRouter API key")
        sys.exit(1)
    except ValueError as ve:
        assert "OpenRouter API Key is missing" in str(ve)
        print("[PASS] test_chain_creation_missing_openrouter_key correctly detected missing key")
    finally:
        if original_key is not None:
            os.environ["OPENROUTER_API_KEY"] = original_key


def test_openrouter_chain_creation():
    """Test chain composition with OpenRouter base_url and model."""
    assert OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"
    assert DEFAULT_OPENROUTER_MODEL == "openai/gpt-4o-mini"

    chain = create_email_chain(
        api_key="sk-or-v1-dummy-key-for-testing",
        model_name="openai/gpt-4o-mini",
    )
    assert chain is not None
    print("[PASS] test_openrouter_chain_creation successfully composed OpenRouter LCEL pipeline")


if __name__ == "__main__":
    print("Running Structured AI Email Generator (OpenRouter) test suite...")
    test_pydantic_valid()
    test_pydantic_invalid_formality_score()
    test_pydantic_empty_string_validation()
    test_prompt_template_formatting()
    test_chain_creation_missing_openrouter_key()
    test_openrouter_chain_creation()
    print("\nAll unit and integration checks passed successfully!")
