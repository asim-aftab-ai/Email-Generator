"""
Test verification script for Structured AI Email Generator.
Tests:
1. Pydantic model creation and validation rules.
2. Boundary checks for formality_score (ge=1, le=10).
3. LangChain prompt formatting and chain creation.
4. Error handling for missing inputs.
"""

import sys
from pydantic import ValidationError
from structured_output import EmailResponse
from langchain_email import EMAIL_PROMPT, create_email_chain


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


def test_chain_creation_missing_api_key():
    """Test chain creation fails cleanly without API key."""
    try:
        create_email_chain(api_key=None)
        print("[INFO] Chain created (OPENAI_API_KEY was found in environment)")
    except ValueError as ve:
        assert "OpenAI API Key is missing" in str(ve)
        print("[PASS] test_chain_creation_missing_api_key correctly detected missing key")


def test_chain_creation_with_dummy_key():
    """Test chain composition with key provided."""
    chain = create_email_chain(api_key="sk-dummykeyforstructuraltest1234567890")
    assert chain is not None
    print("[PASS] test_chain_creation_with_dummy_key successfully composed LCEL pipeline")


if __name__ == "__main__":
    print("Running Structured AI Email Generator test suite...")
    test_pydantic_valid()
    test_pydantic_invalid_formality_score()
    test_pydantic_empty_string_validation()
    test_prompt_template_formatting()
    test_chain_creation_missing_api_key()
    test_chain_creation_with_dummy_key()
    print("\nAll unit and integration checks passed successfully!")

