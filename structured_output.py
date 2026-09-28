"""
Structured Output Schema for Email Generator.

Defines the Pydantic schema representing the contract for AI email generation.
This ensures the model output is validated, typed, and structured rather than raw text.
"""

from pydantic import BaseModel, Field, field_validator


class EmailResponse(BaseModel):
    """
    Pydantic schema representing a structured email response from the AI.
    
    Fields:
        subject: Concise, relevant subject line for the email.
        body: Complete email text including salutation, body paragraphs, and sign-off.
        tone: Detected or applied tone (e.g., Professional, Urgent, Casual, Persuasive).
        estimated_read_time: Human-readable reading duration estimate (e.g., '45 seconds', '2 minutes').
        formality_score: Formality score on an integer scale from 1 (very casual) to 10 (strictly formal).
    """

    subject: str = Field(
        ...,
        description="A concise, compelling email subject line that clearly reflects the intent."
    )
    body: str = Field(
        ...,
        description="The complete email body, properly formatted with greeting, organized paragraphs, and professional closing."
    )
    tone: str = Field(
        ...,
        description="The tone applied to the email (e.g., Professional, Friendly, Urgent, Persuasive, Apologetic)."
    )
    estimated_read_time: str = Field(
        ...,
        description="Estimated reading time in human-friendly format (e.g., '45 seconds', '1-2 minutes')."
    )
    formality_score: int = Field(
        ...,
        description="Formality rating on an integer scale from 1 (casual/informal) to 10 (strictly formal/legal).",
        ge=1,
        le=10,
    )

    @field_validator("subject", "body", "tone", "estimated_read_time")
    @classmethod
    def validate_non_empty_string(cls, value: str, info) -> str:
        """Ensure string fields are not empty or purely whitespace."""
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"Field '{info.field_name}' must not be empty.")
        return stripped
