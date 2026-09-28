"""
LangChain Email Generation Module (OpenRouter Integration).

Implements the modern LangChain composable architecture (LCEL) routed through OpenRouter:
    Prompt Template -> Chat Model (OpenRouter endpoint) -> Structured Output (Pydantic Validation)

Why OpenRouter?
    OpenRouter provides an OpenAI-compatible API interface (base URL: https://openrouter.ai/api/v1)
    that unlocks access to dozens of state-of-the-art models (OpenAI, Anthropic, Meta, etc.)
    under a unified billing and endpoint structure, while preserving full compatibility
    with LangChain's ChatOpenAI abstraction and structured output capabilities.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from structured_output import EmailResponse

# Load environment variables from .env if present
load_dotenv()

# Centralized OpenRouter Configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"

# System prompt defining generation rules
SYSTEM_INSTRUCTIONS = """You are an expert executive communications specialist and email copywriter.
Your task is to write high-impact, professional emails tailored precisely to the user's intent, desired tone, and recipient context.

Rules:
1. Write a clear, concise, and compelling subject line that captures the essence of the email.
2. Structure the email body with:
   - An appropriate opening greeting / salutation tailored to the recipient.
   - Clean, readable paragraphs separated by line breaks.
   - A clear call-to-action (CTA) or next steps where appropriate.
   - A professional sign-off / closing.
3. Match the requested tone accurately.
4. Estimate realistic reading time based on word count (e.g., '~30 seconds', '1 minute').
5. Assign a formality score from 1 (ultra-casual, text-like) to 10 (strict legal/diplomatic formal).
"""

# Reusable ChatPromptTemplate for composing user inputs into the pipeline
EMAIL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_INSTRUCTIONS),
    (
        "human",
        "Generate a structured email for the following request:\n\n"
        "• Email Purpose / Topic: {topic}\n"
        "• Desired Tone: {tone}\n"
        "• Target Recipient: {recipient}\n\n"
        "Return the response adhering strictly to the structured schema."
    ),
])


def create_email_chain(
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.7,
):
    """
    Constructs the modern LangChain LCEL pipeline with structured output via OpenRouter.

    Pipeline:
        ChatPromptTemplate | ChatOpenAI(OpenRouter endpoint).with_structured_output(EmailResponse)

    Args:
        api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY environment variable).
        model_name: OpenRouter model identifier (defaults to OPENROUTER_MODEL or 'openai/gpt-4o-mini').
        temperature: Sampling temperature for generation creativity (0.0 to 1.0).

    Returns:
        A runnable LangChain LCEL chain that produces validated EmailResponse instances.
    """
    resolved_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not resolved_api_key or not resolved_api_key.strip():
        raise ValueError(
            "OpenRouter API Key is missing. Please set OPENROUTER_API_KEY in your .env file "
            "or enter it in the application sidebar."
        )

    resolved_model = (
        model_name
        or os.getenv("OPENROUTER_MODEL")
        or DEFAULT_OPENROUTER_MODEL
    )
    if not resolved_model or not resolved_model.strip():
        raise ValueError(
            "OpenRouter model identifier is missing. Please set OPENROUTER_MODEL or select a model."
        )

    # Initialize ChatOpenAI configured with OpenRouter base URL and headers
    llm = ChatOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=resolved_api_key.strip(),
        model=resolved_model.strip(),
        temperature=temperature,
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Structured AI Email Generator",
        },
    )

    # Bind the Pydantic schema for native structured output
    structured_llm = llm.with_structured_output(EmailResponse)

    # Modern composable LCEL chain: Prompt -> Structured LLM
    chain = EMAIL_PROMPT | structured_llm
    return chain


def generate_email(
    topic: str,
    tone: str = "Professional",
    recipient: str = "General",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.7,
) -> EmailResponse:
    """
    High-level entry point to invoke the OpenRouter-backed email generation workflow.

    Args:
        topic: Description of what the email is about.
        tone: The target tone for the email (e.g., Professional, Friendly, Urgent).
        recipient: Optional description of who the email is addressed to.
        api_key: Optional OpenRouter API key override.
        model_name: Optional OpenRouter model identifier override.
        temperature: Creativity parameter (default 0.7).

    Returns:
        A validated EmailResponse object.
    """
    if not topic or not topic.strip():
        raise ValueError("Email topic / request cannot be empty.")

    chain = create_email_chain(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
    )

    # Invoke the chain with dynamic inputs
    response: EmailResponse = chain.invoke({
        "topic": topic.strip(),
        "tone": tone.strip() if tone else "Professional",
        "recipient": recipient.strip() if recipient else "General Audience",
    })

    return response
