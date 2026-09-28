"""
LangChain Email Generation Module.

Implements the modern LangChain composable architecture (LCEL):
    Prompt Template -> ChatOpenAI -> Structured Output (Pydantic Validation)

Historical Note for Learners:
    In earlier versions of LangChain, workflows were constructed using `LLMChain`
    with string-based output parsers (e.g., `PydanticOutputParser` or `OutputFixingParser`),
    which prompted the model to format raw JSON and parsed it via regex or json.loads.
    In modern LangChain, we compose pipelines using the pipe operator (`|`) and
    leverage `.with_structured_output(PydanticModel)` which interfaces with the
    underlying model's native tool-calling / JSON schema enforcement.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from structured_output import EmailResponse

# Load environment variables from .env if present
load_dotenv()

# System prompt giving explicit generation rules to the model
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
    Constructs the modern LangChain LCEL pipeline with structured output.

    Pipeline:
        ChatPromptTemplate | ChatOpenAI.with_structured_output(EmailResponse)

    Args:
        api_key: OpenAI API key (falls back to OPENAI_API_KEY environment variable).
        model_name: OpenAI model identifier (default: gpt-4o-mini or OPENAI_MODEL env).
        temperature: Sampling temperature for generation creativity (0.0 to 1.0).

    Returns:
        A runnable LangChain LCEL chain that produces validated EmailResponse instances.
    """
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "OpenAI API Key is missing. Please set OPENAI_API_KEY in your .env file "
            "or enter it in the application sidebar."
        )

    resolved_model = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Initialize ChatOpenAI LLM
    llm = ChatOpenAI(
        model=resolved_model,
        temperature=temperature,
        api_key=resolved_api_key,
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
    High-level entry point to invoke the LangChain email generation workflow.

    Args:
        topic: Description of what the email is about.
        tone: The target tone for the email (e.g., Professional, Friendly, Urgent).
        recipient: Optional description of who the email is addressed to.
        api_key: Optional OpenAI API key override.
        model_name: Optional model override.
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
