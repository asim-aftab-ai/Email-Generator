"""
Streamlit Frontend for Structured AI Email Generator (OpenRouter Integration).

Entry point for the application. Collects user input, calls the LangChain
generation workflow routed via OpenRouter, and displays structured, validated email fields.
"""

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_email import DEFAULT_OPENROUTER_MODEL, OPENROUTER_BASE_URL, generate_email
from structured_output import EmailResponse

# Load environment variables from .env
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Email Generator — OpenRouter & Structured Output",
    page_icon="✉️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean cards and visual separation of structured fields
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6c757d;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .field-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        border-left: 4px solid #4f46e5;
        margin-bottom: 1rem;
    }
    .field-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #4b5563;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .subject-text {
        font-size: 1.2rem;
        font-weight: 600;
        color: #111827;
    }
    .body-box {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.2rem;
        white-space: pre-wrap;
        font-family: inherit;
        line-height: 1.6;
        color: #1f2937;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_sidebar() -> tuple[str, str, float]:
    """Renders the sidebar configuration and returns user settings."""
    with st.sidebar:
        st.header("⚙️ OpenRouter Config")

        env_key = os.getenv("OPENROUTER_API_KEY", "")
        has_env_key = bool(env_key)

        api_key_input = st.text_input(
            "OpenRouter API Key",
            value=env_key if has_env_key else "",
            type="password",
            help="Your OpenRouter key (sk-or-v1-...). Handled securely in memory.",
            placeholder="sk-or-v1-...",
        )

        if has_env_key and not api_key_input:
            api_key_input = env_key

        if api_key_input:
            st.success("✓ OpenRouter key configured", icon="🔑")
        else:
            st.warning("⚠️ OpenRouter API Key required.", icon="⚠️")

        st.caption(f"**Endpoint:** `{OPENROUTER_BASE_URL}`")

        st.divider()

        st.subheader("Model Selection")
        env_model = os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)

        model_preset_options = [
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "anthropic/claude-3.5-haiku",
            "meta-llama/llama-3.3-70b-instruct",
            "Custom Model ID...",
        ]

        preset_index = 0
        if env_model in model_preset_options:
            preset_index = model_preset_options.index(env_model)
        elif env_model != DEFAULT_OPENROUTER_MODEL:
            preset_index = len(model_preset_options) - 1

        selected_preset = st.selectbox(
            "OpenRouter Model",
            options=model_preset_options,
            index=preset_index,
            help="Select any model supported by OpenRouter with tool-calling/structured outputs.",
        )

        if selected_preset == "Custom Model ID...":
            model_name = st.text_input(
                "Enter Model ID",
                value=env_model if env_model not in model_preset_options else "",
                placeholder="e.g., google/gemini-2.0-flash-001",
            )
        else:
            model_name = selected_preset

        temperature = st.slider(
            "Creativity (Temperature)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values make output more creative; lower values make it more deterministic.",
        )

        st.divider()
        st.subheader("📐 Architecture")
        st.markdown(
            """
            **Pipeline Flow:**
            1. User Input Form
            2. `ChatPromptTemplate`
            3. `ChatOpenAI` (OpenRouter API)
            4. `with_structured_output`
            5. `Pydantic` Validation (`EmailResponse`)
            6. Structured UI Display
            """
        )

    return api_key_input, model_name, temperature


def render_results(email: EmailResponse):
    """Renders the structured fields into separated, dedicated UI components."""
    st.markdown("---")
    st.subheader("📬 Generated Email")

    # 1. Subject Line Field
    st.markdown(
        f"""
        <div class="field-card">
            <div class="field-label">Subject Line</div>
            <div class="subject-text">{email.subject}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Metadata Metrics: Tone, Read Time, Formality Score
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Tone", value=email.tone)
    with col2:
        st.metric(label="Est. Read Time", value=email.estimated_read_time)
    with col3:
        st.metric(label="Formality Score", value=f"{email.formality_score} / 10")
        st.progress(email.formality_score / 10.0)

    # 3. Email Body
    st.markdown("### Email Body")
    st.markdown(
        f'<div class="body-box">{email.body}</div>',
        unsafe_allow_html=True,
    )

    # Quick copy utility
    with st.expander("📋 Copy Email Text"):
        copy_text = f"Subject: {email.subject}\n\n{email.body}"
        st.text_area("Plain text version (ready to copy)", value=copy_text, height=180)

    # Educational Inspector: Raw Validated Pydantic Schema
    with st.expander("🔍 Inspect Validated Pydantic Data (JSON Schema)"):
        st.json(email.model_dump())


def main():
    api_key, model_name, temperature = render_sidebar()

    st.markdown('<div class="main-header">Structured AI Email Generator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Generate validated, structured executive emails via OpenRouter & LangChain.</div>',
        unsafe_allow_html=True,
    )

    # Main input form
    with st.form("email_generator_form"):
        topic = st.text_area(
            "What is this email about? (Required)",
            placeholder="e.g., Ask my team lead for a 3-day extension on the project deliverable due to testing delays.",
            height=120,
        )

        col_tone, col_rec = st.columns([1, 1])

        with col_tone:
            tone_options = [
                "Professional",
                "Friendly / Casual",
                "Urgent & Direct",
                "Persuasive",
                "Apologetic & Courteous",
                "Executive / Concise",
                "Custom...",
            ]
            selected_tone = st.selectbox("Desired Tone", options=tone_options, index=0)
            custom_tone = ""
            if selected_tone == "Custom...":
                custom_tone = st.text_input("Enter custom tone", placeholder="e.g., Diplomatic yet firm")

        with col_rec:
            recipient = st.text_input(
                "Target Recipient (Optional)",
                placeholder="e.g., Team Lead, HR Manager, Client, All Staff",
            )

        submit_button = st.form_submit_button("✨ Generate Structured Email", type="primary", use_container_width=True)

    # Handling Form Submission
    if submit_button:
        # 1. Validate User Input
        if not topic or not topic.strip():
            st.error("Please enter what the email is about before generating.", icon="⚠️")
            return

        # 2. Validate API Key & Model
        if not api_key or not api_key.strip():
            st.error(
                "OpenRouter API Key is missing! Please provide your key in the sidebar or set `OPENROUTER_API_KEY` in `.env`.",
                icon="🔑",
            )
            return

        if not model_name or not model_name.strip():
            st.error(
                "OpenRouter model identifier is missing. Please select or enter a model ID.",
                icon="⚠️",
            )
            return

        final_tone = custom_tone.strip() if selected_tone == "Custom..." and custom_tone else selected_tone

        # 3. Call LangChain Pipeline with graceful error handling
        with st.spinner(f"Generating structured email via OpenRouter ({model_name})..."):
            try:
                email_result: EmailResponse = generate_email(
                    topic=topic,
                    tone=final_tone,
                    recipient=recipient,
                    api_key=api_key,
                    model_name=model_name,
                    temperature=temperature,
                )
                render_results(email_result)

            except ValueError as ve:
                st.error(f"Validation Error: {str(ve)}", icon="❌")
            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg or "unauthorized" in error_msg.lower() or "authentication" in error_msg.lower():
                    st.error("Authentication failed: Invalid OpenRouter API Key. Please verify your credentials at openrouter.ai/keys.", icon="🔑")
                elif "402" in error_msg or "credits" in error_msg.lower():
                    st.error("Insufficient credits: Your OpenRouter account balance is empty. Please top up at openrouter.ai/credits.", icon="💳")
                elif "429" in error_msg or "rate_limit" in error_msg.lower():
                    st.error("OpenRouter Rate limit reached. Please wait a moment and try again.", icon="⏳")
                else:
                    st.error(f"An unexpected error occurred during generation: {error_msg}", icon="🚨")


if __name__ == "__main__":
    main()
