# Project 02: Structured AI Email Generator

A complete, production-grade GenAI application built with **LangChain (LCEL)**, **OpenAI Structured Outputs**, **Pydantic v2**, and **Streamlit**.

Unlike basic prompt-and-response applications that treat AI output as unstructured text, this project implements a contract-based architecture where the LLM produces strictly validated, strongly typed data.

---

## Architecture Overview

```text
                    USER
                      │
                      ▼
              ┌───────────────┐
              │   Streamlit   │  (email_generator.py)
              │   Frontend    │  Collects topic, tone, & recipient
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ Prompt Template│  (langchain_email.py)
              │ (ChatPrompt)  │  Composes system & user prompt
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │  ChatOpenAI   │  (gpt-4o-mini / gpt-4o)
              │  Model Call   │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ Structured    │  .with_structured_output(EmailResponse)
              │    Output     │  Native function calling / JSON schema
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │    Pydantic   │  (structured_output.py)
              │   Validation  │  Validates fields, types, & ranges
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ Parsed Email  │  Validated EmailResponse instance
              │     Data      │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │   Streamlit   │  Renders subject, body, tone, read time,
              │    Results    │  and formality score separately
              └───────────────┘
```

---

## 1. What the Project Does

The **Structured AI Email Generator** transforms high-level instructions (topic, target recipient, desired tone) into a fully drafted email. 

Instead of dumping an uncontrolled block of text, it extracts and returns five distinct, validated attributes:
- **`subject`**: A clear, engaging subject line.
- **`body`**: Formatted email body with salutation, structured paragraphs, call to action, and sign-off.
- **`tone`**: Verified tone applied to the writing.
- **`estimated_read_time`**: Estimated reading time (e.g., `"45 seconds"`, `"1-2 minutes"`).
- **`formality_score`**: An integer score between `1` (ultra-casual) and `10` (strict legal/formal).

---

## 2. Why the Project Exists

In production Generative AI systems, applications rarely consume freeform text directly. Downstream services require structured contracts:
- Email dispatch APIs (e.g., SendGrid, AWS SES) require `subject` and `body` as separate fields.
- Analytics pipelines track metrics like tone and formality.
- UI components render distinct cards, badges, and copy buttons rather than raw markdown strings.

This project bridges the gap between raw LLM completions and reliable, software-ready data structures.

---

## 3. Technologies Used

- **Python 3.10+** (Tested on Python 3.14 / uv runtime)
- **[uv](https://github.com/astral-sh/uv)**: Extremely fast Python package and virtual environment manager.
- **[LangChain Core & LangChain OpenAI](https://python.langchain.com/)**: Modern LangChain Expression Language (LCEL) composable pipeline.
- **[OpenAI API](https://platform.openai.com/)**: Utilizing `gpt-4o-mini` with native Structured Outputs.
- **[Pydantic v2](https://docs.pydantic.dev/)**: Data validation, typing, and schema definition.
- **[Streamlit](https://streamlit.io/)**: Interactive web application interface.
- **[python-dotenv](https://github.com/theskumar/python-dotenv)**: Safe management of environment variables.

---

## 4. File Structure & Responsibilities

```text
Email-Generator/
├── structured_output.py   # Pydantic schema (EmailResponse) and field validators
├── langchain_email.py     # Modern LangChain LCEL pipeline and prompt templates
├── email_generator.py     # Streamlit entry point and UI rendering logic
├── requirements.txt       # Project dependencies
├── .env.example           # Template for environment configuration
├── .gitignore             # Git ignore patterns (.venv, .env, __pycache__)
└── README.md              # Project documentation
```

### File Breakdown:
- **`structured_output.py`**: Defines the data contract without coupling to LangChain or Streamlit.
- **`langchain_email.py`**: Encapsulates LLM logic, prompt templates, and the LCEL runnable pipeline.
- **`email_generator.py`**: Handles user input, state management, loading feedback, and displays the structured attributes in dedicated cards.

---

## 5. Structured Outputs & Pydantic Validation

### The Pydantic Model (`EmailResponse`)
```python
class EmailResponse(BaseModel):
    subject: str = Field(description="A concise, compelling email subject line.")
    body: str = Field(description="The complete email body with greeting and sign-off.")
    tone: str = Field(description="The tone applied to the email.")
    estimated_read_time: str = Field(description="Estimated reading time.")
    formality_score: int = Field(
        description="Formality rating from 1 to 10.",
        ge=1,
        le=10
    )
```

### Why Not Just Ask the LLM for JSON?
Prompting an LLM with *"Return a JSON object"* frequently fails in production:
1. LLMs may prepend markdown code blocks (````json ... ````).
2. LLMs can omit required keys or hallucinate new ones.
3. String numbers (e.g., `"9"`) or boundary violations (e.g., score = `15`) can break downstream parsers.

Using `.with_structured_output(EmailResponse)` sends the Pydantic JSON schema directly to OpenAI's tool/schema engine, ensuring the model output conforms strictly to the schema before reaching application logic.

---

## 6. Modern LangChain (LCEL) vs Legacy `LLMChain`

### The Older Way (`LLMChain`)
Historically, LangChain used `LLMChain` paired with `PydanticOutputParser` or `OutputFixingParser`:
```python
# DEPRECATED / LEGACY PATTERN:
# chain = LLMChain(prompt=prompt, llm=llm)
# output = chain.run(input)
# parsed = output_parser.parse(output) # Fragile regex or json.loads
```
This was prone to formatting hallucinations and required retry loops.

### The Modern Way (LCEL Pipe Composition)
Modern LangChain uses declarative composition via the pipe (`|`) operator:
```python
structured_llm = llm.with_structured_output(EmailResponse)
chain = prompt | structured_llm
result: EmailResponse = chain.invoke({"topic": topic, "tone": tone, "recipient": recipient})
```
This approach is type-safe, natively integrated with OpenAI's API, and simpler to reason about.

---

## 7. Environment Variables & Setup

### 1. Configure API Key
Create a `.env` file from `.env.example`:
```bash
cp .env.example .env
```
Add your OpenAI API key inside `.env`:
```env
OPENAI_API_KEY=sk-proj-yourActualKeyHere
OPENAI_MODEL=gpt-4o-mini
```
*(Alternatively, enter your API key directly in the Streamlit sidebar at runtime).*

---

## 8. How to Run the Application

Activate the virtual environment and launch Streamlit:

### Windows (PowerShell)
```powershell
.venv\Scripts\activate
streamlit run email_generator.py
```

### Windows (CMD)
```cmd
.venv\Scripts\activate.bat
streamlit run email_generator.py
```

### Linux / macOS
```bash
source .venv/bin/activate
streamlit run email_generator.py
```

Streamlit will launch locally at `http://localhost:8501`.

---

## 9. Example Input & Output

### Input
- **Topic**: *"Request a 3-day extension on the Q3 Financial Audit Report due to delayed invoices from the APAC vendor."*
- **Tone**: *"Professional"*
- **Recipient**: *"Finance Director"*

### Structured Output Rendered in UI

| Field | Output |
| :--- | :--- |
| **Subject** | `Request for Extension: Q3 Financial Audit Report Submission` |
| **Tone** | `Professional` |
| **Read Time** | `~45 seconds` |
| **Formality Score** | `8 / 10` |
| **Body** | *Dear Director Carter,<br><br>I am writing to formally request a short three-day extension for the submission of our Q3 Financial Audit Report...<br><br>Sincerely,<br>[Your Name]* |

---

## 10. Key Learning Outcomes

1. **Structured Data over Raw Text**: Learned why enterprise LLM applications must enforce schemas at the model boundary.
2. **LangChain Expression Language (LCEL)**: Mastered pipeline composition using `prompt | llm.with_structured_output(Schema)` over legacy `LLMChain`.
3. **Pydantic Validation Guardrails**: Enforced runtime guarantees such as non-empty text fields and strict boundary limits (`ge=1, le=10`).
4. **Clean UI Architecture**: Decoupled the Streamlit presentation layer from prompt formatting and API invocation.
