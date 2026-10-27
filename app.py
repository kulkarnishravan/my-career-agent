import os
import re
import json
import html
import threading
from datetime import datetime
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel
from markdown_it import MarkdownIt
import pypdf
import gradio as gr
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, ReplyTo

# ------------------------------------------------------------------------------
# 1. Environment & Client Setup
# ------------------------------------------------------------------------------
# Load environment variables (.env file)
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file or environment variables.")

client = OpenAI(api_key=api_key)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

INDEX_DIR = Path(__file__).parent / "faiss_index"
DATA_DIR = Path(__file__).parent / "data"
PDF_FILE = DATA_DIR / "Shravan Resume Rag feed.pdf"
RESUME_DOWNLOAD_FILE = DATA_DIR / "Shravan Resume 2026.pdf"
IMAGE_FILE = DATA_DIR / "image.png"
MAX_QUERY_CHARS = 250

if IMAGE_FILE.exists():
    import base64
    AVATAR_B64 = base64.b64encode(IMAGE_FILE.read_bytes()).decode("utf-8")
else:
    AVATAR_B64 = ""

HERO_BANNER_HTML = f"""
<div class="hero-banner">
    <div class="hero-avatar-wrap">
        <img class="hero-avatar" src="data:image/png;base64,{AVATAR_B64}" alt="Shravan A. Kulkarni" />
        <span class="avatar-status-beacon" title="System Online & Active"></span>
    </div>
    <div class="hero-content">
        <div class="hero-title-row">
            <h1 class="hero-name">Shravan A. Kulkarni</h1>
            <span class="hero-tagline">Career & Resume AI Assistant</span>
        </div>
        <div class="hero-pills">
            <div class="hero-pill status-pill">
                <span class="pulse-green-dot"></span>
                <span class="status-pill-text">System Online</span>
            </div>
            <div class="hero-pill open-to-work-pill">
                <span class="pill-icon">💼</span>
                <span>Open to Work</span>
            </div>
            <div class="hero-pill agentic-pill">
                <span class="pill-icon">🤖</span>
                <span>Agentic AI Developer</span>
            </div>
            <div class="hero-pill ui-pill">
                <span class="pill-icon">💻</span>
                <span>Senior UI Developer</span>
            </div>
            <div class="hero-pill agile-pill">
                <span class="pill-icon">⚡</span>
                <span>Agile / SDLC</span>
            </div>
        </div>
    </div>
</div>
"""

# JD Match limits & retrieval tuning
MIN_JD_CHARS = 200
MAX_JD_CHARS = 8000
MAX_JD_LINE_QUERIES = 25
JD_MATCH_MIN_COSINE = 0.40  # unrelated chunks score ~0.1-0.3, relevant ones 0.4+

# Candidate contact details for feedback & direct queries
contact_details_env = os.getenv("CONTACT_DETAILS", "").strip()
phone_match = re.search(r"Phone:\s*([^\s|]+)", contact_details_env)
email_match = re.search(r"Email:\s*([^\s|]+)", contact_details_env)

CANDIDATE_PHONE = os.getenv("CANDIDATE_PHONE") or (phone_match.group(1) if phone_match else "")
CANDIDATE_EMAIL = os.getenv("CANDIDATE_EMAIL") or (email_match.group(1) if email_match else "shravankulkarni.a@gmail.com")

# Only mention the phone number when one is configured via the environment
contact_options = ([f"call him on his number {CANDIDATE_PHONE}"] if CANDIDATE_PHONE else []) + [f"email to {CANDIDATE_EMAIL}"]

NO_INFO_FALLBACK_MESSAGE = (
    "I could not find any relevant information from the documents shared with me. "
    "Could you please leave it as a feedback to Mr. Shravan? "
    f"Feel free to {' or '.join(contact_options)}."
)


# ------------------------------------------------------------------------------
# 2. Vector Store Initialization (Load or Auto-Build)
# ------------------------------------------------------------------------------
def get_or_create_vector_store(force_rebuild: bool = False):
    """Loads existing FAISS index or creates/rebuilds if the PDF has been updated."""
    index_file = INDEX_DIR / "index.faiss"

    # Check if index exists and is up to date with the PDF file
    if not force_rebuild and INDEX_DIR.exists() and index_file.exists():
        if PDF_FILE.exists() and PDF_FILE.stat().st_mtime > index_file.stat().st_mtime:
            print(f"PDF resume ({PDF_FILE.name}) was updated after index creation. Auto-rebuilding FAISS index...")
        else:
            print(f"Loading existing FAISS index from {INDEX_DIR}...")
            return FAISS.load_local(
                str(INDEX_DIR),
                embeddings,
                allow_dangerous_deserialization=True
            )

    print(f"Building/rebuilding FAISS index from {PDF_FILE}...")
    if not PDF_FILE.exists():
        raise FileNotFoundError(f"Resume PDF not found at {PDF_FILE}")

    # Extract text
    reader = pypdf.PdfReader(PDF_FILE)
    full_text = "\n".join([page.extract_text() or "" for page in reader.pages])

    # Two-level chunking
    section_pattern = r"(?:\r?\n[ \t]*){4,}"
    sections = [s.strip() for s in re.split(section_pattern, full_text) if s.strip()]

    bullet_regex = r"(?:^|\n)[ \t]*[●•\uf0b7\-\*][ \t]*"
    chunks = []
    for sec in sections:
        bullet_items = [b.strip() for b in re.split(bullet_regex, sec) if b.strip()]
        if len(bullet_items) > 1:
            chunks.extend(bullet_items)
        elif bullet_items:
            chunks.append(bullet_items[0])

    documents = [
        Document(
            page_content=chunk,
            metadata={"source": PDF_FILE.name, "chunk_id": idx}
        )
        for idx, chunk in enumerate(chunks, start=1)
    ]

    vector_store = FAISS.from_documents(documents, embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(INDEX_DIR))
    print(f"FAISS vector store saved to {INDEX_DIR} ({len(chunks)} chunks).")
    return vector_store


vector_store = get_or_create_vector_store()


# ------------------------------------------------------------------------------
# 3. Guardrail: Job Hunt & Candidate Relevance Check
# ------------------------------------------------------------------------------
def is_job_hunt_query(query: str) -> tuple[bool, str]:
    """
    Evaluates whether the user's query is relevant to a professional job hunt,
    resume evaluation, skills, or career history.
    """
    guardrail_system_prompt = """You are a strict query classifier for a candidate's career, resume, and recruiting assistant.

Your task is to classify whether a user's question is RELEVANT to the candidate's resume, credentials, and professional profile, or OFF-TOPIC.

CLASSIFY AS RELEVANT (is_relevant: true):
- Educational background, universities, colleges, degrees, academics, graduation years, GPA/scores, or coursework.
- Work experience, employment history, company names, job titles, and roles.
- Technical skills, programming languages, frameworks, cloud platforms, tools, and libraries.
- Projects, architectural achievements, responsibilities, and deliverables.
- Certifications, licenses, training, awards, or achievements.
- Fit/suitability for specific job roles, positions, or hiring requirements.
- Resume overview, summary, or requests to review/download credentials.
- Salary Expectations, or Salary Range, relocation, hybrid/remote policy.
- Any requests for contact details, phone number, email, etc.

CLASSIFY AS IRRELEVANT (is_relevant: false):
- Private personal life (e.g., dating, marital status, relationships, religion, politics).
- Hobbies, leisure activities, social hangouts, or weekend plans.
- Off-topic trivia, jokes, coding problems unrelated to candidate evaluation, cooking recipes, weather, general chit-chat.
- Inappropriate or harmful queries.

Output strictly valid JSON with two fields:
{
  "is_relevant": true | false,
  "deflection_message": ""
}
If is_relevant is false, write a deflection that is funny but never rude, in three parts:
1. Set-up: one short sentence describing why the question is outside what you can answer (you only know Shravan's professional qualifications, education, skills, and work history).
2. Punchline: one light joke that lands the set-up and is clearly tied to the specific thing the recruiter asked (e.g. if they asked about salary, joke about negotiating; if they asked about a hobby, joke that it didn't make the shortlist; if they asked about a relationship or personal life, joke that you're a resume bot, not a matchmaker). Reference the topic of their actual question, and never mock the user.
3. Redirect: one warm sentence inviting the recruiter to reach Shravan directly, followed by the contact details below. If no contact details are provided, invite them to keep asking about Shravan's resume and never invent an email, phone number, or link.
Always refer to the candidate by name ("Shravan" or "Shravan's"), never as "the candidate" or "the candidate's"."""

    contact_details = os.getenv("CONTACT_DETAILS", "").strip()
    if contact_details:
        guardrail_system_prompt += f"\n\nCONTACT DETAILS FOR THE REDIRECT (use these exactly as written):\n{contact_details}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": guardrail_system_prompt},
            {"role": "user", "content": query}
        ]
    )

    result = json.loads(response.choices[0].message.content)
    return result["is_relevant"], result.get("deflection_message", "")


# ------------------------------------------------------------------------------
# 4. Strict Grounded Answer Generation
# ------------------------------------------------------------------------------
def generate_grounded_answer(query: str, context_chunks: list) -> str:
    """
    Sends retrieved chunks and query to OpenAI with strict anti-hallucination instructions.
    """
    if not context_chunks:
        return NO_INFO_FALLBACK_MESSAGE

    system_prompt = f"""You are a precise, factual career assistant for Shravan A. Kulkarni.

CRITICAL INSTRUCTIONS:
1. Answer the question STRICTLY using ONLY the information provided in the "Source Material" below.
2. Do NOT use any outside knowledge, assumptions, extrapolation, or imagination.
3. If the source material mentions relevant context, policies, preferences, or statements related to the topic (such as salary expectations, location preferences, work arrangements, or experience), clearly explain what is stated in the documents.
4. ONLY if the provided source material has NO relevant information whatsoever on the asked topic, you MUST state:
   "{NO_INFO_FALLBACK_MESSAGE}"
5. Do not speculate or invent numbers, qualifications, or details not present in the text."""

    context_text = "\n\n".join([
        f"[Source Chunk {doc.metadata.get('chunk_id', idx + 1)}]:\n{doc.page_content}"
        for idx, doc in enumerate(context_chunks)
    ])

    user_prompt = f"""--- SOURCE MATERIAL ---
{context_text}

--- USER QUESTION ---
{query}

Please answer the question strictly based on the source material above."""

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content


# ------------------------------------------------------------------------------
# 4b. JD Match: Retrieval, Structured Evaluation & Scoring
# ------------------------------------------------------------------------------
class Requirement(BaseModel):
    skill: str
    priority: Literal["must", "nice"]
    status: Literal["met", "partial", "missing"]
    evidence: str


class JDMatch(BaseModel):
    is_job_description: bool
    role_title: str
    requirements: list[Requirement]
    fit_reasons: list[str]
    gap_reasons: list[str]
    missing_skills: list[str]
    missing_experience: list[str]
    summary: str


NOT_A_JD_MESSAGE = (
    "That doesn't look like a job description to me. Please paste the full JD "
    "(role, responsibilities and requirements) and I'll match it against Shravan's resume."
)

WRONG_FIT_MESSAGE = (
    "### 🎯 JD Match\n\n**🔴 Wrong fit**: nothing in Shravan's resume matches this job description."
)


def validate_jd(jd_text: str) -> list[str]:
    """Returns a list of validation errors (empty if the JD is acceptable)."""
    length = len((jd_text or "").strip())
    if length < MIN_JD_CHARS:
        return [f"Please paste the full job description (at least {MIN_JD_CHARS} characters; currently {length})."]
    if length > MAX_JD_CHARS:
        return [f"The job description is too long ({length} characters). Please trim it to {MAX_JD_CHARS} characters."]
    return []


def retrieve_jd_chunks(jd_text: str) -> list:
    """
    Returns every resume chunk above JD_MATCH_MIN_COSINE for the whole JD or any of its lines.
    Line-level queries catch requirement-specific chunks that a single long-JD embedding dilutes.
    """
    lines = [l.strip() for l in re.split(r"\r?\n|[●•\uf0b7]", jd_text)]
    queries = [jd_text] + [l for l in lines if len(l) >= 15][:MAX_JD_LINE_QUERIES]

    total_chunks = vector_store.index.ntotal
    matched = {}
    for query in queries:
        for doc, sq_l2_distance in vector_store.similarity_search_with_score(query, k=total_chunks):
            # OpenAI embeddings are unit-length, so squared L2 distance = 2 - 2 * cosine
            if 1 - sq_l2_distance / 2 >= JD_MATCH_MIN_COSINE:
                matched.setdefault(doc.metadata.get("chunk_id"), doc)

    return sorted(matched.values(), key=lambda d: d.metadata.get("chunk_id", 0))


def evaluate_jd_match(jd_text: str, context_chunks: list) -> JDMatch:
    """Asks GPT-4o for a strictly grounded, structured requirement-by-requirement JD evaluation."""
    system_prompt = """You are a precise, factual recruiting analyst evaluating how well Shravan A. Kulkarni fits a job description.

CRITICAL INSTRUCTIONS:
1. First decide whether the pasted text is genuinely a job description. If not, set is_job_description to false and leave the other fields empty.
2. Extract the key requirements from the JD (skills, technologies, years/type of experience, education, domain). Mark each as "must" (required/mandatory) or "nice" (preferred/bonus/plus).
3. For each requirement, judge Shravan STRICTLY using ONLY the "Source Material" chunks:
   - "met": a chunk clearly demonstrates it.
   - "partial": a chunk shows related or adjacent experience, but not fully.
   - "missing": NO chunk supports it.
   Before marking anything "missing", scan EVERY chunk for the skill, a synonym, or a closely related term; if any chunk mentions it, it is at least "partial".
   In "evidence", provide concise supporting evidence directly from Shravan's resume (e.g. "Developed full-stack web applications using React and Node.js") or write "Not found in resume". Do NOT mention chunk numbers or chunk IDs.
4. Do NOT use outside knowledge, assumptions, or extrapolation. Never invent skills, numbers, or experience.
5. fit_reasons: short bullets on why Shravan fits. gap_reasons: short bullets on where he falls short.
6. missing_skills: short skill/tool names (1-3 words, e.g. "Kubernetes", "Go") for requirements marked "missing" or "partial". missing_experience: missing experience areas (seniority, domain, years). Never list something as missing that a chunk supports.
7. summary: 2-3 sentences giving a balanced verdict.
8. Always refer to the candidate as "Shravan", never as "the candidate"."""

    context_text = "\n\n".join([
        f"[Source Chunk {doc.metadata.get('chunk_id', idx + 1)}]:\n{doc.page_content}"
        for idx, doc in enumerate(context_chunks)
    ])

    user_prompt = f"""--- SOURCE MATERIAL ---
{context_text}

--- JOB DESCRIPTION ---
{jd_text}

Evaluate Shravan's fit for this job description strictly based on the source material above."""

    response = client.chat.completions.parse(
        model="gpt-4o",
        temperature=0.0,
        response_format=JDMatch,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.parsed


def compute_match_score(requirements: list[Requirement]) -> int:
    """Deterministic score: must-haves weigh 2x, partial matches earn half credit."""
    weights = {"must": 2, "nice": 1}
    credit = {"met": 1.0, "partial": 0.5, "missing": 0.0}
    max_points = sum(weights[r.priority] for r in requirements)
    if not max_points:
        return 0
    earned = sum(weights[r.priority] * credit[r.status] for r in requirements)
    return round(earned / max_points * 100)


def clean_evidence(evidence: str) -> str:
    """Strips chunk markers (e.g. 'Chunk 12:', '[Source Chunk 3]:', '(Chunk 5)') from evidence text."""
    if not evidence:
        return ""
    cleaned = re.sub(
        r"^(?:\[?(?:Source\s+)?Chunk\s*\d+\]?[:\s\-–—]*|\((?:Source\s+)?Chunk\s*\d+\)[:\s\-–—]*)",
        "",
        evidence,
        flags=re.IGNORECASE
    )
    cleaned = re.sub(r"\s*[\(\[](?:Source\s+)?Chunk\s*\d+[\)\]]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*[:\-–—]\s*", "", cleaned)
    return cleaned.strip()


def format_jd_match_markdown(result: JDMatch, score: int) -> str:
    """Renders the JD Match result as a chat-friendly markdown answer."""
    if score >= 75:
        band = "🟢 Strong fit"
    elif score >= 50:
        band = "🟡 Moderate fit"
    else:
        band = "🔴 Weak fit"

    def bullets(items):
        return "\n".join(f"- {item}" for item in items)

    def cell(text):
        return text.replace("|", "\\|").replace("\n", " ")

    parts = [
        f"### 🎯 JD Match: {result.role_title or 'Job Description'}",
        f"**Match Score: {score}/100** {band}",
        f"> {result.summary}",
    ]
    if result.fit_reasons:
        parts.append(f"**✅ Why Shravan fits**\n{bullets(result.fit_reasons)}")
    if result.gap_reasons:
        parts.append(f"**⚠️ Where it falls short**\n{bullets(result.gap_reasons)}")
    if result.missing_skills:
        parts.append("**🧩 Missing skills:** " + " · ".join(f"`{s}`" for s in result.missing_skills))
    if result.missing_experience:
        parts.append(f"**📉 Missing experience**\n{bullets(result.missing_experience)}")
    if result.requirements:
        status_icon = {"met": "✅ Met", "partial": "🟡 Partial", "missing": "❌ Missing"}
        rows = "\n".join(
            f"| {cell(r.skill)} | {'Must' if r.priority == 'must' else 'Nice'} | {status_icon[r.status]} | {cell(clean_evidence(r.evidence))} |"
            for r in result.requirements
        )
        parts.append(
            "**Requirement breakdown**\n\n"
            "| Requirement | Priority | Status | Evidence |\n|---|---|---|---|\n" + rows
        )
    return "\n\n".join(parts)


# ------------------------------------------------------------------------------
# 5. Dynamic Stylish Pipeline Status (Zero Numbers)
# ------------------------------------------------------------------------------
def render_pipeline_status(step: str, title: str, subtitle: str) -> str:
    """
    Renders a stylish color-coded pipeline status card without any numbers.
    Colors dynamically shift based on the pipeline stage:
      - idle: Soft Slate
      - guardrail: Amber / Gold
      - retrieval: Cyan / Sky Blue
      - synthesis: Electric Violet / Purple
      - complete: Mint / Emerald Green
      - deflected: Rose / Coral
      - download: Royal Blue
    """
    schemes = {
        "idle": {
            "bg": "rgba(248, 250, 252, 0.95)",
            "border": "rgba(226, 232, 240, 0.95)",
            "color": "#475569",
            "beacon": "#94a3b8",
            "active_idx": 0,
            "shimmer": False,
        },
        "guardrail": {
            "bg": "rgba(255, 251, 235, 0.95)",
            "border": "rgba(253, 230, 138, 0.95)",
            "color": "#b45309",
            "beacon": "#f59e0b",
            "active_idx": 1,
            "shimmer": True,
        },
        "retrieval": {
            "bg": "rgba(240, 249, 255, 0.95)",
            "border": "rgba(186, 230, 253, 0.95)",
            "color": "#0369a1",
            "beacon": "#0284c7",
            "active_idx": 2,
            "shimmer": True,
        },
        "synthesis": {
            "bg": "rgba(245, 243, 255, 0.95)",
            "border": "rgba(221, 214, 254, 0.95)",
            "color": "#6d28d9",
            "beacon": "#8b5cf6",
            "active_idx": 3,
            "shimmer": True,
        },
        "complete": {
            "bg": "rgba(236, 253, 245, 0.95)",
            "border": "rgba(167, 243, 208, 0.95)",
            "color": "#047857",
            "beacon": "#10b981",
            "active_idx": 4,
            "shimmer": False,
        },
        "deflected": {
            "bg": "rgba(255, 241, 242, 0.95)",
            "border": "rgba(254, 205, 211, 0.95)",
            "color": "#be123c",
            "beacon": "#f43f5e",
            "active_idx": -1,
            "shimmer": False,
        },
        "download": {
            "bg": "rgba(239, 246, 255, 0.95)",
            "border": "rgba(191, 219, 254, 0.95)",
            "color": "#1d4ed8",
            "beacon": "#3b82f6",
            "active_idx": -1,
            "shimmer": False,
        },
    }

    cfg = schemes.get(step, schemes["idle"])

    stages = [
        ("Guardrail", "🛡️"),
        ("Retrieval", "🔍"),
        ("Synthesis", "🤖"),
        ("Grounded", "✨")
    ]

    pills_html = []
    active_idx = cfg["active_idx"]

    if active_idx == -1:
        badge_text = "🛡️ Guardrail Deflection" if step == "deflected" else "📥 PDF Delivered"
        pills_html.append(
            f'<div class="pipeline-pill" style="background:{cfg["beacon"]}20; color:{cfg["color"]}; flex: 1; border: 1px solid {cfg["border"]};">'
            f'{badge_text}'
            f'</div>'
        )
    else:
        for i, (name, icon) in enumerate(stages, start=1):
            if i < active_idx:
                pills_html.append(
                    f'<div class="pipeline-pill" style="background:rgba(16, 185, 129, 0.12); color:#059669; border: 1px solid rgba(16, 185, 129, 0.25);">'
                    f'✓ {name}'
                    f'</div>'
                )
            elif i == active_idx:
                pills_html.append(
                    f'<div class="pipeline-pill" style="background:{cfg["beacon"]}; color:#ffffff; font-weight:700; box-shadow:0 2px 10px {cfg["beacon"]}66;">'
                    f'{icon} {name}'
                    f'</div>'
                )
            else:
                pills_html.append(
                    f'<div class="pipeline-pill" style="background:rgba(148, 163, 184, 0.08); color:#94a3b8; border: 1px dashed rgba(203, 213, 225, 0.7);">'
                    f'{icon} {name}'
                    f'</div>'
                )

    shimmer_bar = f'<div class="pipeline-shimmer" style="color:{cfg["beacon"]};"></div>' if cfg["shimmer"] else ""

    return f"""
<div class="status-card" style="background:{cfg['bg']}; border-color:{cfg['border']}; color:{cfg['color']};">
    <div class="status-card-header">
        <div class="status-beacon" style="background:{cfg['beacon']};"></div>
        <div class="status-title-group">
            <h4 class="status-headline" style="color:{cfg['color']};">{title}</h4>
            <p class="status-subtext" style="color:{cfg['color']};">{subtitle}</p>
        </div>
    </div>
    <div class="pipeline-track">
        {''.join(pills_html)}
    </div>
    {shimmer_bar}
</div>
"""


def render_loader(label: str) -> str:
    """Minimal glowing blue→violet progress bar shown under the user's bubble while the answer loads."""
    return (
        '<div class="rag-loader">'
        '<div class="rag-loader-track"><div class="rag-loader-glow"></div></div>'
        f'<div class="rag-loader-label">{label}</div>'
        '</div>'
    )


# ------------------------------------------------------------------------------
# 6. Chatbot Handler for Gradio
# ------------------------------------------------------------------------------
def format_sources(docs: list) -> str:
    """Markdown listing of retrieved chunks for the Developer Mode sources panel."""
    return "\n\n---\n\n".join([
        f"**Source Chunk {doc.metadata.get('chunk_id', idx + 1)}** *(File: `{doc.metadata.get('source', PDF_FILE.name)}`)*:\n\n{doc.page_content}"
        for idx, doc in enumerate(docs)
    ])


def chat_handler(user_message, history, top_k=4):
    if history is None:
        history = []

    default_sources = "*Retrieved chunks will appear here after asking a relevant question.*"

    if not user_message or not user_message.strip():
        yield history, render_pipeline_status("idle", "Input Required", "Please enter a question to evaluate."), default_sources
        return

    # 0. Check if user is requesting resume download
    download_triggers = [
        "download resume", "download cv", "get resume", "get cv",
        "pdf resume", "resume pdf", "give me the resume", "share resume", 
        "view resume", "download the resume"
    ]
    user_lower = user_message.strip().lower()
    if any(trigger in user_lower for trigger in download_triggers):
        file_data = gr.FileData(path=str(RESUME_DOWNLOAD_FILE))
        history = history + [
            {"role": "user", "content": user_message},
            {
                "role": "assistant",
                "content": f"📄 **Official Resume of Shravan A. Kulkarni**\n\nYou can click the file below to view or download the complete PDF resume directly:"
            },
            {"role": "assistant", "content": file_data}
        ]
        status_html = render_pipeline_status("download", "Resume Delivered", "Official PDF resume served directly in conversation.")
        yield history, status_html, "*No chunks retrieved (resume download served).*"
        return

    # Show the user's bubble right away, with the loader directly beneath it
    base_history = history + [{"role": "user", "content": user_message}]

    def pending(label):
        return base_history + [{"role": "assistant", "content": render_loader(label)}]

    # 1. Guardrail Check (Yield active state so changing color shows immediately)
    status_html = render_pipeline_status("guardrail", "Guardrail Verification", "Evaluating relevance to candidate qualifications...")
    yield pending("Understanding your question"), status_html, default_sources

    is_relevant, deflection_msg = is_job_hunt_query(user_message)

    if not is_relevant:
        # Off-topic: Deflect politely, do NOT retrieve or send chunks
        history = base_history + [{"role": "assistant", "content": deflection_msg}]
        status_html = render_pipeline_status("deflected", "Guardrail Deflected", "Query classified as off-topic or personal.")
        yield history, status_html, "*No chunks retrieved (guardrail triggered - off-topic query).*"
        return

    # 2. Retrieve Chunks
    status_html = render_pipeline_status("retrieval", "Context Retrieval", "Fetching relevant chunks from FAISS vector store...")
    yield pending("Searching the resume"), status_html, "*Retrieving relevant chunks...*"

    retrieved_docs = vector_store.similarity_search(user_message, k=int(top_k))

    if not retrieved_docs:
        history = base_history + [{"role": "assistant", "content": NO_INFO_FALLBACK_MESSAGE}]
        status_html = render_pipeline_status("complete", "Information Not Found", "No relevant chunks retrieved from vector database.")
        yield history, status_html, "*No chunks retrieved from vector database.*"
        return

    formatted_sources = format_sources(retrieved_docs)

    # 3. Grounded Answer Generation
    status_html = render_pipeline_status("synthesis", "Grounded Synthesis", "Synthesizing answer strictly under source constraints...")
    yield pending("Composing answer"), status_html, formatted_sources

    answer = generate_grounded_answer(user_message, retrieved_docs)

    history = base_history + [{"role": "assistant", "content": answer}]
    status_html = render_pipeline_status("complete", "Complete & Grounded", "Answer verified strictly within source material.")
    yield history, status_html, formatted_sources


def jd_match_handler(jd_text, history):
    """Retrieves every matching chunk for the JD, evaluates fit, and yields (history, status, sources)."""
    history = history or []
    jd_text = jd_text.strip()
    base_history = history + [{"role": "user", "content": f"🎯 JD Match requested ({len(jd_text)} chars)"}]

    def pending(label):
        return base_history + [{"role": "assistant", "content": render_loader(label)}]

    try:
        # 1. Retrieve all chunks above the relevance threshold
        status_html = render_pipeline_status("retrieval", "JD Context Retrieval", "Fetching every resume chunk relevant to the JD...")
        yield pending("Searching the resume"), status_html, "*Retrieving relevant chunks...*"

        retrieved_docs = retrieve_jd_chunks(jd_text)
        if not retrieved_docs:
            send_jd_match_email(jd_text, WRONG_FIT_MESSAGE, "JD Match: Wrong fit")
            history = base_history + [{"role": "assistant", "content": WRONG_FIT_MESSAGE}]
            status_html = render_pipeline_status("complete", "Wrong Fit", "No resume chunks matched the job description.")
            yield history, status_html, "*No chunks matched the job description.*"
            return

        formatted_sources = format_sources(retrieved_docs)

        # 2. Structured, grounded evaluation + deterministic score
        status_html = render_pipeline_status("synthesis", "JD Match Synthesis", f"Scoring the JD against {len(retrieved_docs)} retrieved chunks...")
        yield pending("Matching JD to resume"), status_html, formatted_sources

        result = evaluate_jd_match(jd_text, retrieved_docs)
        if not result.is_job_description:
            history = base_history + [{"role": "assistant", "content": NOT_A_JD_MESSAGE}]
            status_html = render_pipeline_status("deflected", "Not a Job Description", "Pasted text was not recognised as a JD.")
            yield history, status_html, formatted_sources
            return

        score = compute_match_score(result.requirements)
        answer = format_jd_match_markdown(result, score)
        send_jd_match_email(jd_text, answer, f"JD Match: {result.role_title or 'Job Description'} ({score}/100)")
    except Exception as exc:
        print(f"[JD MATCH] Evaluation failed: {type(exc).__name__}: {exc}")
        history = base_history + [{
            "role": "assistant",
            "content": "Sorry, I couldn't complete the JD match right now. Please try again in a moment."
        }]
        status_html = render_pipeline_status("deflected", "JD Match Failed", "An error occurred while evaluating the JD.")
        yield history, status_html, "*JD match failed.*"
        return

    history = base_history + [{"role": "assistant", "content": answer}]
    status_html = render_pipeline_status("complete", "JD Match Complete", "JD scored strictly against retrieved resume chunks.")
    yield history, status_html, formatted_sources


# ------------------------------------------------------------------------------
# 6b. Feedback Validation
# ------------------------------------------------------------------------------
PHONE_PATTERN = re.compile(r"^\+?[\d\s\-()]{7,20}$")
EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def validate_feedback(name: str, contact: str, email: str, feedback: str) -> list[str]:
    """Returns a list of validation errors (empty if the feedback form is valid)."""
    name, contact, email, feedback = (v.strip() if v else "" for v in (name, contact, email, feedback))
    errors = []
    if not name:
        errors.append("Please enter your name.")
    if not contact and not email:
        errors.append("Please provide a contact number or an e-mail address.")
    if contact:
        digit_count = len(re.sub(r"\D", "", contact))
        if not PHONE_PATTERN.match(contact) or not 7 <= digit_count <= 15:
            errors.append("Please enter a valid contact number (7–15 digits; +, spaces, dashes, brackets allowed).")
    if email and not EMAIL_PATTERN.match(email):
        errors.append("Please enter a valid e-mail address (e.g. you@company.com).")
    if not feedback:
        errors.append("Please enter your feedback.")
    return errors


# ------------------------------------------------------------------------------
# 6c. Feedback Delivery via SendGrid
# ------------------------------------------------------------------------------
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "").strip()
# The sender must be a verified Single Sender (or domain) in SendGrid
FEEDBACK_FROM_EMAIL = os.getenv("FEEDBACK_FROM_EMAIL", "").strip() or CANDIDATE_EMAIL
FEEDBACK_TO_EMAIL = os.getenv("FEEDBACK_TO_EMAIL", "").strip() or CANDIDATE_EMAIL


def send_feedback_email(name: str, contact: str, email: str, feedback: str) -> tuple[bool, str]:
    """Emails the feedback to Shravan via SendGrid. Returns (success, error_detail)."""
    if not SENDGRID_API_KEY:
        return False, "SENDGRID_API_KEY is not configured"

    rows = [("Name", name), ("Contact", contact or "—"), ("E-mail", email or "—")]
    rows_html = "".join(
        f'<tr><td style="padding:4px 12px 4px 0;color:#64748b;"><b>{label}</b></td><td>{html.escape(value)}</td></tr>'
        for label, value in rows
    )
    body_html = (
        f'<div style="font-family:Arial,sans-serif;font-size:14px;color:#0f172a;">'
        f'<h3 style="color:#4338ca;margin:0 0 12px 0;">New feedback from the Resume Chatbot</h3>'
        f'<table>{rows_html}</table>'
        f'<p style="margin:16px 0 4px 0;"><b>Feedback:</b></p>'
        f'<div style="white-space:pre-wrap;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px;">'
        f'{html.escape(feedback)}</div></div>'
    )

    reply_to = ReplyTo(email, name) if email else None
    return send_email_to_shravan(f"Resume Chatbot feedback from {name}", body_html, reply_to)


def send_email_to_shravan(subject: str, body_html: str, reply_to: ReplyTo | None = None) -> tuple[bool, str]:
    """Sends an HTML e-mail to Shravan via SendGrid. Returns (success, error_detail)."""
    if not SENDGRID_API_KEY:
        return False, "SENDGRID_API_KEY is not configured"

    message = Mail(
        from_email=FEEDBACK_FROM_EMAIL,
        to_emails=FEEDBACK_TO_EMAIL,
        subject=subject,
        html_content=body_html,
    )
    if reply_to:
        message.reply_to = reply_to

    try:
        response = SendGridAPIClient(SENDGRID_API_KEY).send(message)
    except Exception as exc:  # SendGrid raises HTTPError subclasses for 4xx/5xx
        return False, f"{type(exc).__name__}: {getattr(exc, 'body', exc)}"
    if 200 <= response.status_code < 300:
        return True, ""
    return False, f"HTTP {response.status_code}: {response.body}"


# Raw HTML disabled so pasted JD/LLM text can't inject markup into the e-mail
email_markdown = MarkdownIt("commonmark", {"html": False}).enable("table")
EMAIL_TABLE_STYLES = {
    "<table>": '<table style="border-collapse:collapse;width:100%;font-size:13px;">',
    "<th>": '<th style="border:1px solid #e2e8f0;background:#f1f5f9;padding:6px;text-align:left;">',
    "<td>": '<td style="border:1px solid #e2e8f0;padding:6px;vertical-align:top;">',
}


def send_jd_match_email(jd_text: str, result_markdown: str, subject: str) -> None:
    """E-mails the JD Match result plus the original JD to Shravan in a background thread."""
    result_html = email_markdown.render(result_markdown)
    for tag, styled in EMAIL_TABLE_STYLES.items():
        result_html = result_html.replace(tag, styled)

    body_html = (
        f'<div style="font-family:Arial,sans-serif;font-size:14px;color:#0f172a;">'
        f'<h3 style="color:#4338ca;margin:0 0 12px 0;">New JD Match from the Resume Chatbot</h3>'
        f'<p style="color:#64748b;margin:0 0 12px 0;">Run at {datetime.now().strftime("%d %b %Y, %H:%M")}</p>'
        f'{result_html}'
        f'<p style="margin:20px 0 4px 0;"><b>Job Description:</b></p>'
        f'<div style="white-space:pre-wrap;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px;">'
        f'{html.escape(jd_text)}</div></div>'
    )

    def _send():
        sent, error_detail = send_email_to_shravan(subject, body_html)
        if not sent:
            print(f"[JD MATCH] E-mail delivery failed: {error_detail}")

    threading.Thread(target=_send, daemon=True).start()


# ------------------------------------------------------------------------------
# 6. Gradio UI Construction
# ------------------------------------------------------------------------------
light_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate",
    text_size=gr.themes.sizes.text_sm,
    spacing_size=gr.themes.sizes.spacing_sm,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "-apple-system", "sans-serif"],
).set(
    body_background_fill="transparent",
    block_background_fill="rgba(255, 255, 255, 0.75)",
    block_border_width="1px",
    block_border_color="rgba(226, 232, 240, 0.8)",
    block_shadow="0 8px 24px -4px rgba(15, 23, 42, 0.05)",
    button_primary_background_fill="linear-gradient(135deg, #4f46e5 0%, #0ea5e9 100%)",
    button_primary_text_color="#ffffff",
    input_background_fill="rgba(255, 255, 255, 0.95)",
    chatbot_text_size="0.72rem",
)

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Outfit:wght@300;400;500;600;700&display=swap');

/* ── Luminous Ambient Mesh Gradient Background (Not Generic Gradio White/Gray) ── */
.gradio-container {
    background: radial-gradient(at 0% 0%, rgba(224, 242, 254, 0.85) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(238, 242, 255, 0.9) 0px, transparent 50%),
                radial-gradient(at 50% 100%, rgba(240, 249, 255, 0.8) 0px, transparent 50%),
                linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
    min-height: 100vh;
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    color: #0f172a !important;
}

#app-container {
    max-width: 1240px;
    margin: auto;
    padding: 16px;
}

/* ── Frosted Crystal Glass Cards ── */
.block, .form, .gradio-row, .gr-panel, .gr-box {
    background: rgba(255, 255, 255, 0.72) !important;
    backdrop-filter: blur(18px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(18px) saturate(180%) !important;
    border: 1px solid rgba(255, 255, 255, 0.9) !important;
    outline: 1px solid rgba(226, 232, 240, 0.7) !important;
    border-radius: 18px !important;
    box-shadow: 0 10px 30px -5px rgba(15, 23, 42, 0.05), 0 4px 6px -2px rgba(15, 23, 42, 0.02) !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.block:hover {
    outline-color: rgba(99, 102, 241, 0.3) !important;
    box-shadow: 0 14px 36px -6px rgba(79, 70, 229, 0.08), 0 4px 10px rgba(15, 23, 42, 0.03) !important;
}

/* ── Chatbot Glass Canvas & Sleek Compact Conversation Typography ── */
:root,
.gradio-container,
.chatbot,
#main-chatbot,
[data-testid="chatbot"] {
    --chatbot-text-size: 0.72rem !important;
}

#main-chatbot,
.chatbot {
    background: rgba(255, 255, 255, 0.65) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(226, 232, 240, 0.85) !important;
    border-radius: 18px !important;
    box-shadow: 0 8px 30px rgba(15, 23, 42, 0.04) !important;
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

/* ── Clean Single Speech Bubble Layout (Eliminates Box-in-Box Nesting) ── */
#main-chatbot [class*="bubble-wrap"],
.chatbot [class*="bubble-wrap"],
#main-chatbot .bubble-wrap,
.chatbot .bubble-wrap,
#main-chatbot [class*="panel-wrap"],
.chatbot [class*="panel-wrap"] {
    padding: 4px 6px !important;
    padding-top: 4px !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Strip borders, backgrounds, and shadows from all outer rows and wrappers (Removes outermost box) */
#main-chatbot .bubble,
.chatbot .bubble,
#main-chatbot [class*="bubble"],
.chatbot [class*="bubble"],
#main-chatbot .user-row,
.chatbot .user-row,
#main-chatbot .bot-row,
.chatbot .bot-row,
#main-chatbot [class*="user-row"],
.chatbot [class*="user-row"],
#main-chatbot [class*="bot-row"],
.chatbot [class*="bot-row"],
#main-chatbot .message-row,
.chatbot .message-row,
#main-chatbot [class*="message-row"],
.chatbot [class*="message-row"],
#main-chatbot .message-bubble-border,
.chatbot .message-bubble-border,
#main-chatbot [class*="message-bubble-border"],
.chatbot [class*="message-bubble-border"],
#main-chatbot .panel,
.chatbot .panel,
#main-chatbot [class*="panel"],
.chatbot [class*="panel"],
#main-chatbot .bot,
.chatbot .bot,
#main-chatbot .user,
.chatbot .user,
#main-chatbot .message,
.chatbot .message,
#main-chatbot .bot.message,
.chatbot .bot.message,
#main-chatbot .user.message,
.chatbot .user.message,
#main-chatbot .message.bot,
.chatbot .message.bot,
#main-chatbot .message.user,
.chatbot .message.user,
#main-chatbot [class*="message"],
.chatbot [class*="message"],
#main-chatbot [class*="bot"],
.chatbot [class*="bot"],
#main-chatbot [class*="user"],
.chatbot [class*="user"] {
    background: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Inner content elements must have no background or borders */
#main-chatbot .message-content,
.chatbot .message-content,
#main-chatbot [class*="message-content"],
.chatbot [class*="message-content"],
#main-chatbot .prose,
.chatbot .prose,
#main-chatbot [class*="prose"],
.chatbot [class*="prose"],
#main-chatbot .flex-wrap,
.chatbot .flex-wrap,
#main-chatbot [class*="flex-wrap"],
.chatbot [class*="flex-wrap"] {
    background: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    min-height: 0 !important;
    height: auto !important;
}

/* ── SINGLE ELEGANT BOT RESPONSE BOX (THE ONLY CARD) ── */
#main-chatbot [data-testid="bot"],
.chatbot [data-testid="bot"] {
    background: rgba(255, 255, 255, 0.95) !important;
    color: #0f172a !important;
    border: 1px solid rgba(226, 232, 240, 0.95) !important;
    outline: none !important;
    border-radius: 12px 12px 12px 2px !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
    font-weight: 400 !important;
    padding: 6px 12px !important;
    max-width: 90% !important;
    width: fit-content !important;
    margin: 2px 0 !important;
}

/* ── SINGLE ELEGANT USER RESPONSE BOX (THE ONLY CARD) ── */
#main-chatbot [data-testid="user"],
.chatbot [data-testid="user"] {
    background: linear-gradient(135deg, #4f46e5 0%, #0ea5e9 100%) !important;
    color: #ffffff !important;
    border: none !important;
    outline: none !important;
    border-radius: 12px 12px 2px 12px !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.22) !important;
    font-weight: 450 !important;
    padding: 6px 12px !important;
    max-width: 90% !important;
    width: fit-content !important;
    margin: 2px 0 !important;
}

#main-chatbot [data-testid="user"] p,
.chatbot [data-testid="user"] p,
#main-chatbot [data-testid="user"] *,
.chatbot [data-testid="user"] * {
    color: #ffffff !important;
}

#main-chatbot [data-testid="bot"] p,
.chatbot [data-testid="bot"] p,
#main-chatbot [data-testid="bot"] *,
.chatbot [data-testid="bot"] * {
    color: #0f172a !important;
}

/* Comprehensive font size reduction for conversation area */
#main-chatbot,
#main-chatbot *,
.chatbot,
.chatbot *,
#main-chatbot [class*="prose"] *,
.chatbot [class*="prose"] *,
#main-chatbot [class*="message"] *,
.chatbot [class*="message"] * {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    font-size: 0.72rem !important;
    line-height: 1.40 !important;
}

#main-chatbot p,
.chatbot p,
#main-chatbot [class*="prose"] p,
.chatbot [class*="prose"] p,
#main-chatbot li,
.chatbot li,
#main-chatbot span,
.chatbot span {
    margin: 1px 0 !important;
    font-size: 0.72rem !important;
    line-height: 1.36 !important;
}

#main-chatbot .prose h1, .chatbot .message h1 { font-size: 0.88rem !important; font-weight: 700 !important; margin: 3px 0 !important; }
#main-chatbot .prose h2, .chatbot .message h2 { font-size: 0.82rem !important; font-weight: 600 !important; margin: 2px 0 !important; }
#main-chatbot .prose h3, .chatbot .message h3 { font-size: 0.78rem !important; font-weight: 600 !important; margin: 2px 0 !important; }
#main-chatbot .prose ul, #main-chatbot .prose ol, .chatbot .message ul, .chatbot .message ol {
    margin: 2px 0 !important;
    padding-left: 14px !important;
}
#main-chatbot .prose code, .chatbot .message code {
    font-size: 0.68rem !important;
    padding: 1px 4px !important;
}

/* ── Clean Glass Input & Textarea ── */
textarea, input[type="text"] {
    background: rgba(255, 255, 255, 0.92) !important;
    border: 1px solid rgba(203, 213, 225, 0.8) !important;
    border-radius: 12px !important;
    color: #0f172a !important;
    box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.03) !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: #6366f1 !important;
    outline: none !important;
    background: #ffffff !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.18), 0 4px 14px rgba(99, 102, 241, 0.08) !important;
}

/* ── Tactile Pill Buttons ── */
.quick-options {
    margin-top: 6px;
    margin-bottom: 10px;
    gap: 8px;
    flex-wrap: wrap;
}

/* Recruiter Mode: quick options stacked vertically beside the conversation */
.quick-options-vertical {
    gap: 10px;
}

/* ── Proportional 8:4 Column Alignment Between Header and Main Content ── */
.header-main-col,
.conversation-col {
    flex: 8 1 0% !important;
    min-width: 0 !important;
}

.conversation-col {
    gap: 0 !important;
}

.conversation-col > *,
.conversation-col .chatbot,
.chatbot {
    margin-bottom: 0 !important;
}

.header-side-col,
.side-panel {
    flex: 4 1 0% !important;
    min-width: 260px !important;
    overflow: hidden;
}

/* Let inspector content reflow inside the narrower panel */
.side-panel .pipeline-track { flex-wrap: wrap; }
.side-panel .pipeline-pill { flex: 1 1 45%; }
.side-panel .sources-box { overflow-wrap: anywhere; }

.quick-options-vertical button, .quick-options-vertical .gr-button, .quick-options-vertical a {
    width: 100% !important;
    justify-content: center;
}

.quick-options button, .quick-options .gr-button, .quick-options-vertical button, .quick-options-vertical .gr-button {
    border-radius: 9999px !important;
    padding: 7px 18px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    background: rgba(255, 255, 255, 0.92) !important;
    border: 1px solid rgba(203, 213, 225, 0.85) !important;
    color: #334155 !important;
    cursor: pointer !important;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04), 0 1px 2px rgba(15, 23, 42, 0.02) !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.quick-options button.primary,
.quick-options .gr-button-primary,
.quick-options a.primary,
.quick-options-vertical button.primary,
.quick-options-vertical .gr-button-primary,
.quick-options-vertical a.primary {
    background: linear-gradient(135deg, #4f46e5 0%, #0ea5e9 100%) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3), 0 1px 2px rgba(0, 0, 0, 0.05) !important;
}

.quick-options button:hover, .quick-options .gr-button:hover, .quick-options-vertical button:hover, .quick-options-vertical .gr-button:hover {
    transform: translateY(-2px) scale(1.02) !important;
    background: #ffffff !important;
    border-color: #6366f1 !important;
    color: #4f46e5 !important;
    box-shadow: 0 6px 18px rgba(99, 102, 241, 0.18), 0 2px 4px rgba(99, 102, 241, 0.08) !important;
}

.quick-options button.primary:hover,
.quick-options .gr-button-primary:hover,
.quick-options a.primary:hover,
.quick-options-vertical button.primary:hover,
.quick-options-vertical .gr-button-primary:hover,
.quick-options-vertical a.primary:hover {
    background: linear-gradient(135deg, #4338ca 0%, #0284c7 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 8px 22px rgba(79, 70, 229, 0.42), 0 2px 6px rgba(14, 165, 233, 0.25) !important;
}

.quick-options button:active, .quick-options .gr-button:active, .quick-options-vertical button:active, .quick-options-vertical .gr-button:active {
    transform: translateY(1px) scale(0.98) !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08) !important;
}

/* Quick option button while its query is running: always-lit purple LED with a breathing halo */
@keyframes quick-btn-led {
    0%, 100% { box-shadow: 0 0 4px 1px rgba(139, 92, 246, 0.85), 0 0 8px 2px rgba(139, 92, 246, 0.35); }
    50%      { box-shadow: 0 0 6px 2px rgba(167, 139, 250, 1), 0 0 16px 5px rgba(139, 92, 246, 0.55); }
}

button.quick-btn-loading {
    pointer-events: none !important;
    opacity: 1 !important;
    color: #4f46e5 !important;
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.14), 0 4px 14px rgba(99, 102, 241, 0.18) !important;
}

button.quick-btn-loading::before {
    content: '';
    display: inline-block;
    width: 8px;
    height: 8px;
    margin-right: 8px;
    vertical-align: 1px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, #ede9fe 0%, #a78bfa 40%, #7c3aed 100%);
    animation: quick-btn-led 1.4s ease-in-out infinite;
}

@media (prefers-reduced-motion: reduce) {
    button.quick-btn-loading::before {
        animation: none;
        box-shadow: 0 0 5px 1px rgba(139, 92, 246, 0.9), 0 0 10px 3px rgba(139, 92, 246, 0.4);
    }
}

/* ── Tactile Gradient Primary Button ── */
button.primary, .gr-button-primary {
    background: linear-gradient(135deg, #4f46e5 0%, #0ea5e9 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3), 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

button.primary:hover, .gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 22px rgba(79, 70, 229, 0.42), 0 2px 6px rgba(14, 165, 233, 0.25) !important;
}

button.primary:active, .gr-button-primary:active {
    transform: translateY(1px) !important;
    box-shadow: 0 2px 6px rgba(79, 70, 229, 0.3) !important;
}

/* ── Secondary / Clear Buttons ── */
button.secondary, .gr-button-secondary {
    background: rgba(255, 255, 255, 0.85) !important;
    border: 1px solid rgba(203, 213, 225, 0.85) !important;
    border-radius: 12px !important;
    color: #475569 !important;
    font-weight: 500 !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

button.secondary:hover, .gr-button-secondary:hover {
    background: #ffffff !important;
    color: #0f172a !important;
    border-color: #cbd5e1 !important;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05) !important;
}

/* ── Accordion & Sources Panel ── */
.sources-box {
    max-height: 480px;
    overflow-y: auto;
    font-size: 0.9em;
    padding-right: 4px;
    color: #334155 !important;
}

.sources-box::-webkit-scrollbar {
    width: 6px;
}
.sources-box::-webkit-scrollbar-track {
    background: rgba(241, 245, 249, 0.8);
    border-radius: 8px;
}
.sources-box::-webkit-scrollbar-thumb {
    background: rgba(148, 163, 184, 0.4);
    border-radius: 8px;
}
.sources-box::-webkit-scrollbar-thumb:hover {
    background: rgba(99, 102, 241, 0.5);
}

/* ── Modern Mode Toggle Switch & Header Alignment ── */
.header-row {
    display: flex !important;
    align-items: stretch !important;
    margin-bottom: 4px !important;
    gap: 16px !important;
}

.header-main-col,
.header-side-col {
    display: flex !important;
    flex-direction: column !important;
}

.header-main-col > .block,
.header-side-col > .block,
.header-main-col .prose {
    height: 100% !important;
    min-height: 100% !important;
    box-sizing: border-box !important;
    margin-bottom: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
}

.header-main-col .block {
    padding: 6px 12px !important;
}

/* ── Trendy Hero Banner ── */
.hero-banner {
    display: flex !important;
    align-items: center !important;
    gap: 16px !important;
    padding: 2px 4px !important;
    width: 100% !important;
    height: 100% !important;
    box-sizing: border-box !important;
}

.hero-avatar-wrap {
    position: relative !important;
    flex-shrink: 0 !important;
    width: 72px !important;
    height: 72px !important;
    border-radius: 50% !important;
    padding: 2.5px !important;
    background: linear-gradient(135deg, #6366f1 0%, #0ea5e9 50%, #10b981 100%) !important;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.25) !important;
}

.hero-avatar {
    width: 100% !important;
    height: 100% !important;
    border-radius: 50% !important;
    object-fit: cover !important;
    display: block !important;
    background: #ffffff !important;
}

.avatar-status-beacon {
    position: absolute !important;
    bottom: 2px !important;
    right: 2px !important;
    width: 13px !important;
    height: 13px !important;
    background: #10b981 !important;
    border: 2.5px solid #ffffff !important;
    border-radius: 50% !important;
    box-shadow: 0 0 8px rgba(16, 185, 129, 0.8) !important;
}

.hero-content {
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    gap: 6px !important;
    flex: 1 1 auto !important;
    min-width: 0 !important;
}

.hero-title-row {
    display: flex !important;
    align-items: baseline !important;
    gap: 12px !important;
    flex-wrap: wrap !important;
}

.hero-name {
    margin: 0 !important;
    font-size: 1.35rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.025em !important;
    color: #0f172a !important;
    line-height: 1.15 !important;
}

.hero-tagline {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: #6366f1 !important;
    letter-spacing: 0.01em !important;
}

/* ── Hero Pills Row ── */
.hero-pills {
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
    flex-wrap: wrap !important;
}

.hero-pill {
    display: inline-flex !important;
    align-items: center !important;
    gap: 5px !important;
    padding: 3px 9px !important;
    border-radius: 9999px !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    white-space: nowrap !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    cursor: default !important;
}

.hero-pill:hover {
    transform: translateY(-1px) !important;
}

.pill-icon {
    font-size: 0.80rem !important;
    line-height: 1 !important;
}

/* Pulsing Green Status Pill */
.status-pill {
    background: rgba(16, 185, 129, 0.10) !important;
    border: 1px solid rgba(16, 185, 129, 0.3) !important;
    color: #047857 !important;
    box-shadow: 0 1px 3px rgba(16, 185, 129, 0.08) !important;
}

@keyframes green-beacon-glow {
    0% {
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.8), 0 0 5px rgba(16, 185, 129, 0.9);
    }
    70% {
        box-shadow: 0 0 0 6px rgba(16, 185, 129, 0), 0 0 10px rgba(16, 185, 129, 0.9);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0), 0 0 5px rgba(16, 185, 129, 0.9);
    }
}

.pulse-green-dot {
    width: 7px !important;
    height: 7px !important;
    border-radius: 50% !important;
    background: #10b981 !important;
    display: inline-block !important;
    animation: green-beacon-glow 1.8s infinite ease-in-out !important;
}

.status-pill-text {
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
}

/* Open to work pill */
.open-to-work-pill {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.14) 0%, rgba(52, 211, 153, 0.25) 100%) !important;
    border: 1px solid rgba(16, 185, 129, 0.4) !important;
    color: #065f46 !important;
    font-weight: 700 !important;
    box-shadow: 0 1px 4px rgba(16, 185, 129, 0.12) !important;
}

/* Agentic AI pill */
.agentic-pill {
    background: rgba(99, 102, 241, 0.1) !important;
    border: 1px solid rgba(99, 102, 241, 0.28) !important;
    color: #4338ca !important;
}

/* UI Developer pill */
.ui-pill {
    background: rgba(14, 165, 233, 0.1) !important;
    border: 1px solid rgba(14, 165, 233, 0.28) !important;
    color: #0284c7 !important;
}

/* Agile/SDLC pill */
.agile-pill {
    background: rgba(245, 158, 11, 0.1) !important;
    border: 1px solid rgba(245, 158, 11, 0.28) !important;
    color: #b45309 !important;
}

.main-content-row {
    margin-top: 0 !important;
    gap: 16px !important;
    align-items: flex-start !important;
}

.mode-toggle {
    background: rgba(255, 255, 255, 0.85) !important;
    border: 1px solid rgba(226, 232, 240, 0.95) !important;
    border-radius: 14px !important;
    padding: 11px 14px !important;
    box-sizing: border-box !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.mode-toggle:hover {
    border-color: rgba(99, 102, 241, 0.5) !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.12) !important;
}

/* ── Dynamic Pipeline Status Indicator (Zero Numbers) ── */
@keyframes status-pulse {
    0%, 100% {
        transform: scale(1);
        opacity: 1;
    }
    50% {
        transform: scale(1.45);
        opacity: 0.55;
    }
}

@keyframes shimmer-slide {
    0% {
        background-position: -200% 0;
    }
    100% {
        background-position: 200% 0;
    }
}

.status-card {
    border-radius: 16px;
    padding: 14px 16px;
    margin-bottom: 12px;
    box-shadow: 0 4px 18px rgba(15, 23, 42, 0.04);
    border: 1px solid;
    transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
}

.status-card-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}

.status-beacon {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    position: relative;
    flex-shrink: 0;
}

.status-beacon::after {
    content: '';
    position: absolute;
    top: -4px;
    left: -4px;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: inherit;
    opacity: 0.45;
    animation: status-pulse 1.8s infinite ease-in-out;
}

.status-title-group {
    flex-grow: 1;
}

.status-headline {
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    margin: 0;
    line-height: 1.25;
}

.status-subtext {
    font-size: 0.8rem;
    margin: 2px 0 0 0;
    opacity: 0.88;
    line-height: 1.3;
}

.pipeline-track {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 5px;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid rgba(0, 0, 0, 0.06);
}

.pipeline-pill {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
    padding: 5px 6px;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    transition: all 0.3s ease;
    white-space: nowrap;
}

.pipeline-shimmer {
    height: 3px;
    border-radius: 9999px;
    margin-top: 8px;
    background: linear-gradient(90deg, rgba(255,255,255,0.1), currentColor, rgba(255,255,255,0.1));
    background-size: 200% 100%;
    animation: shimmer-slide 1.6s infinite linear;
}

/* ── Minimal Loader: electric blue → violet glow, sits under the user's bubble ── */
@keyframes rag-sweep {
    0%   { transform: translateX(-110%); }
    100% { transform: translateX(260%); }
}

@keyframes rag-breathe {
    0%, 100% { opacity: 0.55; }
    50%      { opacity: 1; }
}

/* Progress bar loader: neatly contained within the card */
#main-chatbot [data-testid="bot"]:has(.rag-loader),
.chatbot [data-testid="bot"]:has(.rag-loader),
#main-chatbot .bot:has(.rag-loader),
.chatbot .bot:has(.rag-loader) {
    width: 220px !important;
    min-width: 220px !important;
    max-width: 220px !important;
    box-sizing: border-box !important;
    overflow: hidden !important;
    padding: 7px 12px 8px 12px !important;
}

.rag-loader {
    display: flex;
    flex-direction: column;
    gap: 6px;
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
    padding: 0 !important;
}

.rag-loader-track {
    position: relative;
    height: 4px;
    width: 100% !important;
    max-width: 100% !important;
    overflow: hidden !important;
    border-radius: 9999px;
    background: rgba(99, 102, 241, 0.15);
    box-shadow: inset 0 0 0 1px rgba(99, 102, 241, 0.2);
    box-sizing: border-box !important;
}

.rag-loader-glow {
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    width: 45%;
    border-radius: 9999px;
    background: linear-gradient(90deg, transparent 0%, #3b82f6 25%, #8b5cf6 75%, transparent 100%);
    box-shadow: 0 0 10px 1px rgba(99, 102, 241, 0.6);
    animation: rag-sweep 1.5s cubic-bezier(0.45, 0.05, 0.35, 1) infinite;
}

.rag-loader-label {
    font-size: 0.70rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: rag-breathe 1.8s ease-in-out infinite;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

@media (prefers-reduced-motion: reduce) {
    .rag-loader-glow { animation-duration: 3s; }
    .rag-loader-label { animation: none; }
}

/* ── Unified Query Input Card ── */
.query-box-card {
    background: #f4f6fb !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 6px 10px 5px 10px !important;
    box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05), 0 2px 6px -1px rgba(15, 23, 42, 0.03) !important;
    margin-top: 0 !important;
    gap: 4px !important;
    display: flex !important;
    flex-direction: column !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

/* Ensure child rows don't get card-on-card glass backgrounds */
.query-box-card .query-top-row,
.query-box-card .query-bottom-row,
.query-box-card .gradio-row {
    background: transparent !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

.query-top-row,
.query-box-card .query-top-row,
.query-box-card .query-top-row.gradio-row,
.query-box-card .query-top-row.row {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: flex-end !important;
    gap: 8px !important;
    width: 100% !important;
}

.query-top-row > * {
    min-width: 0 !important;
    flex-shrink: 0 !important;
}

.input-query-badge-wrap,
.input-query-badge-wrap.block {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 0 !important;
    max-width: max-content !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    display: flex !important;
    align-items: flex-end !important;
    white-space: nowrap !important;
}

.input-query-badge {
    background: #ede9fe !important;
    color: #4f46e5 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 0 12px !important;
    height: 36px !important;
    min-height: 36px !important;
    border-radius: 8px !important;
    white-space: nowrap !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    line-height: 1.2 !important;
    letter-spacing: -0.01em !important;
    box-sizing: border-box !important;
}

.query-msg-input,
.query-msg-input.block,
.query-msg-input label,
.query-msg-input .wrap {
    flex: 1 1 0% !important;
    flex-shrink: 1 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    min-width: 0 !important;
    width: 100% !important;
    min-height: 54px !important;
    height: auto !important;
    box-sizing: border-box !important;
}

.query-msg-input input,
.query-msg-input textarea {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 7px 12px !important;
    font-size: 0.88rem !important;
    color: #1e293b !important;
    line-height: 1.4 !important;
    field-sizing: content !important;
    min-height: 54px !important;
    height: auto !important;
    max-height: 240px !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    width: 100% !important;
    min-width: 0 !important;
    box-sizing: border-box !important;
    outline: none !important;
    resize: none !important;
    overflow-y: auto !important;
    scrollbar-width: thin !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
}

.query-msg-input input:focus,
.query-msg-input textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}

.query-msg-input input::placeholder,
.query-msg-input textarea::placeholder {
    color: #94a3b8 !important;
    font-weight: 400 !important;
}

.query-send-btn,
button.query-send-btn {
    flex: 0 0 auto !important;
    width: 85px !important;
    min-width: 85px !important;
    max-width: 85px !important;
    height: 36px !important;
    min-height: 36px !important;
    background: linear-gradient(135deg, #4f46e5 0%, #2563eb 55%, #0ea5e9 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    line-height: 36px !important;
    border-radius: 10px !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.28) !important;
    cursor: pointer !important;
    padding: 0 !important;
    white-space: nowrap !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease, opacity 0.15s ease !important;
}

.query-send-btn:hover:not(:disabled) {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.38) !important;
}

.query-send-btn:active:not(:disabled) {
    transform: translateY(0) !important;
}

/* ── Bottom Row Layout: Spacer (left), Clear Chat (center), Counter (right) ── */
.query-bottom-row {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: space-between !important;
    width: 100% !important;
    min-height: 28px !important;
    margin: 2px 0 0 0 !important;
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Left spacer matches right counter exactly in flex weight */
.query-bottom-spacer-wrap,
.query-bottom-spacer-wrap.block,
.query-bottom-row > .query-bottom-spacer-wrap {
    flex: 1 1 0% !important;
    min-width: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

.query-clear-btn,
button.query-clear-btn,
.query-clear-btn button {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 160px !important;
    height: 28px !important;
    min-height: 28px !important;
    max-height: 28px !important;
    padding: 2px 24px !important;
    line-height: 1 !important;
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 9999px !important;
    color: #334155 !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    white-space: nowrap !important;
    cursor: pointer !important;
    text-align: center !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin: 0 auto !important;
    transition: all 0.2s ease !important;
}

.query-clear-btn:hover,
button.query-clear-btn:hover,
.query-clear-btn button:hover {
    background: #f8fafc !important;
    border-color: #94a3b8 !important;
    color: #0f172a !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08) !important;
}

/* Right counter aligns to the right edge (right below Send button) */
.char-counter-wrapper,
.char-counter-wrapper.block,
.query-bottom-row > .char-counter-wrapper {
    flex: 1 1 0% !important;
    min-width: 0 !important;
    display: flex !important;
    justify-content: flex-end !important;
    align-items: center !important;
    text-align: right !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    padding: 0 2px 0 0 !important;
    margin: 0 !important;
}

.char-counter-wrapper .wrap,
.char-counter-wrapper .pending,
.char-counter-wrapper.pending,
.char-counter-wrapper .prose {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    display: inline-flex !important;
    justify-content: flex-end !important;
}

.char-counter {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    font-variant-numeric: tabular-nums !important;
    letter-spacing: 0.03em !important;
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    display: inline-block !important;
    line-height: 1.2 !important;
    transition: color 0.15s ease !important;
}

.char-counter-normal {
    color: #64748b !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.char-counter-exceeded {
    color: #e11d48 !important;
    font-weight: 700 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

button:disabled, .gr-button:disabled {
    opacity: 0.45 !important;
    cursor: not-allowed !important;
    pointer-events: none !important;
    filter: grayscale(0.4) !important;
}

/* ── Feedback Dialog: fixed overlay with a centred card ── */
#feedback-modal,
#jd-modal {
    position: fixed !important;
    inset: 0 !important;
    z-index: 1000 !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 16px !important;
    margin: 0 !important;
    max-width: none !important;
    background: rgba(15, 23, 42, 0.4) !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
    border: none !important;
    outline: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}

#feedback-modal:not(.hide):not(.hidden),
#jd-modal:not(.hide):not(.hidden) {
    display: flex !important;
}

/* Blur the page itself while the dialog is open (backdrop-filter alone is unreliable over glass panels) */
.gradio-container:has(#feedback-modal:not(.hide):not(.hidden)) .header-row,
.gradio-container:has(#jd-modal:not(.hide):not(.hidden)) .header-row,
.gradio-container:has(#feedback-modal:not(.hide):not(.hidden)) .main-content-row,
.gradio-container:has(#jd-modal:not(.hide):not(.hidden)) .main-content-row {
    filter: blur(6px) !important;
    pointer-events: none !important;
    user-select: none !important;
    transition: filter 0.2s ease !important;
}

#feedback-dialog,
#jd-dialog {
    width: 100% !important;
    max-width: 420px !important;
    max-height: 90vh !important;
    overflow-y: auto !important;
    flex: 0 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
    flex-wrap: nowrap !important;
    background: #ffffff !important;
    backdrop-filter: none !important;
    border: 1px solid rgba(226, 232, 240, 0.95) !important;
    outline: none !important;
    border-radius: 16px !important;
    padding: 16px 18px !important;
    gap: 8px !important;
    box-shadow: 0 24px 60px -12px rgba(15, 23, 42, 0.35) !important;
}

/* Stack every field vertically at full width; strip the inherited glass-card styling */
#feedback-dialog .form,
#jd-dialog .form,
#feedback-dialog .block,
#jd-dialog .block,
#feedback-dialog .gradio-row,
#jd-dialog .gradio-row,
#feedback-dialog .fb-actions,
#jd-dialog .fb-actions {
    background: transparent !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

#feedback-dialog .form,
#jd-dialog .form {
    display: flex !important;
    flex-direction: column !important;
    flex-wrap: nowrap !important;
    gap: 8px !important;
}

#feedback-dialog .form > *,
#jd-dialog .form > *,
#feedback-dialog > .block,
#jd-dialog > .block {
    width: 100% !important;
    min-width: 0 !important;
    flex: 0 0 auto !important;
}

#feedback-dialog h3,
#jd-dialog h3 {
    margin: 0 0 2px 0 !important;
    font-size: 0.95rem !important;
    color: #4338ca !important;
}

#feedback-dialog label span,
#jd-dialog label span,
#feedback-dialog [data-testid="block-info"],
#jd-dialog [data-testid="block-info"] {
    font-size: 0.74rem !important;
    font-weight: 600 !important;
    color: #475569 !important;
    margin-bottom: 3px !important;
}

#feedback-dialog input,
#jd-dialog input,
#feedback-dialog textarea,
#jd-dialog textarea {
    font-size: 0.8rem !important;
    line-height: 1.3 !important;
    padding: 5px 10px !important;
    border-radius: 8px !important;
}

#feedback-dialog input,
#jd-dialog input {
    height: 30px !important;
    min-height: 30px !important;
}

#feedback-dialog input::placeholder,
#jd-dialog input::placeholder,
#feedback-dialog textarea::placeholder,
#jd-dialog textarea::placeholder {
    font-size: 0.78rem !important;
}

.fb-hint p {
    margin: 0 !important;
    font-size: 0.72rem !important;
    color: #64748b !important;
}

/* Error list: force red on every descendant so the theme's prose colours can't override it */
.fb-error,
.fb-error *,
#feedback-dialog .fb-error-list,
#jd-dialog .fb-error-list,
#feedback-dialog .fb-error-list li,
#jd-dialog .fb-error-list li {
    color: #dc2626 !important;
    font-size: 0.74rem !important;
}

#feedback-dialog .fb-error-list,
#jd-dialog .fb-error-list {
    margin: 0 !important;
    padding-left: 16px !important;
}

#feedback-dialog .fb-error-list li::marker,
#jd-dialog .fb-error-list li::marker {
    color: #dc2626 !important;
}

#feedback-dialog .fb-actions,
#jd-dialog .fb-actions {
    display: flex !important;
    flex-direction: row !important;
    justify-content: flex-end !important;
    gap: 8px !important;
    margin-top: 4px !important;
}

#feedback-dialog .fb-actions button,
#jd-dialog .fb-actions button {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 0 !important;
    height: 28px !important;
    padding: 0 14px !important;
    font-size: 0.76rem !important;
    border-radius: 8px !important;
}

/* JD dialog: wider to fit a pasted job description */
#jd-dialog {
    max-width: 640px !important;
}
"""

with gr.Blocks(title="Career & Resume RAG Assistant") as demo:
    with gr.Row(elem_classes=["header-row"]):
        with gr.Column(scale=8, elem_classes=["header-main-col"]):
            gr.HTML(HERO_BANNER_HTML, elem_classes=["hero-banner-card"])
        with gr.Column(scale=4, min_width=260, elem_classes=["header-side-col"]):
            mode_toggle = gr.Radio(
                choices=["Recruiter Mode", "Developer Mode"],
                value="Recruiter Mode",
                label="Mode",
                elem_classes=["mode-toggle"]
            )

    with gr.Row(elem_classes=["main-content-row"]):
        with gr.Column(scale=8, elem_classes=["conversation-col"]):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=240,
                elem_id="main-chatbot",
                elem_classes=["chatbot", "main-chatbot"]
            )

            # Ready Options / Quick Chips inside the chatbot frame (Developer Mode only)
            eval_quick_heading = gr.Markdown("**⚡ Quick Options (click to run):**", visible=False)
            with gr.Row(elem_classes=["quick-options"], visible=False) as eval_quick_row:
                opt_download = gr.DownloadButton(
                    label="📥 Download Resume",
                    value=str(RESUME_DOWNLOAD_FILE),
                    size="sm",
                    variant="primary"
                )
                opt_jd_match = gr.Button("🎯 JD Match", size="sm", variant="primary")
                opt_agentic = gr.Button("🤖 Agentic AI Experience", size="sm", variant="secondary")
                opt_skills = gr.Button("💻 Web Development", size="sm", variant="secondary")
                opt_edu = gr.Button("🎓 Education", size="sm", variant="secondary")
                opt_feedback = gr.Button("💬 Leave Feedback", size="sm", variant="primary")

            with gr.Column(elem_classes=["query-box-card"]):
                with gr.Row(elem_classes=["query-top-row"]):
                    gr.HTML('<div class="input-query-badge">Your Question</div>', elem_classes=["input-query-badge-wrap"], container=False)
                    msg_input = gr.Textbox(
                        placeholder="Ask about Shravan's skills, experience, projects, or job role suitability...",
                        show_label=False,
                        container=False,
                        scale=8,
                        min_width=0,
                        lines=2,
                        max_lines=10,
                        elem_classes=["query-msg-input"]
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=0, min_width=95, elem_classes=["query-send-btn"])
                with gr.Row(elem_classes=["query-bottom-row"]):
                    gr.HTML('<div class="query-bottom-spacer"></div>', elem_classes=["query-bottom-spacer-wrap"], container=False)
                    clear_btn = gr.ClearButton(
                        components=[msg_input, chatbot],
                        value="Clear Chat",
                        elem_classes=["query-clear-btn"]
                    )
                    char_counter = gr.HTML(
                        value=f'<div class="char-counter char-counter-normal">0/{MAX_QUERY_CHARS}</div>',
                        elem_classes=["char-counter-wrapper"],
                        container=False
                    )

        # Quick Options stacked vertically to the right of the conversation (Recruiter Mode only)
        with gr.Column(scale=4, min_width=260, elem_classes=["quick-options-vertical", "side-panel"]) as recruiter_quick_col:
            gr.Markdown("**⚡ Quick Options**")
            rec_download = gr.DownloadButton(
                label="📥 Download Resume",
                value=str(RESUME_DOWNLOAD_FILE),
                size="sm",
                variant="primary"
            )
            rec_jd_match = gr.Button("🎯 JD Match", size="sm", variant="primary")
            rec_agentic = gr.Button("🤖 Agentic AI Experience", size="sm", variant="secondary")
            rec_skills = gr.Button("💻 Web Development", size="sm", variant="secondary")
            rec_edu = gr.Button("🎓 Education", size="sm", variant="secondary")
            rec_feedback = gr.Button("💬 Leave Feedback", size="sm", variant="primary")

        with gr.Column(scale=4, min_width=260, visible=False, elem_classes=["side-panel"]) as inspector_col:
            top_k_slider = gr.Slider(
                minimum=1,
                maximum=8,
                value=4,
                step=1,
                label="Top-K Chunks to Retrieve"
            )
            
            status_box = gr.HTML(
                value=render_pipeline_status("idle", "System Ready", "Awaiting question to evaluate candidate"),
                label="Pipeline Status"
            )
            
            with gr.Accordion("📄 Retrieved Source Chunks (Grounding Context)", open=True):
                sources_box = gr.Markdown(
                    value="*Retrieved chunks will appear here after asking a relevant question.*",
                    elem_classes=["sources-box"]
                )

    # Feedback dialog: a hidden column rendered as a fixed overlay via CSS
    with gr.Column(visible=False, elem_id="feedback-modal") as feedback_modal:
        with gr.Column(elem_id="feedback-dialog"):
            gr.Markdown("### 💬 Leave Feedback for Shravan")
            fb_name = gr.Textbox(label="Your Name *", max_lines=1)
            fb_contact = gr.Textbox(label="Contact", placeholder="e.g. 9876543210", max_lines=1)
            fb_email = gr.Textbox(label="E-mail", placeholder="you@company.com", max_lines=1)
            gr.Markdown("*Please provide at least one: Contact or E-mail.*", elem_classes=["fb-hint"])
            fb_text = gr.Textbox(label="Feedback *", lines=5)
            fb_error = gr.HTML("", elem_classes=["fb-error"])
            with gr.Row(elem_classes=["fb-actions"]):
                fb_reset = gr.Button("Reset", variant="secondary", size="sm", scale=0, min_width=0)
                fb_cancel = gr.Button("Cancel", variant="secondary", size="sm", scale=0, min_width=0)
                fb_submit = gr.Button("Submit Feedback", variant="primary", size="sm", scale=0, min_width=0)

    # JD Match dialog: same fixed-overlay pattern as the feedback dialog
    with gr.Column(visible=False, elem_id="jd-modal") as jd_modal:
        with gr.Column(elem_id="jd-dialog"):
            gr.Markdown("### 🎯 Match a Job Description")
            jd_text = gr.Textbox(
                label="Job Description *",
                placeholder="Paste the full job description (role, responsibilities, requirements)...",
                lines=12,
                max_lines=20
            )
            gr.Markdown(f"*Between {MIN_JD_CHARS} and {MAX_JD_CHARS} characters.*", elem_classes=["fb-hint"])
            jd_error = gr.HTML("", elem_classes=["fb-error"])
            with gr.Row(elem_classes=["fb-actions"]):
                jd_cancel = gr.Button("Cancel", variant="secondary", size="sm", scale=0, min_width=0)
                jd_submit = gr.Button("Analyze Match", variant="primary", size="sm", scale=0, min_width=0)

    # Event handlers
    def on_mode_change(mode):
        is_developer = (mode == "Developer Mode")
        return (
            gr.update(visible=is_developer),      # inspector panel
            gr.update(visible=is_developer),      # developer quick options heading
            gr.update(visible=is_developer),      # developer quick options row
            gr.update(visible=not is_developer),  # recruiter vertical quick options
        )

    mode_toggle.change(
        fn=on_mode_change,
        inputs=mode_toggle,
        outputs=[inspector_col, eval_quick_heading, eval_quick_row, recruiter_quick_col]
    )

    def submit_interaction(user_message, history, top_k):
        # Clear the input box immediately and yield responses
        for h, status, sources in chat_handler(user_message, history, top_k):
            yield h, status, sources, ""

    # Live character counter & disabling Send button when exceeding 250 characters
    def on_input_change(text):
        length = len(text or "")
        is_exceeded = length > MAX_QUERY_CHARS
        counter_class = "char-counter-exceeded" if is_exceeded else "char-counter-normal"
        counter_html = f'<div class="char-counter {counter_class}">{length}/{MAX_QUERY_CHARS}</div>'
        return counter_html, gr.update(interactive=not is_exceeded)

    # Fires on every edit (typing, backspace, paste, delete)
    msg_input.change(
        on_input_change,
        inputs=msg_input,
        outputs=[char_counter, send_btn],
        show_progress="hidden"
    )

    clear_btn.click(
        fn=lambda: (f'<div class="char-counter char-counter-normal">0/{MAX_QUERY_CHARS}</div>', gr.update(interactive=True)),
        outputs=[char_counter, send_btn],
        show_progress="hidden"
    )

    def submit_user_message(user_message, history, top_k):
        if not user_message or not user_message.strip() or len(user_message) > MAX_QUERY_CHARS:
            return
        for h, status, sources, msg in submit_interaction(user_message, history, top_k):
            yield h, status, sources, msg, f'<div class="char-counter char-counter-normal">0/{MAX_QUERY_CHARS}</div>', gr.update(interactive=True)

    msg_input.submit(
        submit_user_message,
        inputs=[msg_input, chatbot, top_k_slider],
        outputs=[chatbot, status_box, sources_box, msg_input, char_counter, send_btn],
        show_progress_on=[chatbot, status_box, sources_box]
    )

    send_btn.click(
        submit_user_message,
        inputs=[msg_input, chatbot, top_k_slider],
        outputs=[chatbot, status_box, sources_box, msg_input, char_counter, send_btn],
        show_progress_on=[chatbot, status_box, sources_box]
    )

    # Quick option button handlers (within the chatbot frame)
    def make_quick_handler(prompt_text):
        def _handler(history, top_k):
            # Animate the clicked button while the query runs, then restore it
            busy = gr.update(interactive=False, elem_classes=["quick-btn-loading"])
            for h, status, sources, msg in submit_interaction(prompt_text, history, top_k):
                yield h, status, sources, msg, busy
            yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.update(interactive=True, elem_classes=[])
        return _handler

    quick_options_map = [
        (opt_agentic, "What experience does Shravan have with Agentic AI and multi-agent frameworks?"),
        (opt_skills, "What are his core technical skills in web development and frontend architecture?"),
        (opt_edu, "What is Shravan's educational background and degrees?")
    ]
    # Recruiter Mode buttons reuse the same prompts
    quick_options_map += [
        (rec_btn, prompt) for rec_btn, (_, prompt) in zip(
            [rec_agentic, rec_skills, rec_edu], quick_options_map
        )
    ]

    for btn, prompt in quick_options_map:
        btn.click(
            fn=make_quick_handler(prompt),
            inputs=[chatbot, top_k_slider],
            outputs=[chatbot, status_box, sources_box, msg_input, btn],
            show_progress="hidden"
        )

    # Feedback dialog handlers
    feedback_outputs = [feedback_modal, fb_name, fb_contact, fb_email, fb_text, fb_error]

    def open_feedback():
        return gr.update(visible=True), ""

    def reset_feedback():
        return "", "", "", "", ""

    def close_feedback():
        return gr.update(visible=False), *reset_feedback()

    def submit_feedback(name, contact, email, feedback):
        errors = validate_feedback(name, contact, email, feedback)
        if errors:
            error_html = '<ul class="fb-error-list">' + "".join(f"<li>{e}</li>" for e in errors) + "</ul>"
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), error_html

        name, contact, email, feedback = name.strip(), contact.strip(), email.strip(), feedback.strip()
        print(
            f"[FEEDBACK {datetime.now().isoformat(timespec='seconds')}] "
            f"name={name!r} contact={contact!r} email={email!r}\n{feedback}"
        )

        sent, error_detail = send_feedback_email(name, contact, email, feedback)
        if not sent:
            print(f"[FEEDBACK] SendGrid delivery failed: {error_detail}")
            error_html = (
                '<ul class="fb-error-list"><li>Sorry, your feedback could not be sent right now. '
                f"Please try again later or e-mail {html.escape(CANDIDATE_EMAIL)} directly.</li></ul>"
            )
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), error_html

        gr.Info("Thank you! Your feedback has been sent to Shravan.")
        return close_feedback()

    for btn in (rec_feedback, opt_feedback):
        btn.click(fn=open_feedback, outputs=[feedback_modal, fb_error], show_progress="hidden")

    fb_reset.click(fn=reset_feedback, outputs=feedback_outputs[1:], show_progress="hidden")
    fb_cancel.click(fn=close_feedback, outputs=feedback_outputs, show_progress="hidden")
    fb_submit.click(
        fn=submit_feedback,
        inputs=[fb_name, fb_contact, fb_email, fb_text],
        outputs=feedback_outputs,
        show_progress="hidden"
    )

    # JD Match dialog handlers
    def open_jd_match():
        return gr.update(visible=True), ""

    def close_jd_match():
        return gr.update(visible=False), "", ""

    def run_jd_match(jd, history):
        errors = validate_jd(jd)
        if errors:
            error_html = '<ul class="fb-error-list">' + "".join(f"<li>{e}</li>" for e in errors) + "</ul>"
            yield gr.skip(), gr.skip(), error_html, gr.skip(), gr.skip(), gr.skip()
            return

        # Close the dialog, then stream the pipeline into the chat
        first = True
        for h, status, sources in jd_match_handler(jd, history):
            if first:
                yield *close_jd_match(), h, status, sources
                first = False
            else:
                yield gr.skip(), gr.skip(), gr.skip(), h, status, sources

    for btn in (rec_jd_match, opt_jd_match):
        btn.click(fn=open_jd_match, outputs=[jd_modal, jd_error], show_progress="hidden")

    jd_cancel.click(fn=close_jd_match, outputs=[jd_modal, jd_text, jd_error], show_progress="hidden")
    jd_submit.click(
        fn=run_jd_match,
        inputs=[jd_text, chatbot],
        outputs=[jd_modal, jd_text, jd_error, chatbot, status_box, sources_box],
        show_progress="hidden"
    )


if __name__ == "__main__":
    server_port = int(os.getenv("PORT", 7860))
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        theme=light_theme,
        css=custom_css,
    )
