import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
import pypdf
import gradio as gr
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

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


# ------------------------------------------------------------------------------
# 2. Vector Store Initialization (Load or Auto-Build)
# ------------------------------------------------------------------------------
def get_or_create_vector_store():
    """Loads existing FAISS index or creates a new one from the PDF resume."""
    if INDEX_DIR.exists() and (INDEX_DIR / "index.faiss").exists():
        print(f"Loading existing FAISS index from {INDEX_DIR}...")
        return FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )

    print(f"FAISS index not found. Building index from {PDF_FILE}...")
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
    guardrail_system_prompt = """You are a strict query classifier for a candidate's career and job-search assistant.

Your task:
1. Determine if the user's query is strictly about:
   - Job hunting, recruiting, hiring, or candidate evaluation
   - Professional experience, technical skills, projects, tools, or education
   - Suitability for specific job roles or tasks
2. If the query is personal (e.g., relationships, hobbies, private life) or off-topic (e.g., recipes, general trivia, weather, chit-chat), classify it as IRRELEVANT.

Output strictly valid JSON with two fields:
{
  "is_relevant": true | false,
  "deflection_message": ""
}
If is_relevant is false, provide a polite, professional deflection explaining that you only answer questions related to the candidate's professional background and job profile."""

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
    system_prompt = """You are a precise, factual career assistant.

CRITICAL INSTRUCTIONS:
1. Answer the question STRICTLY using ONLY the information provided in the "Source Material" below.
2. Do NOT use any outside knowledge, assumptions, extrapolation, or imagination.
3. If the answer cannot be fully deduced from the provided source material, you MUST state:
   "Based on the provided source material, this information is not available."
4. Do not speculate or invent qualifications or details."""

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
# 5. Chatbot Handler for Gradio
# ------------------------------------------------------------------------------
def chat_handler(user_message, history, top_k=4):
    if history is None:
        history = []

    default_sources = "*Retrieved chunks will appear here after asking a relevant question.*"

    if not user_message or not user_message.strip():
        yield history, "⚠️ Please enter a valid question.", default_sources
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
        status_md = "📥 **Resume Delivered:** PDF resume served directly inside the conversation frame."
        yield history, status_md, "*No chunks retrieved (resume download served).*"
        return

    # 1. Guardrail Check
    is_relevant, deflection_msg = is_job_hunt_query(user_message)

    if not is_relevant:
        # Off-topic: Deflect politely, do NOT retrieve or send chunks
        history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": deflection_msg}
        ]
        status_md = "🛡️ **Guardrail Triggered:** Query classified as *Off-Topic / Personal*. No resume chunks were retrieved or processed."
        yield history, status_md, "*No chunks retrieved (guardrail triggered - off-topic query).*"
        return

    # 2. Retrieve Chunks
    status_md = "🔍 **Guardrail Passed:** Retrieving context from knowledge base..."
    yield history, status_md, "*Retrieving relevant chunks...*"

    retrieved_docs = vector_store.similarity_search(user_message, k=int(top_k))

    formatted_sources = "\n\n---\n\n".join([
        f"**Source Chunk {doc.metadata.get('chunk_id', idx + 1)}** *(File: `{doc.metadata.get('source', PDF_FILE.name)}`)*:\n\n{doc.page_content}"
        for idx, doc in enumerate(retrieved_docs)
    ])

    # 3. Grounded Answer Generation
    status_md = "🤖 **Generating Answer:** Synthesizing answer strictly within source constraints..."
    yield history, status_md, formatted_sources

    answer = generate_grounded_answer(user_message, retrieved_docs)

    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": answer}
    ]
    status_md = "✅ **Complete:** Answer generated strictly within source constraints."
    yield history, status_md, formatted_sources


# ------------------------------------------------------------------------------
# 6. Gradio UI Construction
# ------------------------------------------------------------------------------
light_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate",
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
)

custom_css = """
/* ── Luminous Ambient Mesh Gradient Background (Not Generic Gradio White/Gray) ── */
.gradio-container {
    background: radial-gradient(at 0% 0%, rgba(224, 242, 254, 0.85) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(238, 242, 255, 0.9) 0px, transparent 50%),
                radial-gradient(at 50% 100%, rgba(240, 249, 255, 0.8) 0px, transparent 50%),
                linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
    min-height: 100vh;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
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

/* ── Chatbot Glass Canvas & Message Bubbles ── */
.chatbot {
    background: rgba(255, 255, 255, 0.65) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(226, 232, 240, 0.85) !important;
    border-radius: 18px !important;
    box-shadow: 0 8px 30px rgba(15, 23, 42, 0.04) !important;
}

.chatbot .user, .chatbot [data-testid="user"] {
    background: linear-gradient(135deg, #4f46e5 0%, #0ea5e9 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 18px 18px 4px 18px !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.25) !important;
}

.chatbot .user p, .chatbot [data-testid="user"] * {
    color: #ffffff !important;
}

.chatbot .bot, .chatbot [data-testid="bot"] {
    background: rgba(255, 255, 255, 0.92) !important;
    color: #0f172a !important;
    border: 1px solid rgba(226, 232, 240, 0.9) !important;
    border-radius: 18px 18px 18px 4px !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03) !important;
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

.quick-options button, .quick-options .gr-button {
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

.quick-options button:hover, .quick-options .gr-button:hover {
    transform: translateY(-2px) scale(1.02) !important;
    background: #ffffff !important;
    border-color: #6366f1 !important;
    color: #4f46e5 !important;
    box-shadow: 0 6px 18px rgba(99, 102, 241, 0.18), 0 2px 4px rgba(99, 102, 241, 0.08) !important;
}

.quick-options button:active, .quick-options .gr-button:active {
    transform: translateY(1px) scale(0.98) !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08) !important;
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
"""

with gr.Blocks(title="Career & Resume RAG Assistant") as demo:
    gr.Markdown(
        """
        # 💼 Career & Resume RAG Chatbot
        ### Guardrailed RAG Agent for Candidate Evaluation & Job Search
        *Powered by FAISS, OpenAI GPT-4o, and Strict Grounding Constraints.*
        """
    )

    with gr.Row():
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500
            )

            # Ready Options / Quick Chips inside the chatbot frame
            gr.Markdown("**⚡ Quick Options (click to run):**")
            with gr.Row(elem_classes=["quick-options"]):
                opt_download = gr.DownloadButton(
                    label="📥 Download Resume",
                    value=str(RESUME_DOWNLOAD_FILE),
                    size="sm",
                    variant="primary"
                )
                opt_agentic = gr.Button("🤖 Agentic AI Experience", size="sm", variant="secondary")
                opt_skills = gr.Button("💻 Web Development", size="sm", variant="secondary")
                opt_cbird = gr.Button("🚀 C-bird Project", size="sm", variant="secondary")
                opt_edu = gr.Button("🎓 Education", size="sm", variant="secondary")

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask about candidate's skills, experience, projects, or job role suitability...",
                    label="Your Question",
                    scale=9,
                    lines=1
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            with gr.Row():
                clear_btn = gr.ClearButton(components=[msg_input, chatbot], value="Clear Chat")

            gr.Examples(
                examples=[
                    "Download Resume (PDF)",
                    "What experience does Shravan have with Agentic AI and multi-agent frameworks?",
                    "What are his core technical skills in web development and frontend architecture?",
                    "Can Shravan oversee my factory's physical construction?",
                    "What are your favorite weekend hobbies and where do you hang out?",
                    "Tell me about the C-bird project at Drone Aerospace Systems."
                ],
                inputs=msg_input,
                label="💡 Try these test queries (Relevant, Negative Test, and Off-Topic Guardrail):"
            )

        with gr.Column(scale=5):
            gr.Markdown("### ⚙️ Pipeline Inspection & Controls")
            
            top_k_slider = gr.Slider(
                minimum=1,
                maximum=8,
                value=4,
                step=1,
                label="Top-K Chunks to Retrieve"
            )
            
            status_box = gr.Markdown(
                value="*Ready. Send a message to start.*",
                label="Current Pipeline Status"
            )
            
            with gr.Accordion("📄 Retrieved Source Chunks (Grounding Context)", open=True):
                sources_box = gr.Markdown(
                    value="*Retrieved chunks will appear here after asking a relevant question.*",
                    elem_classes=["sources-box"]
                )

    # Event handlers
    def submit_interaction(user_message, history, top_k):
        # Clear the input box immediately and yield responses
        for h, status, sources in chat_handler(user_message, history, top_k):
            yield h, status, sources, ""

    msg_input.submit(
        submit_interaction,
        inputs=[msg_input, chatbot, top_k_slider],
        outputs=[chatbot, status_box, sources_box, msg_input]
    )

    send_btn.click(
        submit_interaction,
        inputs=[msg_input, chatbot, top_k_slider],
        outputs=[chatbot, status_box, sources_box, msg_input]
    )

    # Quick option button handlers (within the chatbot frame)
    def make_quick_handler(prompt_text):
        def _handler(history, top_k):
            yield from submit_interaction(prompt_text, history, top_k)
        return _handler

    quick_options_map = [
        (opt_agentic, "What experience does Shravan have with Agentic AI and multi-agent frameworks?"),
        (opt_skills, "What are his core technical skills in web development and frontend architecture?"),
        (opt_cbird, "Tell me about the C-bird project at Drone Aerospace Systems."),
        (opt_edu, "What is Shravan's educational background and degrees?")
    ]

    for btn, prompt in quick_options_map:
        btn.click(
            fn=make_quick_handler(prompt),
            inputs=[chatbot, top_k_slider],
            outputs=[chatbot, status_box, sources_box, msg_input]
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
