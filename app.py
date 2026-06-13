import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

from agents.lecture_agent import LectureAgent
from utils.audio_utils import save_uploaded_audio
from utils.storage import LectureStorage

load_dotenv(dotenv_path=".env", override=True)

st.set_page_config(
    page_title="LectureAI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.lecture-card {
    background: #f8f9fa;
    border-left: 4px solid #6c63ff;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.summary-section h4 { color: #6c63ff; margin-bottom: 0.3rem; }
.chat-user {
    background: #e8e3ff;
    border-radius: 12px 12px 2px 12px;
    padding: 0.6rem 1rem;
    margin: 0.4rem 0;
    text-align: right;
}
.chat-ai {
    background: #f0f0f0;
    border-radius: 12px 12px 12px 2px;
    padding: 0.6rem 1rem;
    margin: 0.4rem 0;
}
.source-chip {
    background: #6c63ff22;
    color: #6c63ff;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin-right: 4px;
}
</style>
""", unsafe_allow_html=True)

if "agent" not in st.session_state:
    st.session_state.agent = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_lecture" not in st.session_state:
    st.session_state.current_lecture = None
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "storage" not in st.session_state:
    st.session_state.storage = LectureStorage()

storage: LectureStorage = st.session_state.storage

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=60)
    st.title("LectureAI")
    st.caption("Powered by Groq + LangChain")

    st.divider()
    st.subheader("⚙️ Settings")

    _env_key = os.getenv("GROQ_API_KEY", "")
    if _env_key:
        groq_api_key = _env_key
    else:
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Add GROQ_API_KEY=your_key to your .env file to skip this",
        )
        if not groq_api_key:
            st.caption("Tip: add GROQ_API_KEY=your_key to .env to auto-load")

    model_choice = st.selectbox(
        "LLM Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"],
        index=0,
    )

    st.divider()
    st.subheader("📚 Past Lectures")
    lectures = storage.list_lectures()
    if lectures:
        selected = st.selectbox("Load lecture", ["-- Select --"] + [l["title"] for l in lectures])
        if selected != "-- Select --":
            lecture_data = storage.get_lecture(selected)

            col_load, col_del = st.columns(2)

            # ── Load ──────────────────────────────────────────────────────
            with col_load:
                if lecture_data and st.button("Load", use_container_width=True):
                    st.session_state.transcript = lecture_data["transcript"]
                    st.session_state.summary = lecture_data["summary"]
                    st.session_state.current_lecture = selected
                    st.session_state.agent = LectureAgent(
                        groq_api_key=groq_api_key,
                        model_name=model_choice,
                        transcript=lecture_data["transcript"],
                    )
                    st.session_state.chat_history = []
                    st.rerun()

            # ── Delete ────────────────────────────────────────────────────
            with col_del:
                if st.button("Delete", use_container_width=True, type="secondary"):
                    st.session_state._confirm_delete = selected

            # Confirm before deleting
            if st.session_state.get("_confirm_delete") == selected:
                st.warning(f"Delete **{selected}**?")
                yes, no = st.columns(2)
                with yes:
                    if st.button("Yes, delete", use_container_width=True):
                        storage.delete_lecture(selected)
                        # Clear session if deleted lecture was loaded
                        if st.session_state.current_lecture == selected:
                            st.session_state.agent = None
                            st.session_state.transcript = None
                            st.session_state.summary = None
                            st.session_state.current_lecture = None
                            st.session_state.chat_history = []
                        st.session_state._confirm_delete = None
                        st.rerun()
                with no:
                    if st.button("Cancel", use_container_width=True):
                        st.session_state._confirm_delete = None
                        st.rerun()

            # ── Lecture info ──────────────────────────────────────────────
            if lecture_data:
                st.caption(f"Words: {lecture_data.get('word_count', 'N/A')} · Saved: {lecture_data.get('created_at', '')[:10]}")

    else:
        st.caption("No lectures saved yet.")

st.title("🎓 AI Lecture Assistant")
st.caption("Record → Transcribe → Summarize → Ask anything about your lecture")

tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📋 Summary", "💬 Ask Questions"])

# ════════════════════════════════════════════════════════
# TAB 1
# ════════════════════════════════════════════════════════
with tab1:
    st.subheader("Upload Lecture Audio")
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload audio file",
            type=["mp3", "wav", "m4a", "ogg", "flac"],
            help="Supports most audio/video formats",
        )
        lecture_title = st.text_input("Lecture title", placeholder="e.g. Introduction to Neural Networks")

        # Duplicate name check
        is_duplicate = False
        if lecture_title:
            existing_titles = [l["title"] for l in storage.list_lectures()]
            if lecture_title in existing_titles:
                st.warning(f"⚠️ A lecture named **{lecture_title}** already exists. Please choose a different name.")
                is_duplicate = True

    with col2:
        st.info("**Supported formats**\n\nMP3 · WAV · M4A · OGG · FLAC\n\n**Max size:** ~25 MB")

    if uploaded_file and lecture_title and not is_duplicate:
        if st.button("🚀 Process Lecture", type="primary", use_container_width=True):
            if not groq_api_key:
                st.error("Please enter your Groq API key in the sidebar.")
            else:
                with st.status("Processing lecture...", expanded=True) as status:
                    st.write("Saving audio file...")
                    audio_path = save_uploaded_audio(uploaded_file, lecture_title)

                    st.write("Initializing AI agent...")
                    agent = LectureAgent(
                        groq_api_key=groq_api_key,
                        model_name=model_choice,
                    )

                    st.write("Transcribing audio (Groq Whisper)...")
                    transcript = agent.transcribe(audio_path)
                    st.session_state.transcript = transcript

                    st.write("Generating structured summary...")
                    summary = agent.summarize(transcript)
                    st.session_state.summary = summary

                    st.write("Building knowledge index (RAG)...")
                    agent.build_rag_index(transcript)

                    st.write("Saving to storage...")
                    storage.save_lecture(lecture_title, transcript, summary)

                    st.session_state.agent = agent
                    st.session_state.current_lecture = lecture_title
                    st.session_state.chat_history = []
                    status.update(label="Done!", state="complete")

                st.success(f"Lecture **{lecture_title}** processed successfully!")

    if st.session_state.transcript:
        with st.expander("📄 View Full Transcript", expanded=False):
            st.text_area("Transcript", st.session_state.transcript, height=300, disabled=True)

    # ── Rename current lecture (Update) ───────────────────────────────────────
    if st.session_state.current_lecture:
        with st.expander("✏️ Rename this lecture", expanded=False):
            new_name = st.text_input("New name", value=st.session_state.current_lecture, key="rename_input")
            existing_titles = [l["title"] for l in storage.list_lectures()]
            name_taken = new_name in existing_titles and new_name != st.session_state.current_lecture
            if name_taken:
                st.warning("⚠️ That name already exists.")
            if st.button("Save new name", disabled=name_taken or not new_name):
                # Load existing data
                lecture_data = storage.get_lecture(st.session_state.current_lecture)
                if lecture_data:
                    # Delete old, save under new name
                    storage.delete_lecture(st.session_state.current_lecture)
                    storage.save_lecture(new_name, lecture_data["transcript"], lecture_data["summary"])
                    st.session_state.current_lecture = new_name
                    st.success(f"Renamed to **{new_name}**")
                    st.rerun()

# ════════════════════════════════════════════════════════
# TAB 2
# ════════════════════════════════════════════════════════
with tab2:
    if st.session_state.summary:
        st.subheader(f"📋 Summary: {st.session_state.current_lecture}")
        summary = st.session_state.summary

        if summary.get("overview"):
            st.markdown("### 🔭 Overview")
            st.info(summary["overview"])

        st.markdown("### 🎯 Key Topics")
        key_topics = summary.get("key_topics", [])
        if key_topics and st.session_state.agent:
            for topic in key_topics:
                with st.expander(f"{topic}", expanded=False):
                    with st.spinner(f"Loading info on '{topic}'..."):
                        result = st.session_state.agent.ask(
                            f"In 2-3 sentences, what does the lecture say about: {topic}?"
                        )
                    st.markdown(result["answer"])
        else:
            for topic in key_topics:
                st.markdown(f"- {topic}")

        st.divider()

        sections = {
            "📌 Main Points": summary.get("main_points", []),
            "💡 Key Concepts": summary.get("key_concepts", []),
            "📝 Action Items": summary.get("action_items", []),
        }
        col1, col2 = st.columns(2)
        cols = [col1, col2]
        for i, (title, items) in enumerate(sections.items()):
            with cols[i % 2]:
                st.markdown(f"<div class='lecture-card'><h4>{title}</h4>", unsafe_allow_html=True)
                if isinstance(items, list):
                    for item in items:
                        st.markdown(f"- {item}")
                else:
                    st.markdown(items)
                st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        st.subheader("🧠 Generate Quiz")
        num_q = st.slider("Number of questions", 3, 10, 5)
        if st.button("Generate Quiz", use_container_width=True):
            if st.session_state.agent:
                with st.spinner("Generating quiz..."):
                    quiz = st.session_state.agent.generate_quiz(
                        st.session_state.transcript, num_questions=num_q
                    )
                for i, q in enumerate(quiz, 1):
                    with st.expander(f"Q{i}. {q['question']}"):
                        for opt in q.get("options", []):
                            st.markdown(f"  - {opt}")
                        st.success(f"**Answer:** {q.get('answer', '')}")
                        st.caption(f"_{q.get('explanation', '')}_")
    else:
        st.info("Upload and process a lecture to see the summary here.")

# ════════════════════════════════════════════════════════
# TAB 3
# ════════════════════════════════════════════════════════
with tab3:
    if st.session_state.agent and st.session_state.transcript:
        st.subheader(f"💬 Ask about: {st.session_state.current_lecture}")

        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"<div class='chat-user'>{msg['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='chat-ai'>{msg['content']}</div>", unsafe_allow_html=True)

        st.divider()
        st.caption("Try asking:")
        suggestions = [
            "What are the main topics covered?",
            "Explain the key concepts in simple terms",
            "What are the most important takeaways?",
            "Summarize the second half of the lecture",
        ]
        scols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with scols[i % 2]:
                if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
                    st.session_state._pending_question = suggestion

        question = st.chat_input("Ask anything about this lecture...")

        if hasattr(st.session_state, "_pending_question"):
            question = st.session_state._pending_question
            del st.session_state._pending_question

        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.spinner("Thinking..."):
                result = st.session_state.agent.ask(question)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
            })
            st.rerun()
    else:
        st.info("Process a lecture first to enable Q&A.")
        if not st.session_state.agent:
            st.caption("Go to **Upload & Process** tab to get started.")