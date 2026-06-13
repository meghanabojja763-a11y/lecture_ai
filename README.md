# 🎓 LectureAI — AI-Powered Lecture Assistant

A full-stack AI system that records lectures, transcribes speech to text,
generates structured summaries, and lets students ask questions using RAG.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| LLM | Groq API (LLaMA 3 / Mixtral) |
| Speech-to-Text | Groq Whisper large-v3 |
| Agentic AI | LangChain ReAct Agent |
| RAG | LangChain + FAISS + HuggingFace Embeddings |
| Storage | JSON / SQLite |

## Project Structure

```
lecture_ai/
├── app.py                   # Streamlit frontend (main entry point)
├── agents/
│   └── lecture_agent.py     # LangChain ReAct agent orchestrator
├── rag/
│   └── rag_pipeline.py      # RAG pipeline (chunking + embeddings + retrieval)
├── tools/
│   ├── transcribe_tool.py   # Groq Whisper transcription
│   ├── summarize_tool.py    # LLM-powered summarization
│   ├── retrieval_tool.py    # RAG retrieval wrapper
│   └── quiz_tool.py         # MCQ quiz generator
├── utils/
│   ├── audio_utils.py       # Audio file handling
│   └── storage.py           # JSON persistence layer
├── requirements.txt
└── .env.example
```

## Setup

### 1. Clone and install dependencies
```bash
git clone <repo>
cd lecture_ai
pip install -r requirements.txt
```

### 2. Get a free Groq API key
Visit https://console.groq.com and create a free account.
Groq gives you fast inference on LLaMA 3 and Whisper for free.

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 4. Run the app
```bash
streamlit run app.py
```

## How It Works

### Upload Flow
1. Student uploads an audio/video file
2. `TranscribeTool` sends it to Groq Whisper → timestamped transcript
3. `SummarizeTool` sends transcript to Groq LLaMA 3 → structured JSON summary
4. `RAGPipeline` chunks transcript → embeds with `all-MiniLM-L6-v2` → stores in FAISS

### Q&A Flow
1. Student types a question
2. `LectureAgent` (ReAct loop) decides: use `RetrieveFromLecture` or `SummarizeTopic`
3. Tool fetches semantically similar chunks from FAISS
4. Groq LLM generates a grounded answer with source citations

### Agentic Behavior
The LangChain ReAct agent can:
- Route to different tools based on question type
- Maintain multi-turn conversation memory
- Combine retrieval + summarization in a single response
- Generate quizzes from lecture content

## Models Used

| Task | Model | Speed |
|------|-------|-------|
| Transcription | `whisper-large-v3` | ~1 min audio in 5s |
| Q&A / Summary | `llama3-8b-8192` | ~200 tok/s |
| High-quality Q&A | `llama3-70b-8192` | ~100 tok/s |
| Long context | `mixtral-8x7b-32768` | ~150 tok/s |
| Embeddings | `all-MiniLM-L6-v2` | Local CPU |
