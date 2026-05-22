# FarmerChat

FarmerChat is an advanced multi-lingual voice and text-based conversational assistant designed to help farmers with weather forecasts, agricultural knowledge query lookup (RAG), and real-time audio translation/transcription.

Built with **FastAPI**, **LangGraph**, **ChromaDB**, **Twilio WhatsApp API**, and **Sarvam AI**.

## Folder Structure

```text
FarmerChat/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Environment variables & API keys
│   ├── database.py             # ChromaDB / Vector store setup
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py            # LangGraph State definitions
│   │   ├── workflow.py         # Main LangGraph orchestration compilation
│   │   └── nodes/
│   │       ├── __init__.py
│   │       ├── router.py       # LLM intent router node
│   │       ├── weather.py      # Weather API calling node
│   │       ├── rag.py          # Vector DB lookup node
│   │       └── translation.py  # Sarvam AI STT/TTS translation nodes
│   └── services/
│       ├── __init__.py
│       ├── twilio_sms.py       # Twilio WhatsApp webhook handlers
│       └── sarvam_api.py       # Direct Sarvam API wrappers
├── data/
│   ├── raw/                    # Raw Kisan Call Center transcripts / PDFs
│   └── chroma_db/              # Local vector database storage (ignored by git)
├── .env                        # Local environment secrets (API Keys)
├── .gitignore
├── README.md
└── requirements.txt            # Project dependencies
```

## Setup Instructions

1. **Clone the repository and enter the directory**:
   ```bash
   cd FarmerChat
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Unix or MacOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Copy the `.env` template and fill in your API keys:
   ```bash
   # Add keys to the .env file
   ```

5. **Run the application**:
   ```bash
   uvicorn app.main:app --reload
   ```
