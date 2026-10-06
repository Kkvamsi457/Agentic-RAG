# Atlas — Agentic RAG & Multi-Tool System

This implementation follows the uploaded assignment: document ingestion, chunking and metadata,
vector indexing, dynamic tool selection, calculator/Python tools, agent routing, and citations.

## Backend

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
ollama pull qwen2.5:3b
python -m app.ingestion
python run.py
```

Put the four PDFs in `backend/data/`.

## Frontend

Open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL, normally `http://127.0.0.1:5173`.

## Flow

Every query first reaches the agent router. It selects RAG Search, Calculator,
Python Tool, or Direct Answer. The UI displays the selected tool before the answer.

## Assignment test prompts

1. What are the 6 pillars of the AWS Well-Architected Framework, and which pillar covers incident response?
2. Look up asyncio.TaskGroup in the Python reference and write a minimal concurrent example.
3. Using NIST SP 800-145, define the essential characteristics of cloud computing and state if on-demand self-service requires human interaction.
4. Calculate (1250 * 0.18) + 47.5.

The Python executor in this demo is intentionally restricted. For production use,
replace it with an isolated sandbox/container.
