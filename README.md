# 🧠 multi-llm

A fast, profanity-enabled local AI chat engine.  
No internet. No accounts. No bullshit. Just local inference with memory, safety, and speed.

---

## ✅ Features

- Multiple model routing (chat, code, long-form)
- Brutally honest tone — no flattery or apologies
- Safety filtering (credit cards, CP, bombs, scams)
- Optional RAG via FAISS
- FastAPI backend with SSE streaming
- Simple frontend UI (no JS frameworks)
- LRU cache + memory of previous 5 messages

---

## 🖥 Requirements

- GPU: RTX 3080+ (or Apple M1/M2/M3 for Metal)
- RAM: 16 GB minimum
- OS: Windows (tested) or Linux
- Python: 3.10

---

## 🚀 Quick Setup (Windows)

### 1. Install Python 3.10

Download from: https://www.python.org/downloads/release/python-3100/  
✅ **Check “Add Python to PATH” during install**

---

### 2. Download and unzip

Unzip the full folder to somewhere like:

```text
C:\AI\multi-llm

3. Open Command Prompt inside the folder
cmd
Copy
Edit
cd C:\AI\multi-llm

4. Create and activate the Python environment
cmd
Copy
Edit
python -m venv .venv
.venv\Scripts\activate

5. Install required libraries
cmd
Copy
Edit
pip install -r requirements.txt

6. Download models
Put these .gguf files into the /models folder:

Mistral-7B-Instruct.Q4_K_M

Llama-3-8B-Instruct.Q4_K_M

Your folder should look like:

text
Copy
Edit
multi-llm/
├── models/
│   ├── mistral-7b-instruct-v0.1.Q4_K_M.gguf
│   └── Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
(If you use different models, change their paths in multi_llm_backend.py.)

7. Run the backend server
cmd
Copy
Edit
python multi_llm_backend.py
You should see:

text
Copy
Edit
INFO:     Uvicorn running on http://127.0.0.1:8000

8. Open the chat UI
Go to your browser and visit:

arduino
Copy
Edit
http://localhost:8000
You're in.

🔐 Safety Filters
The backend blocks:

Credit card numbers (real PANs)

Child exploitation prompts

Bomb/explosive crafting guides

Fraud / scam phrasing

💡 Tips
System remembers last 5 messages per session

All processing is local — nothing leaves your PC

You can tweak tone, model routing, safety, and memory in multi_llm_backend.py

📃 License
MIT — no restrictions, fork and go.
