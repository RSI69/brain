# 🧠 multi-llm

Local, private, multi-model AI chat engine with brutal honesty and real safety filters.  
**Runs 100% offline.**

---

## Features

- Multiple local LLMs (route code, chat, long-context)
- No flattery, no apologies — brutal, profanity-enabled replies (edit backend to change)
- Remembers last 5 messages per session
- Built-in safety filters (blocks credit cards, CP, bomb guides, scams)
- Streaming replies with % progress bar
- FastAPI backend + simple local web UI
- Open source, MIT license

---

## Requirements

- **GPU:** NVIDIA 3080+ or Apple M-series  
- **RAM:** 16GB minimum  
- **OS:** Windows (tested) or Linux  
- **Python:** 3.10

---

## 🚀 Quick Setup (Windows, 5 Minutes)

### 1. Install Python 3.10

- [Download Python 3.10](https://www.python.org/downloads/release/python-3100/)
- **IMPORTANT:** Check "Add Python to PATH" during install

---

### 2. Download and Unzip

Unzip the `multi-llm` folder anywhere (e.g., `C:\AI\multi-llm`)

---

### 3. Open Command Prompt in the Folder

```cmd
cd C:\AI\multi-llm
4. Create and Activate Python Environment
cmd
Copy
Edit
python -m venv .venv
.venv\Scripts\activate
5. Install Dependencies
cmd
Copy
Edit
pip install -r requirements.txt
6. Download Model Files
Get these .gguf files and place them in the /models folder:

Mistral-7B-Instruct-Q4_K_M.gguf

Meta-Llama-3-8B-Instruct-Q4_K_M.gguf

Your folder should look like:

markdown
Copy
Edit
multi-llm/
 └─ models/
      ├─ mistral-7b-instruct-v0.1.Q4_K_M.gguf
      └─ Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
7. Start the Server
cmd
Copy
Edit
python multi_llm_backend.py
You should see something like:

arduino
Copy
Edit
INFO:     Uvicorn running on http://127.0.0.1:8000
8. Open Your Browser
Go to http://localhost:8000

FAQ
Does it use the internet?
No. All inference is local. Nothing is logged or sent anywhere.

Change tone or models?
Edit the variables at the top of multi_llm_backend.py.

Linux?
Use source .venv/bin/activate to activate venv, rest is the same.

Out of memory?
Use a smaller model or close other GPU apps.

License
MIT — Do whatever you want.

yaml
Copy
Edit

---
