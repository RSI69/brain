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

### 3. Open Command Prompt in the Folder cmd

```cmd
cd C:\AI\multi-llm

in cmd enter **python -m venv .venv
.venv\Scripts\activate**

in cmd enter **pip install -r requirements.txt**

On the internet, go download these .gguf files and place them in the **/models** folder:

**Mistral-7B-Instruct-Q4_K_M.gguf** search this on google and download (Q4_K_M version) in multi-llm/
 └─ models/ folder 

**Meta-Llama-3-8B-Instruct-Q4_K_M.gguf** search this on google and download (Q4_K_M version) in multi-llm/
 └─ models/ folder 

Your folder should look like:

multi-llm/
 └─ models/
      ├─ mistral-7b-instruct-v0.1.Q4_K_M.gguf
      └─ Meta-Llama-3-8B-Instruct.Q4_K_M.gguf

in **cmd** enter **python multi_llm_backend.py --host 0.0.0.0 --port 8000**

You should see: INFO:     Uvicorn running on http://127.0.0.1:8000

Open Your Browser Go to **http://localhost:8000**

FAQ
Does it use the internet?
No. All inference is local. Nothing is logged or sent anywhere.

Linux?
Use source .venv/bin/activate to activate venv. All other steps are the same.

Out of memory?
Use a smaller model or close other GPU apps.

License
MIT — Do whatever you want.
