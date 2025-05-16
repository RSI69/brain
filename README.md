# 🧠 multi-llm chat

A fast, offline, profanity-enabled AI chatbot that runs completely on your PC. No internet, no accounts, no spyware.

---

## ✅ Quick Start (Windows Only)

### 1. Install Python 3.10 (if you don’t have it)

Go to: https://www.python.org/downloads/release/python-3100/  
Download the Windows installer and **check "Add to PATH"** during installation.

---

### 2. Open Command Prompt

Press `Win + R`, type `cmd`, hit enter.

Then use `cd` to go to the folder where you downloaded this repo.  
For example:

**cd C:\Users\YourName\Downloads\multi-llm**

yaml
Copy
Edit

---

### 3. Activate Python environment

If you see a `Scripts` folder inside `.venv`, run:

**.venv\Scripts\activate**

lua
Copy
Edit

If not, create the environment and install everything:

**python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt**

yaml
Copy
Edit

---

### 4. Make sure model files are in the `/models` folder

This folder should contain:

- `mistral-7b-instruct-v0.1.Q4_K_M.gguf`
- `Meta-Llama-3-8B-Instruct.Q4_K_M.gguf`

(You can use different .gguf models if you want — just update the paths in `multi_llm_backend.py`.)

---

### 5. Run the server

In the same Command Prompt window:

**python multi_llm_backend.py**

sql
Copy
Edit

You should see logs like:

INFO: Uvicorn running on http://127.0.0.1:8000

yaml
Copy
Edit

---

### 6. Open the chat

Go to this address in your browser:

**http://localhost:8000**

yaml
Copy
Edit

You can now chat with your local AI system — it remembers previous messages and streams answers.

---

## 🛡️ Safety & Filters

The system **blocks:**

- Credit card numbers
- Child exploitation
- Explosive instructions
- Known scam patterns

---

## 💡 Tips

- Everything runs offline, no internet needed after model files are present
- You can modify the tone, memory length, or models inside `multi_llm_backend.py`
- The system streams replies word-by-word and shows a % loading bar

---

## 📃 License

MIT — do whatever you want.
