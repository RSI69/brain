
🧠 multi-llm
Local, private, multi-model AI chat engine with brutal honesty and real safety filters. Runs 100% offline.

Features
Runs entirely on your PC, no internet required after setup

Multiple local LLMs (route code, chat, long-context)

Remembers last 5 messages per session

Brutal, profanity-enabled replies (can be changed in backend)

Built-in safety filters (no scams, CP, or bomb guides)

Streams replies with progress bar

Open source, MIT license

Hardware Requirements
GPU: NVIDIA 3080+ or Apple M-series

RAM: 16GB minimum

OS: Windows (tested) or Linux

Python: 3.10

🚀 Setup (Windows) — 5 Minutes
1. Install Python 3.10
Download: python.org/downloads/release/python-3100/

When installing, check "Add Python to PATH"

2. Download and Unzip
Unzip the provided multi-llm folder anywhere (e.g., C:\AI\multi-llm)

3. Open Command Prompt in the Folder
Right click inside the folder → “Open in Terminal”
OR:

Open Command Prompt manually and run:

cmd
Copy
Edit
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
6. Download LLM Model Files
You need two .gguf files:

Mistral-7B-Instruct-Q4_K_M.gguf
Get it from Hugging Face

Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
Get it from Hugging Face

Place both files in the /models folder so it looks like:

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
You should see:

arduino
Copy
Edit
INFO:     Uvicorn running on http://127.0.0.1:8000
8. Open Your Browser
Go to:
http://localhost:8000

You’ll see the multi-llm chat UI—ready to use.

📒 FAQ
Will this send data online?
No. All inference is local. Nothing is logged or shared.

Where do I change model paths or tone?
Edit multi_llm_backend.py (the settings are at the top).

Linux?
Same steps. Activate venv with source .venv/bin/activate

It says “out of memory”?
Try a smaller model or close other GPU-using programs.

🛡️ Safety & Privacy
Blocks: credit card numbers, child abuse, bomb/explosives, scam/fraud

No tracking or analytics

You can adjust safety in backend if needed

License
MIT — Do whatever you want.

Enjoy your private, unfiltered, no-cloud AI chat engine.

