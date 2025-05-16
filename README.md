## 🚀 Quick Setup (Windows)

### 1. ✅ Install Python 3.10

Download from: [https://www.python.org/downloads/release/python-3100/](https://www.python.org/downloads/release/python-3100/)  
☑️ **Check “Add Python to PATH”** during installation

---

### 2. ✅ Download and unzip

Unzip the full folder to somewhere like:

C:\AI\multi-llm

yaml
Copy
Edit

---

### 3. ✅ Open Command Prompt inside the folder

Right-click → “Open in Terminal”  
Or manually in CMD:

```cmd
cd C:\AI\multi-llm
4. ✅ Create and activate the Python environment
cmd
Copy
Edit
python -m venv .venv
.venv\Scripts\activate
5. ✅ Install dependencies
cmd
Copy
Edit
pip install -r requirements.txt
6. ✅ Run the backend server
cmd
Copy
Edit
python multi_llm_backend.py
You should see:

arduino
Copy
Edit
INFO:     Uvicorn running on http://127.0.0.1:8000
7. ✅ Open your browser
Go to:

arduino
Copy
Edit
http://localhost:8000
You’re now chatting with a fully local multi-LLM engine.

yaml
Copy
Edit

---
