# Task-Oriented AI Agent with LangChain and Ollama

This project implements a functional agentic AI system using LangChain and Ollama. It supports tool-based reasoning, memory, local LLMs, and a Streamlit web interface.

---

## Features

- **Local LLMs via Ollama**  
  Supports Mistral, LLaMA 3.2, Gemma, etc.
- **Tool Integration**  
  🔍 Web Search (DuckDuckGo), 🧮 Calculator (LLMMathChain), 🐍 Python REPL (sandboxed)
- **Memory**  
  Short-term (ConversationBuffer), optional long-term memory (FAISS)
- **Web UI**  
  Built with Streamlit

---

## Object Detection Models

This project also includes two image recognition models with separate `predict.py` scripts:

| Model         | Accuracy       | Trained On           | # Object Types |
|---------------|----------------|-----------------------|----------------|
| `Drone_Model` | ✅ Very High    | Drone images only     | 1 type         |
| `Images_Recog`| ⚠️ Lower        | Mixed image dataset   | 14 types       |

Each model has its own `predict.py` script for inference:
 - Drone_Predict.py
 - Images_General_Predict.py

## Quick Start

# 1. Clone the repo
git clone https://github.com/yourusername/langchain-agent.git
cd langchain-agent

# 2. Set up environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download a model
ollama pull llama3.2

# 5. Run the app
streamlit run main_agent.py

## Structure

langchain-agent/
├── main_agent.py          # Main Streamlit app
├── requirements.txt       # Python dependencies
├── Drone_Model/           # High-accuracy drone-only model
│   └── predict.py
├── Images_Recog/          # General-purpose, multi-object model
│   └── predict.py
├── README.md
└── .gitignore

