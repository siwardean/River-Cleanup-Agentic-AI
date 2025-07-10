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

## 🚀 Quick Start

### 1. Clone the Repository

Clone the repository and navigate to the project folder:

    git clone https://github.com/yourusername/langchain-agent.git
    cd langchain-agent

### 2. Set Up the Python Environment

Create a virtual environment:

    python -m venv venv

Activate the environment:

- On **Windows**:

      venv\Scripts\activate

- On **macOS/Linux**:

      source venv/bin/activate

### 3. Install Dependencies

Install the required Python packages:

    pip install -r requirements.txt

### 4. Download a Model with Ollama

Pull a supported local model (e.g., LLaMA 3.2):

    ollama pull llama3.2

Make sure the Ollama server is running:

    ollama serve

### 5. Run the Application

Launch the Streamlit web interface:

    streamlit run main_agent.py

Open your browser and go to: [http://localhost:8501](http://localhost:8501)

---

## 📁 Project Structure

    langchain-agent/
    ├── main_agent.py          # Main Streamlit app (LLM agent)
    ├── requirements.txt       # Python dependencies
    ├── Drone_Model/           # High-accuracy, drone-specific model
    │   └── predict.py         # Inference script
    ├── Images_Recog/          # Multi-class, general image model
    │   └── predict.py         # Inference script
    ├── README.md              # Project documentation
    └── .gitignore             # Git exclusions

