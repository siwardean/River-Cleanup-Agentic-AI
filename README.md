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
# 🌊 River Cleanup Hackathon – AI Agent for River Selection

This project is an AI-powered assistant designed for the **River Cleanup Hackathon**. It helps prioritize river sites for cleanup efforts by analyzing satellite/drone/mobile images and public data, and identifying waste hotspots using vision-based AI and LLM reasoning.

Built with **LangChain**, **Streamlit**, **Azure OpenAI (GPT-4o)**, and custom **object detection models**, this tool combines intelligent analysis and visualization into one streamlined interface.

---

## 🧠 Core Capabilities

- **Image Analysis with GPT-4o Vision**  
  Upload drone/satellite/phone images and get structured JSON outputs with bounding boxes for visible waste.

- **YOLO-Based Object Detection**  
  High-accuracy drone model and general-purpose image recognition model to detect visual waste and annotate images.

- **Tool-Enhanced AI Agent**  
  Uses LangChain tools for:
  - 🔍 Web Search (DuckDuckGo)
  - 🧮 Math Calculations (LLMMathChain)
  - 🐍 Python Scripting (via REPL)
  - 🧠 Transferability logic for river comparison

- **Streamlit Web UI**  
  Easy-to-use browser interface to upload images, run models, and visualize results.

---

## 🔬 Use Case: River Selection

This agent helps answer:
- Which rivers are most impacted visually by waste?
- Where are the highest density zones of litter in a given river image?
- Can cleanup efforts from one river generalize to another (transferability)?
- What does public/satellite data suggest about pollution patterns?

---

## 📦 Models Included

| Model         | Description             | Trained On            | Output         |
|---------------|-------------------------|------------------------|----------------|
| `Drone_Model` | High-precision detector | Drone images only      | 1 class (waste)|
| `Images_Recog`| Broader detection       | Mixed river/waste data | 14 object types|

Each model is used by the unified `predict.py` interface.

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/river-ai-agent.git
cd river-ai-agent
```

### 2. Set Up Python Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare Folders

Ensure these folders exist:

```bash
mkdir -p input static/annotated runs/predict
```

### 5. Run the Web App

```bash
streamlit run main_agent.py
```

Then open your browser at: [http://localhost:8501](http://localhost:8501)

---

## 📁 Project Structure

```
river-ai-agent/
├── main_agent.py              # Streamlit UI + LLM agent
├── openai_agent_setup.py      # Azure OpenAI + tool definitions
├── requirements.txt
├── predict.py                 # Object detection script (YOLO)
├── Drone_Model.pt             # Drone-specific model weights
├── Images_Recog_Model.pt      # General-purpose model weights
├── input/                     # Folder where uploaded images are saved
├── static/annotated/          # Folder for annotated output images
├── .streamlit/secrets.toml    # Azure OpenAI keys (not tracked by Git)
└── README.md
```

---

## 🔑 Configuration

Create a `.streamlit/secrets.toml` file with your Azure OpenAI credentials:

```toml
AZURE_OPENAI_API_KEY = "your-azure-api-key"
AZURE_OPENAI_ENDPOINT = "https://<your-resource>.openai.azure.com/"
AZURE_OPENAI_API_VERSION = "2024-05-01-preview"
AZURE_OPENAI_DEPLOYMENT_GPT4O = "gpt-4o"
AZURE_OPENAI_DEPLOYMENT_EMBEDDING = "text-embedding-3-large"
```

---

## 💡 Example Prompts

> "Analyze the uploaded drone image and return bounding boxes for all visible waste in JSON format."

> "Given satellite image A and mobile image B, which river has more visible litter?"

> "Can cleanup strategies from River X be applied to River Y based on waste types?"

---

## 🤝 Contributing

This tool is a hackathon prototype – feel free to fork, improve, and adapt it for your own river cleanup use case.
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

