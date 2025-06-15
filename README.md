# Task-Oriented AI Agent with LangChain and Ollama

This project implements a fully functional agentic AI system using LangChain and Ollama. It's designed to solve user-defined tasks through multi-step reasoning, tool usage, and memory management, all powered by local LLMs.

## Features

- **Local LLM Integration**: Uses Ollama to run various open-source models (Mistral, LLaMA3.2, Gemma)
- **Multi-Tool Integration**:
  - 🔍 Web Search (DuckDuckGo)
  - 🧮 Calculator (LLMMathChain)
  - 🐍 Python REPL (with safety restrictions)
- **Memory Systems**:
  - Short-term: ConversationBufferWindowMemory
  - Long-term: FAISS vector store (optional)
- **Web Interface**: Intuitive Streamlit-based UI with chat history
- **Advanced Features**:
  - Multi-turn conversations
  - Code execution with safety measures
  - Mathematical calculations
  - Web search capabilities

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running
- At least one LLM model downloaded (e.g., `ollama pull llama3.2`)
- Git (for cloning the repository)
- pip (Python package manager)

## 🚀 Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/langchain-agent.git
   cd langchain-agent
   ```

2. **Set up the environment:**
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   # source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   py -m pip install -r requirements.txt
   ```

4. **Download a model (if not already done):**
   ```bash
   ollama pull llama3.2  # or any other supported model
   ```

5. **Run the application:**
   ```bash
   py -m streamlit run main_agent.py
   ```
   The app will be available at `http://localhost:8501`

## 🛠️ Implementation Details

### Agent Architecture

```mermaid
graph LR
    A[User Input] --> B[Agent]
    B --> C[Tool Selection]
    C --> D[Search Tool]
    C --> E[Calculator]
    C --> F[Python REPL]
    B --> G[Memory]
    G --> H[Short-term Memory]
    G --> I[Long-term Memory]
    B --> J[Response Generation]
    J --> K[User Output]
```

### Tool Integration

1. **Search Tool (DuckDuckGo)**
   - No API key required
   - Used for real-time information retrieval
   - Handles web searches and information lookups

2. **Calculator (LLMMathChain)**
   - Handles mathematical expressions
   - Uses SymPy for symbolic mathematics
   - Safe execution environment

3. **Python REPL**
   - Executes Python code
   - Safety restrictions in place
   - Timeout protection
   - Blocked operations (file system, network, etc.)

### Memory System

- **Short-term Memory**: `ConversationBufferWindowMemory`
  - Maintains context of recent conversation
  - Configurable window size (default: 5 messages)
  - Prevents context window overflow

- **Long-term Memory** (Optional)
  - FAISS vector store for persistent memory
  - Stores and retrieves relevant past interactions
  - Uses sentence-transformers for embeddings

## 🧠 Implementation Deep Dive

### 1. Core Architecture

#### Reasoning Engine
- **Base Model**: Uses Ollama with LLaMA 3.2 as the default LLM
- **Agent Type**: `CONVERSATIONAL_REACT_DESCRIPTION` for natural conversation flow
- **Memory**: Sliding window conversation memory (default: 5 messages)
- **Temperature**: 0.2 (balanced between creativity and focus)

#### Key Components
```mermaid
graph TD
    A[User Input] --> B[Agent]
    B --> C{Decision}
    C -->|Needs Search| D[Web Search]
    C -->|Needs Math| E[Calculator]
    C -->|Needs Code| F[Python REPL]
    D --> B
    E --> B
    F --> B
    B --> G[Formatted Response]
```

### 2. Prompt Design

#### System Prompt
The agent uses an implicit system prompt that:
1. Establishes the assistant's helpful, precise personality
2. Sets context awareness
3. Defines tool usage guidelines

#### Dynamic Prompting
- **Context Window**: 2048 tokens
- **Message Format**:
  ```python
  [
      {"role": "system", "content": "You are a helpful assistant..."},
      {"role": "user", "content": "What's 2+2?"},
      {"role": "assistant", "content": "4", "tool_calls": [...]}
  ]
  ```

### 3. Tool Integration

#### Web Search
- **Tool**: DuckDuckGo Search
- **Use Case**: Current events, general knowledge
- **Safety**: No API key required, rate-limited by default

#### Calculator
- **Implementation**: `LLMMathChain` for safe math evaluation
- **Capabilities**:
  - Basic arithmetic
  - Advanced functions (sqrt, sin, cos, etc.)
  - Unit conversions

#### Python REPL
- **Features**:
  - Full Python 3 execution
  - Sandboxed environment
  - Timeout protection
- **Safety**:
  - Blocks dangerous imports
  - Limited execution time
  - Memory constraints

### 4. Memory System

#### Short-term Memory
- **Type**: Sliding window buffer
- **Size**: Configurable (default: 5 messages)
- **Purpose**: Maintains conversation context

#### Implementation
```python
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,  # Number of messages to remember
    return_messages=True,
    output_key="output"
)
```

### 5. Error Handling

#### Graceful Degradation
1. **Model Failures**: Falls back to simple responses
2. **Tool Errors**: Provides user-friendly error messages
3. **Rate Limiting**: Implements exponential backoff

#### Debug Mode
- Toggle in UI
- Shows detailed error logs
- Displays intermediate reasoning steps

### 6. Performance Considerations

#### Optimization
- **Caching**: LLM responses are cached
- **Batching**: Multiple tool calls in parallel
- **Streaming**: Real-time response generation

#### Limitations
- Context window constraints
- No long-term persistence by default
- Limited to available system resources

### 7. Security

#### Measures
1. Input sanitization
2. Restricted Python execution
3. No persistent storage of sensitive data
4. Rate limiting on API calls

#### Best Practices
- Environment variables for configuration
- Minimal required permissions
- Regular dependency updates

## ⚙️ Configuration

### Model Selection

Change the default model in the Streamlit UI sidebar or modify `main_agent.py`:

```python
llm = ChatOllama(
    model="llama3.2",  # Change to your preferred model
    temperature=0.2,  # Adjust creativity (0.0 to 1.0)
    num_ctx=2048,    # Context window size
    num_gpu_layers=0, # GPU layers (if available)
    num_thread=4,    # CPU threads
    top_k=40,        # Top-k sampling
    top_p=0.9,       # Nucleus sampling
    repeat_penalty=1.1
)
```

### Available Models

Any model supported by Ollama can be used. Popular choices include:
- `mistral` - Good balance of speed and quality
- `llama3.2` - Meta's latest model
- `gemma` - Google's lightweight model

List available models:
```bash
ollama list
```

### Memory Configuration

Adjust memory settings in the Streamlit UI or modify `main_agent.py`:
- **Memory Window**: Number of recent messages to remember
- **Temperature**: Controls response randomness (0.0 to 1.0)

## 🧪 Evaluation & Testing

### Test Cases

The agent has been tested with various queries including:
1. **Information Retrieval**:
   - "What's the latest news about AI?"
   - "Tell me about quantum computing"

2. **Mathematical Operations**:
   - "Calculate 45 * 89 + (32 / 8)"
   - "What's the square root of 625?"

3. **Code Execution**:
   - "Write a Python function to calculate factorial"
   - "Show me how to sort a dictionary by value"

### Performance Metrics

- Accuracy: High for factual queries with web search
- Limitations: May struggle with very recent events or specialized knowledge

## 📸 Sample Screenshots

| About | Console Layout |
|-------|----------------|
| ![AI_Agent_About](https://github.com/user-attachments/assets/7e44d295-27d4-4b2d-b0b6-04499d507d4b) | ![AI_Agent_Console](https://github.com/user-attachments/assets/7af1d878-f45f-4e7a-b478-7ab6a9d911bc) |

| Interface Layout | Code Generation |
|------------------|-----------------|
| ![AI_Agent_Display](https://github.com/user-attachments/assets/c4636a4f-0698-4b9a-95af-f6e9f669a59a) | ![AI_Agent_Code](https://github.com/user-attachments/assets/c7412f60-d893-4e12-87bc-998dbd8bb8cc) |

| Math Solver | Quick Response |
|-------------|----------------|
| ![AI_Agent_Mathematical_Calculation](https://github.com/user-attachments/assets/d2bbc2c1-e932-4748-8f0f-2c602fb25139) | ![AI_Agent_Response](https://github.com/user-attachments/assets/7b003202-a273-46f4-86eb-706b54b58d66) |

## 🐛 Known Issues & Limitations

- First response might be slow as the model warms up
- Complex queries may time out (currently set to 30s)
- Limited context window affects long conversations
- Some mathematical expressions might be misinterpreted

## 🔧 Troubleshooting

### Common Issues

1. **Agent not responding**
   - Ensure Ollama is running: `ollama serve`
   - Check model is downloaded: `ollama list`
   - Verify port 8501 is available

2. **Search not working**
   - Check internet connection
   - Verify DuckDuckGo is accessible

3. **Python REPL errors**
   - Some operations are restricted for security
   - Check error messages for blocked operations

### Getting Help

If you encounter issues:
1. Check the debug info in the sidebar
2. Look for error messages in the console
3. Open an issue on GitHub with details of the problem

## 📚 Documentation

### Project Structure

```
langchain-agent/
├── main_agent.py              # Main application code
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── .gitignore         # Git ignore file
```

### Key Components

1. **main_agent.py**
   - Main Streamlit application
   - Agent initialization and configuration
   - UI components and event handling
   - Tool implementations

2. **requirements.txt**
   - Lists all Python dependencies
   - Pinned versions for reproducibility

## 🚀 Deployment

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
py -m streamlit run main_agent.py
```

### Production Deployment

For production use, consider:
1. Using a WSGI server like Gunicorn with a reverse proxy (Nginx/Apache)
2. Setting up a systemd service for the Ollama server
3. Implementing proper logging and monitoring

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [LangChain](https://python.langchain.com/) for the agent framework
- [Ollama](https://ollama.ai/) for local LLM support
- [Streamlit](https://streamlit.io/) for the web interface
- All open-source contributors and model creators
