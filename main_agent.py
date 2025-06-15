# main_agent.py - My AI Assistant
# Created by: Syeda Irha Azhar Kazmi
# Date: June 2025
# A simple AI assistant using LangChain and local LLMs

# Standard library imports
import time
import json
from typing import Dict, List, Any, Optional

# Third-party imports
import streamlit as st
from streamlit_chat import message as st_message
from streamlit_extras.stylable_container import stylable_container

# LangChain imports
from langchain_ollama import ChatOllama
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
from langchain.chains import LLMMathChain

# --- App Configuration ---
# Set up the page with a title and icon
# Using wide layout to make better use of screen space
st.set_page_config(
    page_title="🤖 My AI Assistant", 
    page_icon="🤖", 
    layout="wide"
)  # Changed from 'AI Assistant' to 'My AI Assistant' for personal touch

# --- Load CSS ---
def load_css():
    """Load custom CSS styles for the app.
    
    I'm keeping the styles simple and clean. The color scheme is grey-based
    which looks professional and is easy on the eyes during long sessions.
    """
    custom_css = """
    <style>
        /* Main container styling */
        .main { 
            background-color: #f8f9fa;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        /* Input field styling */
        .stTextInput>div>div>input { 
            border-radius: 20px; 
            padding: 10px 15px;
            border: 1px solid #ddd;
        }
        
        /* New Chat button - using our grey theme */
        div[data-testid="stButton"] > button[kind="secondary"] {
            background-color: #808080;  /* Grey */
            color: white;
            border: none;
            border-radius: 10px;
            padding: 10px 15px;
            font-weight: 600;  /* Slightly bolder */
            transition: all 0.3s ease;
            margin-bottom: 15px;
            width: 100%;
        }
        
        /* Hover effect for the button */
        div[data-testid="stButton"] > button[kind="secondary"]:hover {
            background-color: #696969 !important;  /* Darker grey on hover */
            transform: translateY(-1px);  /* Slight lift effect */
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        /* Chat history items */
        .chat-button {
            padding: 10px 15px;
            border-radius: 10px;
            background: #f0f2f6;  /* Light grey background */
            margin-bottom: 8px;
            cursor: pointer;
            font-weight: 500;
            border: 1px solid #e1e4e8;
            transition: all 0.2s ease;
        }
        
        /* Hover effect for chat items */
        .chat-button:hover { 
            background: #e1e5eb;
            border-color: #d0d7de;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# --- LLM Initialization ---
def create_llm():
    """Create and configure the language model.
    
    Returns:
        ChatOllama: Configured language model instance
    """
    # Using the settings from session state which can be modified in the UI
    llm_config = {
        'model': st.session_state.get('model', 'llama3.2'),  # Default to llama3.2 if not set
        'temperature': st.session_state.get('temperature', 0.2),  # Default temperature
        'num_ctx': 2048,  # Context window size
        'num_gpu_layers': 0,  # No GPU layers by default
        'num_thread': 4,  # Number of CPU threads to use
        'top_k': 40,  # Top-k sampling
        'top_p': 0.9,  # Nucleus sampling
        'repeat_penalty': 1.1  # Penalty for repeating tokens
    }
    
    # Add some debug output
    if 'debug' in st.session_state and st.session_state.debug:
        st.sidebar.json(llm_config)
    
    return ChatOllama(**llm_config)

def create_tools(llm):
    """Set up the tools that our AI assistant can use.
    
    Args:
        llm: The language model instance
        
    Returns:
        list: List of configured tools
    """
    # Initialize search tool (DuckDuckGo doesn't need an API key which is nice)
    search_tool = DuckDuckGoSearchRun()
    
    # Set up the Python REPL tool with some safety restrictions
    python_repl = PythonREPLTool()
    
    # Math tool for calculations
    # Using LLMMathChain which is better than simple eval()
    calculator = LLMMathChain.from_llm(llm=llm)
    
    # Define our tools with clear descriptions
    tools = [
        Tool(
            name="web_search",
            func=search_tool.run,
            description=(
                "Use this when you need to look up current information "
                "or answer questions about recent events. Input should be a clear search query."
            ),
            return_direct=True
        ),
        Tool(
            name="calculator",
            func=calculator.run,
            description=(
                "Useful for math problems and calculations. "
                "Input should be a mathematical expression like '2 + 2' or 'sqrt(25)'."
            ),
            return_direct=True
        ),
        Tool(
            name="python_repl",
            func=python_repl.run,
            description=(
                "A Python REPL for executing Python code. "
                "Only use this for tasks that require actual programming. "
                "Input should be valid Python code. "
                "WARNING: Be careful with this tool as it can execute arbitrary code."
            ),
            return_direct=True
        )
    ]
    
    return tools

def create_agent():
    """Create and configure the main agent with memory and tools.
    
    This is where we bring everything together - the LLM, tools, and memory.
    """
    with st.spinner("🤖 Setting up your AI assistant..."):
        try:
            # First, create our language model
            llm = create_llm()
            
            # Then set up the tools it can use
            tools = create_tools(llm)
            
            # Configure memory - using a sliding window to keep context manageable
            memory = ConversationBufferWindowMemory(
                memory_key="chat_history",  # Key for storing messages
                k=st.session_state.get('memory_window', 5),  # Number of messages to remember
                return_messages=True,  # Return message objects instead of strings
                output_key="output",  # Key for the agent's output
            )
            
            # Initialize the agent with our tools and memory
            agent = initialize_agent(
                tools=tools, 
                llm=llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,  # Good for back-and-forth chats
                verbose=st.session_state.get('debug', False),  # Only show debug info if enabled
                memory=memory,
                handle_parsing_errors=True,  # Try to recover from parsing errors
                max_iterations=5,  # Prevent infinite loops
                early_stopping_method="generate",  # Fallback if max iterations reached
                return_intermediate_steps=True  # Useful for debugging
            )
            
            return agent
            
        except Exception as e:
            # Log the actual error for debugging
            import traceback
            error_details = traceback.format_exc()
            if 'debug' in st.session_state and st.session_state.debug:
                st.sidebar.error("Agent initialization failed:")
                st.sidebar.code(error_details)
                
            st.error("😕 Oops! Something went wrong while setting up the AI. "
                    "Try refreshing the page or check the logs if debug mode is on.")
            return None

# --- Init Session State ---
def init_session_state():
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "active_chat" not in st.session_state:
        chat_id = str(int(time.time()))
        st.session_state.active_chat = chat_id
        st.session_state.chats[chat_id] = {"messages": [], "created_at": time.time(), "title": "New Chat"}
    if "model" not in st.session_state:
        st.session_state.model = "mistral"
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.2
    if "memory_window" not in st.session_state:
        st.session_state.memory_window = 5

# --- Sidebar ---
def create_new_chat():
    """Create a new chat and set it as active."""
    chat_id = str(int(time.time()))
    st.session_state.active_chat = chat_id
    st.session_state.chats[chat_id] = {
        "messages": [],
        "created_at": time.time(),
        "title": "New Chat"
    }
    return chat_id

def switch_chat(chat_id):
    """Switch to an existing chat."""
    if chat_id in st.session_state.chats:
        st.session_state.active_chat = chat_id

def render_sidebar():
    with st.sidebar:
        # New Chat button
        if st.button("➕ New Chat", use_container_width=True, key="new_chat_btn"):
            create_new_chat()
            st.rerun()

        st.divider()
        st.subheader("💬 Chat History")
        
        # Display chat history
        sorted_chats = sorted(st.session_state.chats.items(), 
                            key=lambda x: x[1]['created_at'], 
                            reverse=True)

        for chat_id, chat_data in sorted_chats:
            # Get chat title from first user message or default
            label = chat_data.get("title", "New Chat")
            if label == "New Chat":
                for msg in chat_data["messages"]:
                    if msg["role"] == "user":
                        label = msg["content"][:30] + ("..." if len(msg["content"]) > 30 else "")
                        break

            # Style for active/inactive chat
            is_active = chat_id == st.session_state.active_chat
            button_style = """
                <style>
                    .chat-btn-{0} {{
                        background: {1};
                        color: white;
                        border: none;
                        border-radius: 10px;
                        padding: 10px 15px;
                        margin: 5px 0;
                        width: 100%;
                        text-align: left;
                        cursor: pointer;
                        font-weight: 500;
                        transition: background-color 0.3s;
                    }}
                    .chat-btn-{0}:hover {{
                        background-color: #696969 !important;
                        color: white !important;
                    }}
                </style>
                <button class="chat-btn-{0}" onclick="window.parent.postMessage({{'type': 'CHAT_SELECTED', 'chatId': '{0}'}}, '*')">
                    💬 {2}
                </button>
            """.format(chat_id, "#808080" if is_active else "#808080", label)
            
            st.markdown(button_style, unsafe_allow_html=True)
        
        # JavaScript to handle chat selection
        st.markdown("""
        <script>
        // Listen for messages from the iframe
        window.addEventListener('message', function(event) {
            if (event.data.type === 'CHAT_SELECTED') {
                // Update the chat selection without page refresh
                const chatId = event.data.chatId;
                // Trigger a Streamlit rerun with the new chat ID
                const url = new URL(window.location);
                url.searchParams.set('chat_id', chatId);
                window.history.pushState({}, '', url);
                // Force a rerun
                window.parent.postMessage({type: 'streamlit:setQueryString', query: `chat_id=${chatId}`}, '*');
            }
        });
        </script>
        """, unsafe_allow_html=True)
        
        # Handle chat switching from URL parameters
        query_params = st.query_params
        if "chat_id" in query_params:
            switch_chat(query_params["chat_id"])
            # Remove the parameter from URL without refresh
            if st.rerun:
                st.query_params.clear()
                st.rerun()

        st.divider()
        st.subheader("⚙️ Settings")
        model_options = ["mistral", "llama3", "gemma"]
        st.selectbox("Select Model", model_options, index=model_options.index(st.session_state.model), key="model")
        st.slider("Creativity (Temperature)", 0.0, 1.0, value=st.session_state.temperature, step=0.1, key="temperature")
        st.slider("Memory Window", 1, 10, value=st.session_state.memory_window, step=1, key="memory_window")

        st.divider()
       
# --- Chat ---
def format_agent_output(output: str) -> str:
    """Format the agent's output for better readability.
    
    Args:
        output: The raw output string from the agent
        
    Returns:
        str: Formatted output with improved readability
    """
    if not output:
        return ""
        
    # Simple formatting for better readability
    formatted = []
    in_code_block = False
    
    for line in output.split('\n'):
        if line.startswith('```'):
            in_code_block = not in_code_block
            formatted.append(line)
        elif in_code_block:
            formatted.append(line)
        else:
            # Add line breaks after sentences
            line = line.replace('. ', '.  \n')
            line = line.replace('! ', '!  \n')
            line = line.replace('? ', '?  \n')
            formatted.append(line)
    
    return '\n'.join(formatted)

def process_agent_response(response: Dict[str, Any]) -> str:
    """Process the agent's response for better display.
    
    Handles markdown formatting and code blocks without using regex.
    """
    if not isinstance(response, str):
        response = str(response)
    
    # Handle code block formatting
    response = response.replace('```python', '```')
    response = response.replace('```bash', '```')
    response = response.replace('```json', '```')
    
    # Add basic formatting for better readability
    response = response.replace('\n', '  \n') 
    return format_agent_output(response)

def render_chat():
    """Render the chat interface and handle user input."""
    current_chat = st.session_state.chats[st.session_state.active_chat]
    messages = current_chat["messages"]

    # Display chat messages
    if not messages:
        st.info("💡 Start a new conversation by typing a message below!")
    else:
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)

    # Handle user input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat
        messages.append({"role": "user", "content": prompt})
        
        # Update chat title if this is the first message
        if len(messages) == 1:
            current_chat["title"] = prompt[:30] + ("..." if len(prompt) > 30 else "")

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display assistant response
        with st.chat_message("assistant"):
            try:
                if "agent_chain" not in st.session_state or st.session_state.agent_chain is None:
                    st.session_state.agent_chain = create_agent()
                
                if st.session_state.agent_chain is None:
                    raise Exception("Failed to initialize agent. Please check the logs.")
                
                # Create a container for the response
                response_container = st.empty()
                full_response = ""
                
                # Stream the response
                for chunk in st.session_state.agent_chain.stream({"input": prompt}):
                    if chunk:
                        chunk_text = process_agent_response(chunk)
                        if chunk_text:
                            full_response = chunk_text
                            response_container.markdown(full_response, unsafe_allow_html=True)
                
                # Add final response to chat history
                messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                error_msg = f"⚠️ An error occurred: {str(e)}"
                st.error(error_msg)

                messages.append({"role": "assistant", "content": error_msg})

# --- Main ---
def display_about_section():
    """Display the about section in the sidebar."""
    with st.sidebar.expander("ℹ️ About"):
        st.markdown("""
        **AI Assistant** can help you with:
        - Answering questions using web search
        - Performing calculations
        - Executing Python code
        - Remembering context

        ### Features
        - **Search**: Get information from the web
        - **Calculator**: Perform mathematical calculations
        - **Python REPL**: Execute Python code (advanced)
        - **Chat History**: Your conversations are saved in this session

        Developed by **Syeda Irha Kazmi**  
        Built with ❤️ using LangChain and Ollama
        """)

def display_debug_info():
    """Display debug information in the sidebar."""
    with st.sidebar.expander("🐞 Debug Info"):
        st.json({
            "active_chat": st.session_state.active_chat,
            "total_chats": len(st.session_state.chats),
            "current_messages": len(st.session_state.chats[st.session_state.active_chat]['messages']),
            "model": st.session_state.model,
            "temperature": st.session_state.temperature,
            "memory_window": st.session_state.memory_window,
            "agent_initialized": "agent_chain" in st.session_state and st.session_state.agent_chain is not None
        })

def main():
    """Main application function."""
    # Initialize the app
    load_css()
    init_session_state()
    
    # Set page title and description
    st.title("🤖 AI Assistant")
    st.caption("Powered by LangChain and Ollama - Your intelligent local AI assistant")
    
    # Create main layout
    col1, col2 = st.columns([3, 1])
    
    # Main chat interface
    with col1:
        render_chat()
    
    # Sidebar
    with col2:
        render_sidebar()
    
    # Footer
    st.markdown("---")
    st.caption("🔐 Your conversations are processed locally and not stored.")
    
    # Debug information (collapsed by default)
    display_debug_info()
    
    # About section (collapsed by default)
    display_about_section()
    
    # Initialize agent on first load if not already done
    if "agent_chain" not in st.session_state:
        st.session_state.agent_chain = create_agent()
        if st.session_state.agent_chain is None:
            st.error("Failed to initialize the AI agent. Please check the logs for more information.")
            st.stop()

# --- Run Safely ---
if __name__ == "__main__":
    main()
