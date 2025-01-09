import streamlit as st
from streamlit_chat import message
import json
from typing import Dict, Any, Tuple
from .eda_flow import EDAFlow


def setup_page():
    """Initialize Streamlit page configuration"""
    st.set_page_config(
        page_title="KaggleAutoML Explorer",
        page_icon="📊",
        layout="wide"
    )


def init_chat_state():
    """Initialize chat-related session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'generated' not in st.session_state:
        st.session_state.generated = []
    if 'past' not in st.session_state:
        st.session_state.past = []
    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0


def run_data_acquisition(action: str) -> Tuple[str, Dict[str, Any]]:
    """Run the initial data acquisition and processing

    Args:
        action (str): The analysis action/topic

    Returns:
        Tuple[str, Dict[str, Any]]: Tuple containing (json_path, loaded_json_data)
    """
    inputs = {'topic': action}
    eda_flow = EDAFlow()

    with st.spinner("Running data acquisition..."):
        json_path = eda_flow.kickoff(inputs=inputs).raw

        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            return json_path, json_data
        except Exception as e:
            st.error(f"Error loading JSON from {json_path}: {str(e)}")
            return json_path, {}


class StreamlitInterface:
    """Streamlit interface for KaggleAutoML exploration"""

    def __init__(self):
        setup_page()
        init_chat_state()

    def display_chat_interface(self):
        """Display the chat interface using streamlit_chat"""
        st.header("Chat Interface")

        # Container for chat messages
        chat_container = st.container()

        # Display chat messages
        with chat_container:
            for i in range(len(st.session_state.generated)):
                message(st.session_state.past[i], is_user=True, key=f"user_{i}")
                message(st.session_state.generated[i], key=f"bot_{i}")

        # Chat input with dynamic key
        input_key = f"user_input_{st.session_state.input_key}"
        user_input = st.text_input("Ask about the data:", key=input_key, placeholder="Type your question here...")

        if user_input:
            # Add user message to history
            st.session_state.past.append(user_input)

            # Generate bot response (placeholder - customize as needed)
            bot_response = f"I understand your question about: {user_input}"
            st.session_state.generated.append(bot_response)

            # Increment key for next input
            st.session_state.input_key += 1

            # Rerun to update the chat display
            st.rerun()

    def explore(self, action: str):
        """Main exploration interface

        Args:
            action (str): The analysis action/topic
        """
        st.title(f"AutoML Explorer - {action}")

        st.sidebar.title("Controls")
        if st.sidebar.button("Reset"):
            st.session_state.clear()

        try:
            if 'initial_state' in st.session_state:
                st.header("Acquired Data")
                st.text(f"JSON Path: {st.session_state.initial_state['path']}")
                st.json(st.session_state.initial_state['data'])

            # Add chat interface below the data display
            self.display_chat_interface()

        except Exception as e:
            st.error(f"Error during exploration: {str(e)}")
