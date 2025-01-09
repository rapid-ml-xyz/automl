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

    def display_chat_interface(self, container):
        """Display the chat interface using streamlit_chat"""
        container.header("Chat Interface")

        # Container for chat messages
        chat_container = container.container()

        # Display chat messages in a scrollable container
        with chat_container:
            # Add CSS to control chat container height
            st.markdown("""
                <style>
                    .stChatMessageContent {
                        max-width: 100%;
                    }
                    .stTextInput {
                        position: sticky;
                        bottom: 0;
                        background-color: white;
                    }
                </style>
            """, unsafe_allow_html=True)

            # Display messages
            for i in range(len(st.session_state.generated)):
                message(st.session_state.past[i], is_user=True, key=f"user_{i}")
                message(st.session_state.generated[i], key=f"bot_{i}")

        # Chat input with dynamic key
        input_key = f"user_input_{st.session_state.input_key}"
        user_input = container.text_input("Ask about the data:", key=input_key, placeholder="Type your question here...")

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

    def display_visualization_area(self, container):
        """Display the visualization area (placeholder for future implementations)"""
        container.header("Visualizations")

        # Placeholder for future visualizations
        container.markdown("""
            ### Visualization Area
            This area will contain:
            - Data visualizations
            - Interactive charts
            - Analysis results
        """)

        # Example placeholder for a visualization
        with container.container():
            st.markdown("Visualization placeholder")

    def explore(self, action: str):
        """Main exploration interface

        Args:
            action (str): The analysis action/topic
        """
        st.title(f"Exploratory Data Analysis - {action}")

        # Sidebar controls
        st.sidebar.title("Controls")
        if st.sidebar.button("Reset"):
            st.session_state.clear()

        try:
            # Create two columns for the layout
            chat_col, viz_col = st.columns(2)

            # Display chat interface in left column
            with chat_col:
                self.display_chat_interface(chat_col)

            # Display visualization area in right column
            with viz_col:
                self.display_visualization_area(viz_col)

        except Exception as e:
            st.error(f"Error during exploration: {str(e)}")
