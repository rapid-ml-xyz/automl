import streamlit as st
from typing import Dict, Any
from .eda_flow import EDAFlow


def setup_page():
    """Initialize Streamlit page configuration"""
    st.set_page_config(
        page_title="KaggleAutoML Explorer",
        page_icon="📊",
        layout="wide"
    )


def run_data_acquisition(action: str) -> Dict[str, Any]:
    """Run the initial data acquisition and processing

    Args:
        action (str): The analysis action/topic

    Returns:
        Dict[str, Any]: Flow state containing dataset information
    """
    inputs = {'topic': action}
    eda_flow = EDAFlow()
    with st.spinner("Running data acquisition..."):
        return eda_flow.kickoff(inputs=inputs)


class StreamlitInterface:
    """Streamlit interface for KaggleAutoML exploration"""

    def __init__(self):
        setup_page()

    def explore(self, action: str):
        """Main exploration interface

        Args:
            action (str): The analysis action/topic
        """
        st.title(f"AutoML Explorer - {action}")

        # Configure sidebar
        st.sidebar.title("Controls")
        if st.sidebar.button("Reset"):
            st.session_state.clear()

        # Run initial flow if needed
        if 'initial_state' not in st.session_state:
            st.session_state.initial_state = run_data_acquisition(action)

        try:
            # Load and display JSON data
            if 'initial_state' in st.session_state:
                st.header("Acquired Data")
                st.json(st.session_state.initial_state)

        except Exception as e:
            st.error(f"Error during exploration: {str(e)}")
