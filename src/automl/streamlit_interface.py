import streamlit as st
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

        # Load JSON data from the returned path
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

    def explore(self, action: str):
        """Main exploration interface

        Args:
            action (str): The analysis action/topic
        """
        st.title(f"AutoML Explorer - {action}")

        st.sidebar.title("Controls")
        if st.sidebar.button("Reset"):
            st.session_state.clear()

        if 'initial_state' not in st.session_state:
            json_path, json_data = run_data_acquisition(action)
            st.session_state.initial_state = {
                'path': json_path,
                'data': json_data
            }

        try:
            if 'initial_state' in st.session_state:
                st.header("Acquired Data")
                st.text(f"JSON Path: {st.session_state.initial_state['path']}")
                st.json(st.session_state.initial_state['data'])

        except Exception as e:
            st.error(f"Error during exploration: {str(e)}")
