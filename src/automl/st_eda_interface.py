import streamlit as st
import plotly.graph_objects as go
from .eda_flow import DownloadFlow, ExplorationFlow, VisualizationFlow
import json


def setup_page():
    """Initialize Streamlit page configuration"""
    st.set_page_config(
        page_title="RapidML EDA",
        page_icon="📊",
        layout="wide"
    )


def init_state():
    """Initialize session state variables"""
    if 'reports' not in st.session_state:
        st.session_state.reports = []
    if 'visualizations' not in st.session_state:
        st.session_state.visualizations = []
    if 'data_initialized' not in st.session_state:
        st.session_state.data_initialized = False
    if 'json_path' not in st.session_state:
        st.session_state.json_path = None


def run_download_flow(action: str) -> str:
    inputs = {'topic': action}
    download_flow = DownloadFlow()

    with st.spinner("Running data acquisition..."):
        json_path = download_flow.kickoff(inputs=inputs).raw
        print(json_path)

    return json_path


def run_reporting_flow(user_input) -> str:
    inputs = {
        'json_filepath': st.session_state.json_path,
        'user_input': user_input
    }
    exploration_flow = ExplorationFlow()

    with st.spinner("Running data exploration..."):
        summary = exploration_flow.kickoff(inputs=inputs).raw
        print(summary)

    return summary


def run_visualization_flow(user_input):
    inputs = {
        'json_filepath': st.session_state.json_path,
        'user_input': user_input
    }
    visualization_flow = VisualizationFlow()

    with st.spinner("Running visualization creation..."):
        try:
            raw_fig_json = visualization_flow.kickoff(inputs=inputs).raw
            if isinstance(raw_fig_json, str):
                fig_dict = json.loads(raw_fig_json)
            else:
                fig_dict = raw_fig_json

            fig = go.Figure(fig_dict)
            return [fig]
        except json.JSONDecodeError as e:
            st.error(f"Error parsing visualization JSON: {str(e)}")
            return []
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            return []


class StreamlitInterface:
    def __init__(self):
        setup_page()
        init_state()

    def display_text_reports(self, container):
        container.header("Analysis Reports")

        for i, report in enumerate(reversed(st.session_state.reports)):
            report_num = len(st.session_state.reports) - i
            with container.expander(f"Report {report_num}", expanded=True):
                st.markdown(report)

    def display_visualizations(self, container):
        container.header("Visualizations")

        for i, viz in enumerate(reversed(st.session_state.visualizations)):
            viz_num = len(st.session_state.visualizations) - i
            with container.expander(f"Visualization {viz_num}", expanded=True):
                try:
                    st.plotly_chart(viz, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying visualization {viz_num}: {str(e)}")

    def process_query(self, query: str):
        if not query:
            return

        report = run_reporting_flow(query)
        st.session_state.reports.append(report)

        visualizations = run_visualization_flow(query)
        if visualizations:
            st.session_state.visualizations.extend(visualizations)

    def explore(self, action: str):
        """Main exploration interface"""
        st.title("Exploratory Data Analysis")
        st.subheader(action)

        st.sidebar.title("Controls")
        if st.sidebar.button("Reset Analysis"):
            st.session_state.reports = []
            st.session_state.visualizations = []
            st.session_state.data_initialized = False
            st.session_state.json_path = None

        try:
            if not st.session_state.data_initialized:
                st.session_state.json_path = run_download_flow(action)
                st.session_state.data_initialized = True

            query = st.text_input(
                "Ask about the data:",
                placeholder="Type your question here...",
                help="Enter your analysis question"
            )

            if query:
                self.process_query(query)

            left_col, right_col = st.columns(2)
            with left_col:
                self.display_text_reports(left_col)

            with right_col:
                self.display_visualizations(right_col)

        except Exception as e:
            st.error(f"Error during exploration: {str(e)}")
