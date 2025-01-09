import streamlit as st
from .eda_flow import DownloadFlow


def setup_page():
    """Initialize Streamlit page configuration"""
    st.set_page_config(
        page_title="KaggleAutoML Explorer",
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
    """Run the data download flow"""
    inputs = {'topic': action}
    eda_flow = DownloadFlow()

    with st.spinner("Running data acquisition..."):
        json_path = eda_flow.kickoff(inputs=inputs).raw
        print(json_path)

    return json_path


class StreamlitInterface:
    """Streamlit interface for KaggleAutoML exploration"""

    def __init__(self):
        setup_page()
        init_state()

    def display_text_reports(self, container):
        """Display text reports in the left column"""
        container.header("Analysis Reports")

        # Display reports in reverse chronological order
        for i, report in enumerate(reversed(st.session_state.reports)):
            report_num = len(st.session_state.reports) - i
            with container.expander(f"Report {report_num}", expanded=True):
                st.markdown(report)

    def display_visualizations(self, container):
        """Display visualizations in the right column"""
        container.header("Visualizations")

        # Display visualizations in reverse chronological order
        for i, viz in enumerate(reversed(st.session_state.visualizations)):
            viz_num = len(st.session_state.visualizations) - i
            with container.expander(f"Visualization {viz_num}", expanded=True):
                st.plotly_chart(viz, use_container_width=True)

    def process_query(self, query: str):
        """Process the user's query and generate reports/visualizations"""
        if not query:
            return

        # TODO: Replace with actual analysis logic
        # Example: Add a sample report
        report = f"Analysis results for query: {query}\n\n" + \
                 "This is a placeholder for the actual analysis report."
        st.session_state.reports.append(report)

        # Example: Add a sample visualization
        # You would replace this with actual visualization generation
        import plotly.express as px
        fig = px.line(x=[1, 2, 3], y=[1, 2, 3], title=f"Analysis for: {query}")
        st.session_state.visualizations.append(fig)

    def explore(self, action: str):
        """Main exploration interface"""
        st.title(f"Exploratory Data Analysis - {action}")

        # Sidebar controls
        st.sidebar.title("Controls")
        if st.sidebar.button("Reset Analysis"):
            st.session_state.reports = []
            st.session_state.visualizations = []
            st.session_state.data_initialized = False
            st.session_state.json_path = None

        try:
            # Only run data acquisition once
            if not st.session_state.data_initialized:
                st.session_state.json_path = run_download_flow(action)
                st.session_state.data_initialized = True

            # Single input box at the top
            query = st.text_input(
                "Ask about the data:",
                placeholder="Type your question here...",
                help="Enter your analysis question"
            )

            # Process the query when entered
            if query:
                self.process_query(query)

            # Split screen layout
            left_col, right_col = st.columns(2)

            # Display reports and visualizations in their respective columns
            with left_col:
                self.display_text_reports(left_col)

            with right_col:
                self.display_visualizations(right_col)

        except Exception as e:
            st.error(f"Error during exploration: {str(e)}")
