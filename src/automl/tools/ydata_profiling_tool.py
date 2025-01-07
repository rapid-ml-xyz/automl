from crewai.tools import BaseTool
from typing import Type
import pandas as pd
from ydata_profiling import ProfileReport
from pydantic import BaseModel, Field


class YDataProfilingInput(BaseModel):
    """Input schema for DataProfilingTool."""
    dataframe: str = Field(
        ...,
        description="String representation of a pandas DataFrame to be profiled. Should be convertible to a DataFrame using pd.read_json()."
    )
    minimal: bool = Field(
        default=False,
        description="Whether to generate a minimal report (faster) or a complete one."
    )
    title: str = Field(
        default="Data Profiling Report",
        description="Title for the profiling report."
    )


class YDataProfilingTool(BaseTool):
    name: str = "Data Profiling Tool"
    description: str = (
        "A powerful tool for generating comprehensive data profiling reports. "
        "It analyzes datasets and provides detailed statistics, distributions, "
        "correlations, and quality metrics. The tool accepts a DataFrame and "
        "returns an HTML report with insights about the data."
    )
    args_schema: Type[BaseModel] = YDataProfilingInput

    def _run(self, dataframe: str, minimal: bool = False, title: str = "Data Profiling Report") -> str:
        try:
            df = pd.read_json(dataframe)
            profile = ProfileReport(
                df,
                minimal=minimal,
                title=title,
                explorative=True
            )
            output_file = f"profile_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"
            profile.to_file(output_file)
            return f"Profile report successfully generated and saved to {output_file}"

        except Exception as e:
            return f"Error generating profile report: {str(e)}"
