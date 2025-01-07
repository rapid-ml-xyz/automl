from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import pandas as pd
from ydata_profiling import ProfileReport
import json


class YDataProfilerInput(BaseModel):
    """Input schema for DataProfilerTool."""
    filepath: str = Field(..., description="Path to the CSV file to analyze")


class YDataProfilerTool(BaseTool):
    name: str = "Data Profiler Analysis Tool"
    description: str = (
        "Analyzes a CSV dataset and generates a comprehensive profile report including "
        "statistics, correlations, and data quality metrics. The tool reads a CSV file "
        "and returns a JSON report containing detailed analysis of all variables, "
        "their distributions, correlations, and potential issues in the dataset."
    )
    args_schema: Type[BaseModel] = YDataProfilerInput

    def _run(self, filepath: str) -> str:
        try:
            df = pd.read_csv(filepath)
            profile = ProfileReport(
                df,
                title="Mental Health Dataset Analysis",
                explorative=True,
                correlations={
                    "spearman": {"calculate": True},
                    "pearson": {"calculate": True},
                    "cramers": {"calculate": True},
                },
                missing_diagrams={
                    "bar": False,
                    "matrix": True,
                    "heatmap": True,
                }
            )

            report_json = profile.to_json()
            report_dict = json.loads(report_json)
            formatted_report = json.dumps(report_dict, indent=2)
            return formatted_report

        except Exception as e:
            return f"Error generating profile report: {str(e)}"
