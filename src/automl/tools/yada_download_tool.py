from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import pandas as pd
from ydata_profiling import ProfileReport


class YDataDownloadInput(BaseModel):
    """Input schema for DataProfilerTool."""
    filepath: str = Field(..., description="Path to the CSV file to analyze")
    output_path: str = Field(..., description="Path where the JSON report will be saved")


class YDataDownloadTool(BaseTool):
    name: str = "YData Profile Report Tool"
    description: str = (
        "Analyzes a CSV dataset using YData Profiling (formerly pandas-profiling) "
        "and saves the complete profile report as a JSON file locally. "
        "Supports custom column names for CSV parsing."
    )
    args_schema: Type[BaseModel] = YDataDownloadInput

    def _run(self, filepath: str, output_path: str) -> str:
        try:
            df = pd.read_csv(filepath)
            profile = ProfileReport(df, minimal=True)
            report_json = profile.to_json()

            with open(output_path, 'w') as f:
                f.write(report_json)
            return f"Profile report successfully saved to {output_path}"

        except Exception as e:
            return f"Error generating profile report: {str(e)}"
