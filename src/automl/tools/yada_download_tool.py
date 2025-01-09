from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from ydata_profiling import ProfileReport
import json
import pandas as pd


def remove_fields(data):
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            if key in ['histogram', 'length_histogram', 'histogram_length', 'character_counts',
                       'category_alias_values', 'block_alias_values', 'block_alias_char_counts',
                       'script_char_counts', 'category_alias_counts', 'category_alias_char_counts',
                       'package', 'value_counts_without_nan', 'value_counts_index_sorted']:
                continue
            cleaned[key] = remove_fields(value)
        return cleaned
    elif isinstance(data, list):
        return [remove_fields(item) for item in data]
    else:
        return data


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
            report = profile.to_json()
            report_json = json.loads(report)
            truncated_report_json = remove_fields(report_json)

            with open(output_path, 'w') as f:
                json.dump(truncated_report_json, f, indent=4, sort_keys=False)
            return f"Profile report successfully saved to {output_path}"

        except Exception as e:
            return f"Error generating profile report: {str(e)}"
