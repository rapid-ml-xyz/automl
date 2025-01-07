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
    name: str = "YData Profile Report Tool"
    description: str = (
        "Analyzes a CSV dataset using YData Profiling (formerly pandas-profiling) "
        "and returns key metrics and insights from the profile report."
    )
    args_schema: Type[BaseModel] = YDataProfilerInput

    def _run(self, filepath: str) -> str:
        try:
            # Read the CSV file
            df = pd.read_csv(filepath)

            # Generate the profile report
            profile = ProfileReport(df, minimal=True)

            # Get the full report as JSON
            full_report = json.loads(profile.to_json())

            # Extract only the most important metrics
            summary = {
                "dataset_info": {
                    "number_of_variables": full_report.get("table", {}).get("n_var", 0),
                    "number_of_observations": full_report.get("table", {}).get("n", 0),
                    "missing_cells": full_report.get("table", {}).get("n_cells_missing", 0),
                    "missing_cells_percentage": full_report.get("table", {}).get("p_cells_missing", 0),
                },
                "variables": {}
            }

            # Extract key metrics for each variable
            for var_name, var_data in full_report.get("variables", {}).items():
                var_summary = {
                    "type": var_data.get("type", "unknown"),
                    "distinct_count": var_data.get("n_distinct", 0),
                    "missing_count": var_data.get("n_missing", 0),
                    "missing_percentage": var_data.get("p_missing", 0)
                }

                # Add type-specific statistics
                if var_data.get("type") in ["numeric", "integer", "float"]:
                    stats = {
                        "mean": var_data.get("mean", None),
                        "std": var_data.get("std", None),
                        "min": var_data.get("min", None),
                        "max": var_data.get("max", None)
                    }
                    # Only add non-None values
                    stats = {k: v for k, v in stats.items() if v is not None}
                    if stats:
                        var_summary.update(stats)

                elif var_data.get("type") == "categorical":
                    # Include top categories if available
                    value_counts = var_data.get("value_counts_without_nan", {})
                    if value_counts:
                        top_categories = dict(list(value_counts.items())[:5])
                        if top_categories:
                            var_summary["top_categories"] = top_categories

                summary["variables"][var_name] = var_summary

            # Include correlations if available
            correlations = full_report.get("correlations", {})
            if correlations:
                # Get the first correlation method available
                correlation_method = next(iter(correlations), None)
                if correlation_method:
                    corr_data = correlations[correlation_method]
                    correlation_pairs = []

                    # Extract correlations while handling potential missing or None values
                    for var1 in corr_data:
                        for var2 in corr_data.get(var1, {}):
                            if var1 < var2:  # Avoid duplicates
                                value = corr_data[var1].get(var2)
                                if value is not None:
                                    correlation_pairs.append({
                                        "variables": [var1, var2],
                                        "correlation": value
                                    })

                    # Sort by absolute correlation value and take top 10
                    if correlation_pairs:
                        correlation_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
                        summary["correlations"] = correlation_pairs[:10]

            return json.dumps(summary, indent=2)

        except Exception as e:
            return f"Error generating profile report: {str(e)}"
