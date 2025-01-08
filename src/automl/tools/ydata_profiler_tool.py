from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import json


class YDataProfilerInput(BaseModel):
    """Input schema for DataProfilerTool."""
    json_filepath: str = Field(..., description="Path to the YData profile JSON file to process")


class YDataProfilerTool(BaseTool):
    name: str = "YData Profile Report Processor"
    description: str = (
        "Processes a YData Profiling JSON report and returns key metrics and insights. "
        "Produces condensed output suitable for LLMs."
    )
    args_schema: Type[BaseModel] = YDataProfilerInput

    def _run(self, json_filepath: str) -> str:
        try:
            with open(json_filepath, 'r') as f:
                full_report = json.load(f)

            summary = {
                "dataset_info": {
                    "number_of_variables": full_report.get("table", {}).get("n_var", 0),
                    "number_of_observations": full_report.get("table", {}).get("n", 0),
                    "missing_cells": full_report.get("table", {}).get("n_cells_missing", 0),
                    "missing_cells_percentage": full_report.get("table", {}).get("p_cells_missing", 0),
                },
                "variables": {}
            }

            def format_number(num):
                if isinstance(num, (int, float)):
                    return round(num, 4) if isinstance(num, float) else num
                return num

            max_variables = 20
            processed_vars = 0

            for var_name, var_data in full_report.get("variables", {}).items():
                if processed_vars >= max_variables:
                    break

                var_summary = {
                    "type": var_data.get("type", "unknown"),
                    "distinct_count": var_data.get("n_distinct", 0),
                    "missing_count": var_data.get("n_missing", 0),
                    "missing_percentage": format_number(var_data.get("p_missing", 0))
                }

                if var_data.get("type") in ["numeric", "integer", "float"]:
                    stats = {
                        "mean": format_number(var_data.get("mean", None)),
                        "std": format_number(var_data.get("std", None)),
                        "min": format_number(var_data.get("min", None)),
                        "max": format_number(var_data.get("max", None))
                    }
                    stats = {k: v for k, v in stats.items() if v is not None}
                    if stats:
                        var_summary.update(stats)

                elif var_data.get("type") == "categorical":
                    value_counts = var_data.get("value_counts_without_nan", {})
                    if value_counts:
                        top_categories = dict(list(value_counts.items())[:3])
                        if top_categories:
                            var_summary["top_categories"] = {
                                k: format_number(v) for k, v in top_categories.items()
                            }

                summary["variables"][var_name] = var_summary
                processed_vars += 1

            correlations = full_report.get("correlations", {})
            if correlations:
                correlation_method = next(iter(correlations), None)
                if correlation_method:
                    corr_data = correlations[correlation_method]
                    correlation_pairs = []

                    for var1 in corr_data:
                        for var2 in corr_data.get(var1, {}):
                            if var1 < var2:
                                value = corr_data[var1].get(var2)
                                if value is not None:
                                    correlation_pairs.append({
                                        "variables": [var1, var2],
                                        "correlation": format_number(value)
                                    })

                    if correlation_pairs:
                        correlation_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
                        summary["correlations"] = correlation_pairs[:10]

            if processed_vars >= max_variables:
                summary["note"] = f"Output limited to {max_variables} variables for context size"

            return json.dumps(summary, indent=2)

        except Exception as e:
            return f"Error processing profile report: {str(e)}"
