from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


class CsvPreviewInput(BaseModel):
    """Input schema for CsvPreviewTool."""
    file_path: str = Field(..., description="Path to the CSV file.")
    num_rows: int = Field(10, description="Number of rows to preview (default is 10).")


class CsvPreviewTool(BaseTool):
    name: str = "CSV Preview Tool"
    description: str = (
        "This tool reads a CSV file and provides a preview of its contents, "
        "including the header and a specified number of rows."
    )
    args_schema: Type[BaseModel] = CsvPreviewInput

    def _run(self, file_path: str, num_rows: int = 10) -> str:
        """Read the CSV file and return a preview."""
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            lines = content.split('\n')
            header = lines[0]
            preview_data = lines[1:num_rows + 1]
            return '\n'.join([header] + preview_data)
        except Exception as e:
            return f"An error occurred while reading the file: {str(e)}"
