from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import subprocess
import os


class FileExecutorInput(BaseModel):
    """Input schema for FileExecutorTool."""
    file_path: str = Field(..., description="Path of the Python file to execute.")


class FileExecutorTool(BaseTool):
    name: str = "File Executor Tool"
    description: str = (
        "A tool to execute a Python script from a given file path. "
        "Provide the full path of the Python file as an argument."
    )
    args_schema: Type[BaseModel] = FileExecutorInput

    def _run(self, file_path: str) -> str:
        if not os.path.isfile(file_path):
            return f"Error: The file '{file_path}' does not exist."

        try:
            result = subprocess.run(['python', file_path], capture_output=True, text=True)
            output = result.stdout
            error = result.stderr

            if result.returncode == 0:
                return f"Output:\n{output}"
            else:
                return f"Error:\n{error}"
        except Exception as e:
            return f"An error occurred: {str(e)}"
