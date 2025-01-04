from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel
import os


class PWDInput(BaseModel):
    """Input schema for PWDTool."""
    pass  # No input needed for pwd operation


class PWDTool(BaseTool):
    name: str = "PWD Tool"
    description: str = (
        "Returns the absolute path of the current working directory. "
        "Useful for determining the exact location in the filesystem."
    )
    args_schema: Type[BaseModel] = PWDInput

    def _run(self) -> str:
        try:
            current_path = os.getcwd()
            return f"Current working directory: {current_path}"
        except Exception as e:
            return f"Error getting current working directory: {str(e)}"
