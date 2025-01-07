from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import os
from pathlib import Path


class FileOperationInput(BaseModel):
    """Input schema for FileOperationTool."""
    old_path: str = Field(
        ...,
        description="The current path and filename"
    )
    new_path: str = Field(
        ...,
        description="The new path and filename"
    )
    operation: str = Field(
        default="rename",
        description="The file operation to perform (currently supports 'rename')"
    )


class FileOperationTool(BaseTool):
    name: str = "File Operation Tool"
    description: str = (
        "Performs file operations such as renaming files. "
        "Handles both files and directories, with path validation and error handling."
    )
    args_schema: Type[BaseModel] = FileOperationInput

    def _run(self, old_path: str, new_path: str, operation: str = "rename") -> str:
        try:
            # Convert to Path objects for better path handling
            old_path = Path(old_path)
            new_path = Path(new_path)

            # Validate paths
            if not old_path.exists():
                return f"Error: Source path '{old_path}' does not exist"

            if new_path.exists():
                return f"Error: Destination path '{new_path}' already exists"

            # Ensure parent directory of new path exists
            new_path.parent.mkdir(parents=True, exist_ok=True)

            if operation.lower() == "rename":
                os.rename(old_path, new_path)
                return f"Successfully renamed '{old_path}' to '{new_path}'"
            else:
                return f"Error: Unsupported operation '{operation}'"

        except PermissionError:
            return f"Error: Permission denied when accessing '{old_path}' or '{new_path}'"
        except Exception as e:
            return f"Error performing file operation: {str(e)}"
