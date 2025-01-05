import os
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Any, Optional, Type


class FixedDirectoryReadToolSchema(BaseModel):
    """Input for DirectoryReadTool."""
    pass


class DirectoryReadToolInput(BaseModel):
    """Input for DirectoryReadTool."""
    directory: str = Field(..., description="Mandatory directory to list content")


class DirectoryReadTool(BaseTool):
    name: str = "List files in directory"
    description: str = (
        "A tool that can be used to recursively list a directory's content."
    )
    args_schema: Type[BaseModel] = DirectoryReadToolInput
    directory: Optional[str] = None

    def __init__(self, directory: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if directory is not None:
            self.directory = directory
            self.description = f"A tool that can be used to list {directory}'s content."
            self.args_schema = FixedDirectoryReadToolSchema
        self._generate_description()

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        directory = kwargs.get("directory", self.directory)
        if directory and directory.endswith("/"):
            directory = directory[:-1]

        files_list = []
        for root, _, files in os.walk(directory):
            for filename in files:
                rel_path = os.path.relpath(os.path.join(root, filename), directory)
                normalized_path = os.path.normpath(rel_path).replace("\\", "/")
                full_path = f"{directory}/{normalized_path}"
                files_list.append(full_path)

        files_list.sort()

        files = "\n- ".join(files_list)
        return f"File paths:\n- {files}"
