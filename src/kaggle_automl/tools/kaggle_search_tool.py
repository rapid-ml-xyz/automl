from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from kaggle.api.kaggle_api_extended import KaggleApi


class KaggleSearchInput(BaseModel):
    """Input schema for KaggleNotebookTool."""
    user_task: str = Field(
        ...,
        description="The specific task to search for in notebooks"
    )
    user_domain: str = Field(
        ...,
        description="The domain area to search within"
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of notebooks to return"
    )


class KaggleSearchTool(BaseTool):
    name: str = "Kaggle Notebook Searcher"
    description: str = (
        "Searches Kaggle notebooks based on task and domain keywords. "
        "Returns relevant notebook information including titles, URLs, and descriptions."
    )
    args_schema: Type[BaseModel] = KaggleSearchInput

    def _run(self, user_task: str, user_domain: str, max_results: int = 5) -> str:
        try:
            api = KaggleApi()
            api.authenticate()

            search_term = f"{user_task} {user_domain}"
            return api.kernels_list(search=search_term, page_size=max_results)

        except Exception as e:
            return f"Error searching Kaggle notebooks: {str(e)}"
