from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import requests


class PapersWithCodeSearchInput(BaseModel):
    """Input schema for PapersWithCodeTool."""
    user_area: str = Field(
        ...,
        description="The research area to search within"
    )
    user_task: str = Field(
        ...,
        description="The specific task to search for"
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of papers to return"
    )


class PapersWithCodeSearchTool(BaseTool):
    name: str = "Papers With Code Searcher"
    description: str = (
        "Searches Papers With Code for relevant papers and implementations. "
        "Returns paper details including titles, abstracts, and GitHub repositories."
    )
    args_schema: Type[BaseModel] = PapersWithCodeSearchInput

    def _run(self, user_area: str, user_task: str, max_results: int = 5) -> str:
        try:
            base_url = "https://paperswithcode.com/api/v1"
            search_term = f"{user_area} {user_task}"

            response = requests.get(
                f"{base_url}/papers/",
                params={"search": search_term, "items_per_page": max_results}
            )

            if response.status_code == 200:
                papers = response.json()
                results = []
                for paper in papers.get('results', []):
                    results.append({
                        'title': paper.get('title'),
                        'abstract': paper.get('abstract'),
                        'url': paper.get('url_pdf'),
                        'published': paper.get('published')
                    })
                return str(results)
            return "No results found"
        except Exception as e:
            return f"Error searching Papers With Code: {str(e)}"
