from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import arxiv


class ArxivSearchInput(BaseModel):
    """Input schema for ArxivTool."""
    task_kw: str = Field(
        ...,
        description="Task-related keywords for the search"
    )
    domain_kw: str = Field(
        ...,
        description="Domain-related keywords for the search"
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of papers to return"
    )


class ArxivSearchTool(BaseTool):
    name: str = "arXiv Paper Searcher"
    description: str = (
        "Searches arXiv for relevant research papers. "
        "Returns paper details including titles, authors, abstracts, and PDF URLs."
    )
    args_schema: Type[BaseModel] = ArxivSearchInput

    def _run(self, task_kw: str, domain_kw: str, max_results: int = 5) -> str:
        try:
            client = arxiv.Client()
            search_query = arxiv.Search(
                query=f"{task_kw} {domain_kw}",
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )

            results = []
            for paper in client.results(search_query):
                results.append({
                    'title': paper.title,
                    'authors': [author.name for author in paper.authors],
                    'abstract': paper.summary,
                    'url': paper.pdf_url,
                    'published': paper.published.strftime("%Y-%m-%d")
                })
            return str(results)
        except Exception as e:
            return f"Error searching arXiv: {str(e)}"
