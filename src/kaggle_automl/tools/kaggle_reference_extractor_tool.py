from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from kaggle.api.kaggle_api_extended import KaggleApi


class DatasetReferenceInput(BaseModel):
    """Input schema for KaggleReferenceExtractorTool."""
    query: str = Field(
        ...,
        description="The text query that contains a Kaggle dataset reference"
    )


class KaggleReferenceExtractorTool(BaseTool):
    name: str = "Kaggle Reference Extractor"
    description: str = """
    Extracts Kaggle dataset references from text. You should look for patterns like 'owner/dataset-name' 
    in the text and return only that reference. For example, from 'I want to analyze netflix-user/netflix-movies dataset', 
    you should return 'netflix-user/netflix-movies'.
    """
    args_schema: Type[BaseModel] = DatasetReferenceInput

    def _extract_dataset_reference(self, text: str) -> str:
        """Extract Kaggle dataset reference from text"""
        # Validation with Kaggle API
        api = KaggleApi()
        api.authenticate()

        # The LLM's response will be the dataset reference
        try:
            api.dataset_metadata(text)
            return text
        except:
            raise ValueError(f"Invalid Kaggle dataset reference: {text}")

    def _run(self, query: str) -> str:
        try:
            return self._extract_dataset_reference(query)
        except Exception as e:
            return f"Error extracting dataset reference: {str(e)}"
