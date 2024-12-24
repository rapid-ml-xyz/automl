from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import os
import re
from kaggle.api.kaggle_api_extended import KaggleApi


class KaggleDatasetInput(BaseModel):
    """Input schema for KaggleDownloadTool."""
    query: str = Field(
        ...,
        description="The text query that contains a Kaggle dataset reference"
    )
    download_path: str = Field(
        default="./datasets",
        description="Optional path where to download the dataset. Defaults to './datasets'"
    )


class KaggleDownloadTool(BaseTool):
    name: str = "Kaggle Dataset Processor"
    description: str = (
        "Extracts Kaggle dataset references from text, downloads the dataset, and retrieves its metadata. "
        "The tool can understand dataset references in natural language and will extract information like "
        "the dataset introduction and evaluation metrics. Returns all information in a structured format."
    )
    args_schema: Type[BaseModel] = KaggleDatasetInput

    def _extract_dataset_reference(self, text: str) -> str:
        """Extract Kaggle dataset reference from text"""
        pattern = r'[\w-]+/[\w-]+'
        matches = re.findall(pattern, text)

        if not matches:
            raise ValueError("Could not find dataset reference in format 'owner/dataset-name'")

        api = KaggleApi()
        api.authenticate()

        for match in matches:
            try:
                api.dataset_metadata(match)
                return match
            except:
                continue

        raise ValueError(f"No valid Kaggle dataset found in text: {text}")

    def _get_dataset_info(self, dataset_ref: str, api: KaggleApi) -> dict:
        """Get dataset metadata"""
        metadata = api.dataset_metadata(dataset_ref)
        data_card = metadata.description
        metrics = []

        # Extract metrics
        metric_keywords = ['accuracy', 'precision', 'recall', 'f1', 'mae', 'mse', 'rmse']
        metrics = [k for k in metric_keywords if k.lower() in data_card.lower()]

        # Get introduction (first paragraph)
        intro = data_card.split('\n\n')[0] if data_card else "No introduction available"

        return {
            "introduction": intro,
            "evaluation_metrics": metrics
        }

    def _run(self, query: str, download_path: str = "./datasets") -> str:
        try:
            # Extract dataset reference
            dataset_ref = self._extract_dataset_reference(query)

            # Initialize API
            api = KaggleApi()
            api.authenticate()

            # Create download directory
            dataset_path = f"{download_path}/{dataset_ref.replace('/', '_')}"
            os.makedirs(dataset_path, exist_ok=True)

            # Download dataset
            api.dataset_download_files(
                dataset=dataset_ref,
                path=dataset_path,
                unzip=True
            )

            # Get additional information
            info = self._get_dataset_info(dataset_ref, api)

            # Return structured response
            response = {
                "dataset_reference": dataset_ref,
                "download_path": dataset_path,
                "introduction": info["introduction"],
                "evaluation_metrics": info["evaluation_metrics"]
            }

            return str(response)

        except Exception as e:
            return f"Error processing dataset: {str(e)}"
