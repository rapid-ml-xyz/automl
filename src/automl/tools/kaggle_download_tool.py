from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import os
from kaggle.api.kaggle_api_extended import KaggleApi


class KaggleDatasetInput(BaseModel):
    """Input schema for KaggleDownloadTool."""
    dataset_ref: str = Field(
        ...,
        description="The Kaggle dataset reference in 'owner/dataset-name' format"
    )
    download_path: str = Field(
        default="./datasets",
        description="Optional path where to download the dataset. Defaults to './datasets'"
    )


class KaggleDownloadTool(BaseTool):
    name: str = "Kaggle Dataset Downloader"
    description: str = (
        "Downloads a Kaggle dataset given its reference in 'owner/dataset-name' format. "
        "Returns the path where the dataset was downloaded."
    )
    args_schema: Type[BaseModel] = KaggleDatasetInput

    def _run(self, dataset_ref: str, download_path: str = "./datasets") -> str:
        try:
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

            return f"Dataset downloaded successfully to: {dataset_path}"

        except Exception as e:
            return f"Error downloading dataset: {str(e)}"
