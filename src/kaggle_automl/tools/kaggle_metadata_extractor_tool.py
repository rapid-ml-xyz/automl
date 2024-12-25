from crewai.tools import BaseTool
from typing import Type
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import requests
import json
import os


class KaggleMetadataInput(BaseModel):
    """Input schema for KaggleMetadataExtractorTool."""
    dataset_ref: str = Field(
        ...,
        description="The Kaggle dataset reference in 'owner/dataset-name' format"
    )
    dataset_path: str = Field(
        ...,
        description="The path where the dataset is downloaded"
    )


class KaggleMetadataExtractorTool(BaseTool):
    name: str = "Kaggle Metadata Extractor"
    description: str = (
        "Extracts problem statement and metadata from a Kaggle dataset using its reference."
    )
    args_schema: Type[BaseModel] = KaggleMetadataInput

    def _run(self, dataset_ref: str, dataset_path: str) -> str:
        try:
            load_dotenv()
            url = f"https://www.kaggle.com/api/v1/datasets/view/{dataset_ref}"
            headers = {
                'Authorization': os.getenv("KAGGLE_API_KEY")
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            metadata = response.json()
            research_package = {
                "dataset": {
                    "reference": dataset_ref,
                    "local_path": dataset_path,
                    "metadata": metadata
                }
            }

            return json.dumps(research_package, indent=2)

        except requests.exceptions.RequestException as e:
            return f"Error making API request: {str(e)}"
        except Exception as e:
            return f"Error extracting metadata: {str(e)}"
