from crewai.tools import BaseTool
from typing import Type, Dict
from pydantic import BaseModel, Field
from huggingface_hub import HfApi, ModelCard
import json
import requests


class ModelSearchInput(BaseModel):
    """Input schema for searching models."""
    query: str = Field(
        ...,
        description="Search query describing the problem or approach you're looking for"
    )
    limit: int = Field(
        default=5,
        description="Maximum number of models to return"
    )


class HuggingFaceSearchTool(BaseTool):
    """Tool for searching Hugging Face models and fetching model cards."""
    name: str = "Hugging Face Search Tool"
    description: str = (
        "Searches Hugging Face for models and fetches detailed model cards. "
        "Can either search for new models or get details for specific model IDs."
    )
    args_schema: Type[BaseModel] = ModelSearchInput

    def __init__(self):
        super().__init__()
        self._api = HfApi()

    def _sanitize_model_id(self, model_id: str) -> str:
        return model_id.rstrip('.')

    def _request_model_read_me(self, model_id: str) -> str:
        url = f"https://huggingface.co/{model_id}/resolve/main/README.md"
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            return ""

    def _get_model_card(self, model_id: str) -> Dict:
        try:
            sanitized_model_id = self._sanitize_model_id(model_id)
            model_info = self._api.model_info(sanitized_model_id)

            try:
                card = ModelCard.load(sanitized_model_id)
                card_data = card.data.to_dict()
            except Exception:
                card_data = {}

            model_read_me = self._request_model_read_me(sanitized_model_id)

            return {
                "model_id": sanitized_model_id,
                "model_info": {
                    "downloads": model_info.downloads,
                    "likes": model_info.likes,
                    "tags": model_info.tags,
                    "pipeline_tag": model_info.pipeline_tag,
                    "last_modified": str(model_info.lastModified),
                    "author": model_info.author,
                },
                "model_card_metadata": card_data,
                "model_read_me": model_read_me
            }

        except Exception as e:
            return {
                "model_id": model_id,
                "error": str(e)
            }

    def _get_model_cards(self, models) -> str:
        print(models)
        model_ids = models

        model_cards = []
        for model_id in model_ids:
            model_card = self._get_model_card(model_id)
            model_cards.append(model_card)

        return json.dumps(model_cards, indent=2)

    def _run(self, query: str, limit: int = 5) -> str:
        try:
            models = self._api.list_models(
                search=query,
                limit=limit
            )

            model_ids = [model.id for model in models]
            return self._get_model_cards(model_ids)

        except Exception as e:
            error_response = {
                "error": str(e),
                "results": []
            }
            return json.dumps(error_response, indent=2)
