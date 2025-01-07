from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import os
from kaggle.api.kaggle_api_extended import KaggleApi


class KaggleSubmissionInput(BaseModel):
    """Input schema for KaggleSubmissionTool."""
    competition_name: str = Field(
        ...,
        description="The Kaggle competition name (e.g., 'titanic')"
    )
    submission_file: str = Field(
        ...,
        description="The path to the submission file (e.g., './submissions/submission.csv')"
    )
    message: str = Field(
        default="Kaggle submission",
        description="Optional message to accompany the submission"
    )


class KaggleSubmissionTool(BaseTool):
    name: str = "Kaggle Submission Tool"
    description: str = (
        "Submits a file to a specified Kaggle competition. "
        "Requires the competition name and the path to the submission file. "
        "Returns the result of the submission."
    )
    args_schema: Type[BaseModel] = KaggleSubmissionInput

    def _run(self, competition_name: str, submission_file: str, message: str = "Kaggle submission") -> str:
        try:
            api = KaggleApi()
            api.authenticate()

            if not os.path.exists(submission_file):
                return f"Error: Submission file '{submission_file}' does not exist."

            api.competition_submit(
                competition=competition_name,
                file_name=submission_file,
                message=message
            )

            return f"Submission to '{competition_name}' successful. File: {submission_file}, Message: '{message}'"

        except Exception as e:
            return f"Error submitting file: {str(e)}"
