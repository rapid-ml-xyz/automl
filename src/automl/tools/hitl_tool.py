from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


class HumanInTheLoopInput(BaseModel):
    """Input schema for HumanValidationTool."""
    content: str = Field(
        ...,
        description="The content to be validated by the human reviewer."
    )
    validation_type: str = Field(
        ...,
        description="Type of validation needed (e.g., 'dataset', 'analysis', 'results')."
    )
    prompt: str = Field(
        default="Please review the following content:",
        description="Optional custom prompt to show before the content."
    )


class HumanInTheLoopTool(BaseTool):
    name: str = "Human Validation Tool"
    description: str = (
        "A tool for requesting human validation on specific content. "
        "It presents content to a human reviewer and collects their approval "
        "and feedback. Use this when you need human oversight or approval "
        "on important decisions or results."
    )
    args_schema: Type[BaseModel] = HumanInTheLoopInput

    def _run(self, content: str, validation_type: str, prompt: str = "Please review the following content:") -> str:
        """
        Implement the human validation interaction.

        Args:
            content: The content to be validated
            validation_type: Type of validation needed
            prompt: Custom prompt to show before the content

        Returns:
            str: JSON-formatted string containing validation results
        """
        print(f"\n{'='*20} Human Validation Required: {validation_type} {'='*20}")
        print(f"\n{prompt}\n")
        print(f"{content}\n")

        while True:
            approval = input("\nDo you approve? (yes/no): ").lower().strip()
            if approval in ['yes', 'no']:
                break
            print("Please enter 'yes' or 'no'")

        feedback = input("\nEnter your feedback or comments (press Enter if none): ").strip()

        result = {
            "approved": approval == "yes",
            "feedback": feedback or "No feedback provided",
            "validation_type": validation_type
        }

        return str(result)
