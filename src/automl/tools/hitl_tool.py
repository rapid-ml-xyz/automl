from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


class HumanInTheLoopInput(BaseModel):
    """Input schema for HumanValidationTool."""
    content: str = Field(
        ...,
        description="The content to be validated by the human reviewer."
    )
    prompt: str = Field(
        default="Please review the following content:",
        description="Optional custom prompt to show before the content."
    )


class HumanInTheLoopTool(BaseTool):
    name: str = "Human Feedback Tool"
    description: str = (
        "A tool for collecting human feedback on specific content. "
        "It presents content to a human reviewer and collects their comments. "
        "Use this tool when you need insights or opinions on important content."
    )
    args_schema: Type[BaseModel] = HumanInTheLoopInput

    def _run(self, content: str, prompt: str = "Please review the following content:") -> str:
        """
        Implement the human validation interaction.

        Args:
            content: The content to be validated
            prompt: Custom prompt to show before the content

        Returns:
            str: JSON-formatted string containing validation results
        """
        print(f"\n{prompt}\n")
        print(f"{content}\n")

        feedback = input("\nEnter your feedback or comments (press Enter if none): ").strip()

        result = {
            "feedback": feedback or "No feedback provided"
        }

        return str(result)
