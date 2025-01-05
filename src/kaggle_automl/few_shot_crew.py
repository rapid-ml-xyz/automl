import os
from dotenv import load_dotenv
from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import CSVSearchTool
from .tools import CsvPreviewTool, DirectoryReadTool, FileExecutorTool


@CrewBase
class FewShot:
    """KaggleAutoml crew with iterative task loop"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/few_shot_tasks.yaml'
    load_dotenv()

    openai_llm = LLM(
        model=os.getenv("OPENAI_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base="https://api.openai.com/v1"
    )

    anthropic_llm = LLM(
        model=os.getenv("ANTHROPIC_MODEL"),
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.7
    )

    @agent
    def chief_button_pusher(self) -> Agent:
        return Agent(
            allow_delegation=False,
            config=self.agents_config['chief_button_pusher'],
            llm=self.openai_llm,
            tools=[DirectoryReadTool(), FileExecutorTool()],
            verbose=True
        )

    @agent
    def machine_learning_operations_engineer(self) -> Agent:
        return Agent(
            allow_delegation=False,
            config=self.agents_config['machine_learning_operations_engineer'],
            llm=self.anthropic_llm,
            tools=[CSVSearchTool(), CsvPreviewTool(), DirectoryReadTool()],
            verbose=True
        )

    @task
    def code_execution_task(self) -> Task:
        return Task(config=self.tasks_config['code_execution_task'])

    @task
    def code_evaluation_task(self) -> Task:
        return Task(config=self.tasks_config['code_evaluation_task'])

    @crew
    def crew(self) -> Crew:
        """Creates the KaggleAutoml crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
