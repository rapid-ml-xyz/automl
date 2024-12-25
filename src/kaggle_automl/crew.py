import os
from dotenv import load_dotenv
from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task

from .agents import DatasetAcquisitionSpecialist, ReportingAnalyst, Researcher
from .tasks import DatasetTask, ReportingTask, ResearchTask


@CrewBase
class KaggleAutoml:
    """KaggleAutoml crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    load_dotenv()
    openai_llm = LLM(
        model=os.getenv("OPENAI_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base="https://api.openai.com/v1"
    )

    @agent
    def dataset_assessor(self) -> Agent:
        return DatasetAcquisitionSpecialist.create(self.openai_llm, self.agents_config)

    @agent
    def researcher(self) -> Agent:
        return Researcher.create(self.openai_llm, self.agents_config)

    @agent
    def reporting_analyst(self) -> Agent:
        return ReportingAnalyst.create(self.openai_llm, self.agents_config)

    @task
    def dataset_task(self) -> Task:
        return DatasetTask.create(self.tasks_config)

    @task
    def research_task(self) -> Task:
        return ResearchTask.create(self.tasks_config)

    @task
    def reporting_task(self) -> Task:
        return ReportingTask.create(self.tasks_config)

    @crew
    def crew(self) -> Crew:
        """Creates the KaggleAutoml crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
