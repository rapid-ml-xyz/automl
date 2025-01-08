import os
from dotenv import load_dotenv
from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileReadTool
from .tools import (
    CsvPreviewTool,
    DirectoryReadTool,
    FileOperationTool,
    HumanInTheLoopTool,
    KaggleDownloadTool,
    KaggleMetadataExtractorTool,
    PWDTool,
    YDataDownloadTool,
    YDataProfilerTool,
)


@CrewBase
class EDACrew:
    """KaggleAutoml crew with iterative task loop"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/eda_tasks.yaml'
    load_dotenv()

    openai_llm = LLM(
        model=os.getenv("OPENAI_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base="https://api.openai.com/v1"
    )

    @agent
    def dataset_acquisition_specialist(self) -> Agent:
        return Agent(
            allow_delegation=False,
            config=self.agents_config['dataset_acquisition_specialist'],
            llm=self.openai_llm,
            tools=[DirectoryReadTool(), FileOperationTool(), KaggleDownloadTool(),
                   KaggleMetadataExtractorTool(), PWDTool(), YDataDownloadTool()],
            verbose=True
        )

    @agent
    def data_loader_agent(self) -> Agent:
        return Agent(
            allow_delegation=False,
            config=self.agents_config['data_loader_agent'],
            llm=self.openai_llm,
            tools=[DirectoryReadTool(), FileReadTool(), PWDTool()],
            verbose=True
        )

    @agent
    def human_feedback_specialist(self) -> Agent:
        return Agent(
            allow_delegation=False,
            config=self.agents_config['human_feedback_specialist'],
            llm=self.openai_llm,
            tools=[CsvPreviewTool(), DirectoryReadTool(), PWDTool(), HumanInTheLoopTool()],
            verbose=True
        )

    @agent
    def exploratory_data_analyst(self) -> Agent:
        return Agent(
            allow_delegation=False,
            config=self.agents_config['exploratory_data_analyst'],
            llm=self.openai_llm,
            tools=[DirectoryReadTool(), PWDTool(), YDataProfilerTool()],
            verbose=True
        )

    @task
    def dataset_acquisition_task(self) -> Task:
        return Task(config=self.tasks_config['dataset_acquisition_task'])

    @task
    def ydata_download_task(self) -> Task:
        return Task(config=self.tasks_config['ydata_download_task'])

    @task
    def json_loader_task(self) -> Task:
        return Task(config=self.tasks_config['json_loader_task'])

    @task
    def column_selection_task(self) -> Task:
        return Task(config=self.tasks_config['column_selection_task'])

    @task
    def exploratory_data_analysis_task(self) -> Task:
        return Task(config=self.tasks_config['exploratory_data_analysis_task'])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
