import os
from dotenv import load_dotenv
from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileReadTool
from .tools import (
    CsvPreviewTool,
    DataVisualizationTool,
    DirectoryReadTool,
    FileOperationTool,
    KaggleDownloadTool,
    KaggleMetadataExtractorTool,
    PWDTool,
    YDataDownloadTool,
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
    def exploratory_data_analyst(self) -> Agent:
        return Agent(
            allow_delegation=False,
            config=self.agents_config['exploratory_data_analyst'],
            llm=self.openai_llm,
            tools=[DirectoryReadTool(), FileReadTool(), PWDTool()],
            verbose=True
        )

    @agent
    def visualization_agent(self) -> Agent:
        return Agent(
            allow_delegation=False,
            config=self.agents_config['visualization_agent'],
            llm=self.openai_llm,
            tools=[CsvPreviewTool(), DataVisualizationTool(), DirectoryReadTool(),
                   FileReadTool(), PWDTool()],
            verbose=True
        )

    @task
    def dataset_acquisition_task(self) -> Task:
        return Task(config=self.tasks_config['dataset_acquisition_task'])

    @task
    def ydata_download_task(self) -> Task:
        return Task(config=self.tasks_config['ydata_download_task'])

    @task
    def exploratory_data_analysis_task(self) -> Task:
        return Task(config=self.tasks_config['exploratory_data_analysis_task'])

    @task
    def visualization_task(self) -> Task:
        return Task(config=self.tasks_config['visualization_task'])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
