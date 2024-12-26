import os
from dotenv import load_dotenv
from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from .tools import kaggle_download_tool, kaggle_metadata_extractor_tool


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
		return Agent(
			allow_delegation=False,
			config=self.agents_config['dataset_acquisition_specialist'],
			llm=self.openai_llm,
			tools=[
				kaggle_download_tool.KaggleDownloadTool(),
				kaggle_metadata_extractor_tool.KaggleMetadataExtractorTool()
			],
			verbose=True
		)

	@task
	def dataset_task(self) -> Task:
		return Task(config=self.tasks_config['dataset_acquisition_task'])

	@crew
	def crew(self) -> Crew:
		"""Creates the KaggleAutoml crew"""

		return Crew(
			agents=self.agents,
			tasks=self.tasks,
			process=Process.sequential,
			verbose=True,
		)
