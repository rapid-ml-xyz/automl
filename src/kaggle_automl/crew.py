import os
from dotenv import load_dotenv
from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import CSVSearchTool
from .tools import (
	ArxivSearchTool,
	KaggleDownloadTool,
	KaggleMetadataExtractorTool,
	KaggleSearchTool,
	PapersWithCodeSearchTool
)


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
	def senior_project_manager(self) -> Agent:
		return Agent(
			allow_delegation=False,  # Disabling for now, since it isn't actually meant to manage other agents
			config=self.agents_config['senior_project_manager'],
			llm=self.openai_llm,
			# Search results are terrible!
			tools=[ArxivSearchTool(), KaggleSearchTool(), PapersWithCodeSearchTool()],
			verbose=True
		)

	@agent
	def dataset_acquisition_specialist(self) -> Agent:
		return Agent(
			allow_delegation=False,
			config=self.agents_config['dataset_acquisition_specialist'],
			llm=self.openai_llm,
			tools=[KaggleDownloadTool(), KaggleMetadataExtractorTool()],
			verbose=True
		)

	@agent
	def assistant_project_manager(self) -> Agent:
		return Agent(
			allow_delegation=False,
			config=self.agents_config['assistant_project_manager'],
			llm=self.openai_llm,
			verbose=True
		)

	@agent
	def data_scientist(self) -> Agent:
		return Agent(
			allow_delegation=False,
			config=self.agents_config['data_scientist'],
			llm=self.openai_llm,
			tools=[CSVSearchTool()],
			verbose=True
		)

	@agent
	def machine_learning_research_engineer(self) -> Agent:
		return Agent(
			allow_delegation=False,
			config=self.agents_config['machine_learning_research_engineer'],
			llm=self.openai_llm,
			verbose=True
		)

	@agent
	def machine_learning_operations_engineer(self) -> Agent:
		return Agent(
			allow_delegation=False,
			config=self.agents_config['machine_learning_operations_engineer'],
			llm=self.openai_llm,
			verbose=True
		)

	@task
	def request_verification_relevancy_task(self) -> Task:
		return Task(config=self.tasks_config['request_verification_relevancy_task'])

	@task
	def dataset_acquisition_task(self) -> Task:
		return Task(config=self.tasks_config['dataset_acquisition_task'])

	@task
	def request_verification_adequacy_task(self) -> Task:
		return Task(config=self.tasks_config['request_verification_adequacy_task'])

	# Might be redundant, since it's only pretty-printing the dataset_acquisition_task output
	@task
	def request_parsing_task(self) -> Task:
		return Task(config=self.tasks_config['request_parsing_task'])

	@task
	def retrieval_augmented_planning_task(self) -> Task:
		return Task(config=self.tasks_config['retrieval_augmented_planning_task'])

	@task
	def data_decomposition_task(self) -> Task:
		return Task(config=self.tasks_config['data_decomposition_task'])

	@task
	def plan_decomposition_task(self) -> Task:
		return Task(config=self.tasks_config['plan_decomposition_task'])

	@crew
	def crew(self) -> Crew:
		"""Creates the KaggleAutoml crew"""

		return Crew(
			agents=self.agents,
			tasks=self.tasks,
			process=Process.sequential,
			verbose=True,
		)
