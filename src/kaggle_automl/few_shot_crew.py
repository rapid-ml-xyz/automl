import os
from dotenv import load_dotenv
from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import CSVSearchTool, DirectoryReadTool
from .tools import CsvPreviewTool, FileExecutorTool


@CrewBase
class FewShot:
    """KaggleAutoml crew with iterative task loop"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
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
    def machine_learning_operations_engineer(self) -> Agent:
        return Agent(
            allow_delegation=False,
            config=self.agents_config['machine_learning_operations_engineer'],
            llm=self.anthropic_llm,
            tools=[CSVSearchTool(), CsvPreviewTool(), DirectoryReadTool(), FileExecutorTool()],
            verbose=True
        )

    @task
    def code_evaluation_task(self, iteration: int, previous_result: str = None) -> Task:
        config = self.tasks_config['code_evaluation_task'].copy()

        # Update task description to include iteration and previous results
        config['description'] = f"""
        Iteration {iteration}: Evaluate the code execution results and suggest improvements.
        
        Previous execution result:
        {previous_result}
        
        If the code meets all requirements or no further improvements are needed,
        include 'EXIT_CONDITION_MET' in your response.
        
        {config.get('description', '')}
        """

        return Task(
            config=config,
            agent=self.machine_learning_operations_engineer()
        )

    @task
    def code_execution_task(self, iteration: int, code: str = None) -> Task:
        config = self.tasks_config['code_execution_task'].copy()

        # Update task description to include iteration and code to execute
        config['description'] = f"""
        Iteration {iteration}: Execute and analyze the following code:
        
        {code}
        
        {config.get('description', '')}
        """

        return Task(
            config=config,
            agent=self.machine_learning_operations_engineer()
        )

    @crew
    def crew(self) -> Crew:
        """Creates the KaggleAutoml crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
