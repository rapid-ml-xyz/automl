from crewai import Crew, Process
from crewai.flow.flow import Flow, listen, start
from .eda_crew import EDACrew


class EDAFlow(Flow):
    """Flow for orchestrating the EDA process using existing EDACrew components"""

    def __init__(self):
        self.eda_crew = EDACrew()
        super().__init__()

    @start()
    def dataset_acquisition_flow(self):
        crew = Crew(
            agents=[self.eda_crew.dataset_acquisition_specialist()],
            tasks=[self.eda_crew.dataset_acquisition_task()],
            process=Process.sequential,
            verbose=True
        )
        return crew.kickoff(inputs=self.state)

    @listen(dataset_acquisition_flow)
    def ydata_download_flow(self):
        crew = Crew(
            agents=[self.eda_crew.dataset_acquisition_specialist()],
            tasks=[self.eda_crew.ydata_download_task()],
            process=Process.sequential,
            verbose=True
        )
        return crew.kickoff()
