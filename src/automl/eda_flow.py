from crewai import Crew, Process
from crewai.flow.flow import Flow, listen, start
from .eda_crew import EDACrew


class DownloadFlow(Flow):
    """Flow for downloading the dataset and y-data analysis"""

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


class ExplorationFlow(Flow):
    """Flow for EDA on the saved dataset"""

    def __init__(self):
        self.eda_crew = EDACrew()
        super().__init__()

    @start()
    def exploratory_data_analysis_flow(self):
        crew = Crew(
            agents=[self.eda_crew.exploratory_data_analyst()],
            tasks=[self.eda_crew.exploratory_data_analysis_task()],
            process=Process.sequential,
            verbose=True
        )
        return crew.kickoff(inputs=self.state)
