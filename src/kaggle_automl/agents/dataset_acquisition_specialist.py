from crewai import Agent
from ..tools import KaggleDownloadTool, KaggleMetadataExtractorTool


class DatasetAcquisitionSpecialist:
    """Dataset Acquisition Specialist agent"""
    @classmethod
    def create(cls, llm, agents_config) -> Agent:
        return Agent(
            config=agents_config['dataset_acquisition_specialist'],
            llm=llm,
            tools=[KaggleDownloadTool(), KaggleMetadataExtractorTool()],
            verbose=True
        )
