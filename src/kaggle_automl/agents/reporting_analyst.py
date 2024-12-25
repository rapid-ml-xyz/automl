from crewai import Agent


class ReportingAnalyst:
    """Reporting Analyst agent"""
    @classmethod
    def create(cls, llm, agents_config) -> Agent:
        return Agent(
            config=agents_config['reporting_analyst'],
            llm=llm,
            verbose=True
        )
