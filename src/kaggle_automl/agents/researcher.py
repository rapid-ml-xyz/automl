from crewai import Agent


class Researcher:
    """Researcher agent"""
    @classmethod
    def create(cls, llm, agents_config) -> Agent:
        return Agent(
            config=agents_config['researcher'],
            llm=llm,
            verbose=True
        )
