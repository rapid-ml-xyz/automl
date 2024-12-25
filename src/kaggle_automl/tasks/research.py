from crewai import Task


class ResearchTask:
    """Research task"""
    @classmethod
    def create(cls, tasks_config) -> Task:
        return Task(
            config=tasks_config['research_task']
        )
