from crewai import Task


class DatasetTask:
    """Dataset acquisition task"""
    @classmethod
    def create(cls, tasks_config) -> Task:
        return Task(
            config=tasks_config['dataset_acquisition_task']
        )
