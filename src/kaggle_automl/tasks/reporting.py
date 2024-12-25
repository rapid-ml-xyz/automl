from crewai import Task


class ReportingTask:
    """Reporting task"""
    @classmethod
    def create(cls, tasks_config) -> Task:
        return Task(
            config=tasks_config['reporting_task'],
            output_file='report.md'
        )
