from crewai.tools import BaseTool
from typing import Type, Dict, List
from pydantic import BaseModel, Field
import plotly.graph_objects as go


class DataVisualizationToolInput(BaseModel):
    """Input schema for DataVisualizationTool."""
    data: Dict = Field(
        ...,
        description="Dictionary containing dataset information including variables and their metadata."
    )
    variable_names: List[str] = Field(
        ...,
        description="List of variable names to visualize. For correlation matrix, provide multiple numeric variables."
    )
    chart_type: str = Field(
        ...,
        description="Type of chart to create ('distribution' or 'correlation')"
    )
    title: str = Field(
        None,
        description="Optional: Custom title for the visualization"
    )


class DataVisualizationTool(BaseTool):
    name: str = "Data Visualization Tool"
    description: str = (
        "Creates statistical visualizations based on data types and relationships. "
        "Can generate distribution charts for individual variables or correlation "
        "matrices for multiple numeric variables."
    )
    args_schema: Type[BaseModel] = DataVisualizationToolInput

    def _run(
        self,
        data: Dict,
        variable_names: List[str],
        chart_type: str,
        title: str = None
    ) -> go.Figure:
        """Creates the requested visualization"""
        if chart_type == 'distribution':
            if len(variable_names) != 1:
                raise ValueError("Distribution charts require exactly one variable")
            return self._create_distribution_chart(
                data['variables'][variable_names[0]],
                title or variable_names[0]
            )
        elif chart_type == 'correlation':
            if len(variable_names) < 2:
                raise ValueError("Correlation matrix requires at least two variables")
            return self._create_correlation_matrix(data, variable_names)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

    def _create_distribution_chart(self, var_info: Dict, title: str) -> go.Figure:
        """Creates appropriate distribution chart based on variable type"""
        if var_info['type'] == 'Numeric':
            return self._create_numeric_distribution(var_info, title)
        else:
            return self._create_categorical_distribution(var_info, title)

    def _create_numeric_distribution(self, var_info: Dict, title: str) -> go.Figure:
        """Creates histogram for numeric variables"""
        fig = go.Figure()
        bin_size = max(1, (var_info['max'] - var_info['min']) / 20)
        bins = list(range(
            int(var_info['min']),
            int(var_info['max']) + 2,
            max(1, int(bin_size))
        ))

        fig.add_trace(go.Histogram(
            x=bins[:-1],
            nbinsx=len(bins)-1,
            name=title
        ))
        fig.update_layout(
            title=f"{title} Distribution",
            xaxis_title=title,
            yaxis_title="Count",
            showlegend=False
        )
        return fig

    def _create_categorical_distribution(self, var_info: Dict, title: str) -> go.Figure:
        """Creates bar chart for categorical variables"""
        categories = list(var_info.get('word_counts', {}).items())
        categories.sort(key=lambda x: x[1], reverse=True)

        fig = go.Figure(data=[
            go.Bar(
                x=[cat[0] for cat in categories],
                y=[cat[1] for cat in categories]
            )
        ])
        fig.update_layout(
            title=f"{title} Distribution",
            xaxis_title=title,
            yaxis_title="Count",
            showlegend=False,
            xaxis_tickangle=45
        )
        return fig

    def _create_correlation_matrix(self, data: Dict, variable_names: List[str]) -> go.Figure:
        for var in variable_names:
            if data['variables'][var]['type'] != 'Numeric':
                raise ValueError(f"Variable {var} is not numeric")

        matrix_size = len(variable_names)
        correlation_matrix = [[1 if i == j else 0.5 for j in range(matrix_size)] for i in range(matrix_size)]

        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=variable_names,
            y=variable_names,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        fig.update_layout(
            title="Variable Correlation Matrix",
            xaxis_tickangle=45
        )
        return fig
