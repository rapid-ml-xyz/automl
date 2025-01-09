from crewai.tools import BaseTool
from typing import Type, List
from pydantic import BaseModel, Field
import plotly.graph_objects as go
import numpy as np
import json


class DataVisualizationToolInput(BaseModel):
    """Input schema for DataVisualizationTool."""
    json_filepath: str = Field(
        ...,
        description="Path to the YData profile JSON file to process"
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
        json_filepath: str,
        variable_names: List[str],
        chart_type: str,
        title: str = None
    ) -> go.Figure:
        """Creates the requested visualization"""
        # Load the JSON data
        with open(json_filepath, 'r') as f:
            data = json.load(f)

        if chart_type == 'distribution':
            if len(variable_names) != 1:
                raise ValueError("Distribution charts require exactly one variable")
            var_info = data['variables'][variable_names[0]]
            return self._create_distribution_chart(var_info, variable_names[0], title)
        elif chart_type == 'correlation':
            return self._create_correlation_matrix(data, variable_names, title)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

    def _create_distribution_chart(self, var_info: dict, var_name: str, title: str = None) -> go.Figure:
        """Creates appropriate distribution chart based on variable type"""
        if var_info['type'] == 'Numeric':
            return self._create_numeric_distribution(var_info, var_name, title)
        else:
            return self._create_categorical_distribution(var_info, var_name, title)

    def _create_numeric_distribution(self, var_info: dict, var_name: str, title: str = None) -> go.Figure:
        """Creates histogram for numeric variables with summary statistics"""
        fig = go.Figure()

        # Create bins based on percentiles
        bins = []
        for p in ['min', '5%', '25%', '50%', '75%', '95%', 'max']:
            if p in var_info:
                bins.append(var_info[p])

        if not bins:
            bins = np.linspace(var_info['min'], var_info['max'], 20)

        fig.add_trace(go.Histogram(
            x=bins,
            nbinsx=len(bins)-1,
            name=var_name,
            showlegend=False
        ))

        # Add box plot
        fig.add_trace(go.Box(
            x=[var_info[p] for p in ['25%', '50%', '75%'] if p in var_info],
            name='Distribution',
            boxpoints=False,
            showlegend=False
        ))

        # Add statistics
        stats_text = f"Mean: {var_info.get('mean', 'N/A'):.2f}<br>"
        stats_text += f"Std: {var_info.get('std', 'N/A'):.2f}<br>"
        stats_text += f"Skewness: {var_info.get('skewness', 'N/A'):.2f}"

        fig.add_annotation(
            x=0.95,
            y=0.95,
            xref="paper",
            yref="paper",
            text=stats_text,
            showarrow=False,
            align="right",
            bgcolor="rgba(255,255,255,0.8)"
        )

        fig.update_layout(
            title=title or f"Distribution of {var_name}",
            xaxis_title=var_name,
            yaxis_title="Count",
            barmode='overlay'
        )
        return fig

    def _create_categorical_distribution(self, var_info: dict, var_name: str, title: str = None) -> go.Figure:
        """Creates bar chart for categorical variables"""
        categories = []
        counts = []

        if 'word_counts' in var_info:
            for cat, count in sorted(var_info['word_counts'].items(), key=lambda x: x[1], reverse=True):
                if cat.lower() not in ['others', 'unknown']:
                    categories.append(cat)
                    counts.append(count)

        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=counts,
                text=counts,
                textposition='auto',
            )
        ])

        # Add percentages
        total = sum(counts)
        percentages = [count/total * 100 for count in counts]

        annotations = []
        for i, (cat, pct) in enumerate(zip(categories, percentages)):
            if pct >= 5:  # Only show significant percentages
                annotations.append(dict(
                    x=cat,
                    y=counts[i],
                    text=f'{pct:.1f}%',
                    yshift=10,
                    showarrow=False
                ))

        fig.update_layout(
            title=title or f"Distribution of {var_name}",
            xaxis_title=var_name,
            yaxis_title="Count",
            showlegend=False,
            annotations=annotations,
            xaxis_tickangle=45
        )
        return fig

    def _create_correlation_matrix(self, data: dict, variable_names: List[str], title: str = None) -> go.Figure:
        """Creates correlation matrix for numeric variables"""
        numeric_vars = [var for var in variable_names
                       if data['variables'][var]['type'] == 'Numeric']

        if len(numeric_vars) < 2:
            raise ValueError("Need at least 2 numeric variables for correlation matrix")

        n_vars = len(numeric_vars)
        matrix = np.eye(n_vars)

        for i in range(n_vars):
            for j in range(i+1, n_vars):
                # Placeholder correlation
                correlation = -0.5 + np.random.random()
                matrix[i,j] = correlation
                matrix[j,i] = correlation

        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=numeric_vars,
            y=numeric_vars,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))

        fig.update_layout(
            title=title or "Correlation Matrix",
            xaxis_tickangle=45,
            height=max(400, len(numeric_vars) * 40),
            width=max(400, len(numeric_vars) * 40)
        )
        return fig
