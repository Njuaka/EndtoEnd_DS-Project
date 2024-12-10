import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure
from src.constants import *


class GenerateReport:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def generate_yield_trend_plot(self):
        """
        Generate a line plot for average yield values over time, grouped by crop.
        """
        grouped_data = self.data.groupby(['Year', 'crop_types']).agg({'yield_value (hg/ha)': 'mean'}).reset_index()
        fig = px.line(grouped_data, x='Year', y='yield_value (hg/ha)', color='crop_types', title='Average Yield Values Over Time')
        save_figure(fig, FIGURE_PATH +'yield_trend_plot.png')
        return fig


    def generate_summary_dashboard(self) ->Figure:
        """
        Generate a summary dashboard for average yield and pesticide values.
        """
        summary = self.data.groupby('crop_types').agg({
            'yield_value (hg/ha)': 'mean',
            'pest_value (tonnes)': 'mean',
            'average_rain_fall (mm/year)': 'mean'
        }).reset_index()
        
        fig = px.bar(summary, x='crop_types', y='yield_value (hg/ha)', title='Average Yield per Crop')
        save_figure(fig, FIGURE_PATH +'average_yield_per_crop_plot.png')
        return fig



    def descriptive_stats(self) -> None:
        """
        Compute descriptive statistics and save to a file.
        """
        
        desc_stats = self.data.describe()  # Compute descriptive statistics

        save_csv(desc_stats, FIGURE_PATH +'descriptives.csv')   # Save to a CSV file
        
    
        
    def calculate_correlations(self):
        """
        Calculate and return the correlation matrix for the specified columns, grouped country.
        """
        grouped_data = self.data.groupby('Country').agg({
            'pest_value (tonnes)': 'mean',
            'average_rain_fall (mm/year)': 'mean',
            'avg_temp (°C)': 'mean',
            'yield_value (hg/ha)': 'mean'
        }).reset_index()
        correlation_matrix = grouped_data[['pest_value (tonnes)', 'average_rain_fall (mm/year)', 'avg_temp (°C)', 'yield_value (hg/ha)']].corr()
        return correlation_matrix



    def visualize_correlations(self, correlation_matrix):
        """
        Generate a heatmap to visualize the correlation matrix.
        """
        fig = px.imshow(correlation_matrix, text_auto=True, title='Correlation Matrix')
        save_figure(fig, FIGURE_PATH +'correlation_matrix.png')
        return fig


    def plot_rainfall_yield_by_year(self) ->Figure:
        """
        Plots the distribution of pesticide use and rainfall over yield per year using Plotly.
        
        Parameters:
        data (pd.DataFrame): DataFrame containing columns 'Year', 'Pesticide', 'Rainfall', and 'Yield'.
        """
        
        avg_data = self.data.groupby('Year').agg({
            'average_rain_fall (mm/year)': 'mean',
            'yield_value (hg/ha)': 'mean'
        }).reset_index()

    
        # Scatter plot for Rainfall vs Yield
        fig = px.scatter(avg_data, x='Year', y='yield_value (hg/ha)', color='average_rain_fall (mm/year)',
                        title='Rainfall vs Yield Over Years',
                        labels={'Rainfall': 'Average Rainfall (mm/year)', 'Yield': 'yield_value (hg/ha)', 'Year': 'Year'},
                        hover_name='Year')
        fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
        save_figure(fig, FIGURE_PATH +'rainfall_vs_yield_overyears.png')
        return fig
    

    def plot_pesticide_yield_by_year(self) ->Figure:
        """
        Plots the distribution of pesticide use and rainfall over yield per year using Plotly.
        
        Parameters:
        data (pd.DataFrame): DataFrame containing columns 'Year', 'Pesticide', 'Rainfall', and 'Yield'.
        """
        avg_data = self.data.groupby('Year').agg({
            'pest_value (tonnes)': 'mean',
            'yield_value (hg/ha)': 'mean'
        }).reset_index()
        
        # Scatter plot for Pesticide vs Yield
        fig = px.scatter(avg_data, x='Year', y='yield_value (hg/ha)', color='pest_value (tonnes)',
                        title='Pesticide Use vs Yield Over Years',
                        labels={'Pesticide': 'pest_value (tonnes)', 'Yield': 'yield_value (hg/ha)', 'Year': 'Year'},
                        hover_name='Year')
        fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
        save_figure(fig, FIGURE_PATH +'pesticide_vs_yield_overyears.png')
        return fig


    def plot_yield_vs_continent(self) ->Figure:
        """
        Create a scatter plot of Yield vs Crop types within different continent using Plotly Express.
        """
        fig = px.scatter(self.data, 
                            x='crop_types', 
                            y='yield_value (hg/ha)', 
                            color='Continent', 
                            title='Yield vs Crop types within different continent',
                            labels={'Crop_Yield': 'Crop Yield', 'Continent': 'Continent', 'Crop_Type': 'Crop Type'},
                            hover_data=['Country'])
        save_figure(fig, FIGURE_PATH +'yield_vs_continent.png')
        return fig


    def generate_analyst_report(self):
        """Generates the full report for analysts.
        """
        self.generate_yield_trend_plot()
        self.generate_summary_dashboard()
        self.descriptive_stats()
        self.plot_rainfall_yield_by_year()
        self.plot_pesticide_yield_by_year()
        self.plot_yield_vs_continent()
        correlation_matrix= self.calculate_correlations()
        fig = self.visualize_correlations(correlation_matrix)
        return fig
        

    def generate_breeder_report(self):
        """Generates the full report for breeders.
        """
        self.generate_yield_trend_plot()
        self.generate_summary_dashboard()
        self.plot_rainfall_yield_by_year()
        self.plot_pesticide_yield_by_year()
        self.plot_yield_vs_continent()
        correlation_matrix = self.calculate_correlations()
        fig = self.visualize_correlations(correlation_matrix)
        return fig
    
    

class Report:
    def __init__(self, data):
        self.data = data
        self.generator = GenerateReport(data)

    def generate(self, user_type='analyst'):
        if user_type == 'analyst':
            print("Generating report for Analysts...\n")
            self.generator.generate_analyst_report()
        elif user_type == 'breeder':
            print("Generating detailed report for Plant Breeders...\n")
            return self.generator.generate_breeder_report()
    
    

class ModelPlot:
        def __init__(self, model_results):
            self.X_train = model_results['X_train']
            self.X_test = model_results['X_test']
            self.y_train = model_results['y_train']
            self.y_test = model_results['y_test']
            self.y_pred = model_results['y_pred']
            self.feature_importance = model_results['feature_importance']
            
        def plot_feature_importance(self):
            # Feature Importance Plot
            importance_df = pd.DataFrame({
                'Feature': self.X_train.columns,
                'Importance': self.feature_importance
            }).sort_values(by='Importance', ascending=False)

            fig = px.bar(importance_df, x='Feature', y='Importance', title='Feature Importance')
            save_figure(fig, FIGURE_PATH +'feature_Importance.png')
            return fig

        def plot_actual_vs_actual(self):
            # Actual vs Predicted Plot
            fig = px.scatter(x=self.y_test.squeeze(), y=self.y_pred.squeeze(), labels={'x': 'Actual', 'y': 'Predicted'}, title='Actual vs Predicted')
            fig.add_shape(type='line', x0=self.y_test.squeeze().min(), x1=self.y_test.squeeze().max(), y0=self.y_test.squeeze().min(), y1=self.y_test.squeeze().max(), line=dict(color='red', dash='dash'))
            #fig_actual_vs_predicted.show()
            save_figure(fig, FIGURE_PATH +'actual_vs_predicted.png')
            return fig


def save_figure(fig, FIGURE_PATH):
    """
    Save the figure to a file.
    """
    fig.write_image(FIGURE_PATH)
    
    
def save_csv(data, FIGURE_PATH):
    """
    Save the csv to path.
    """
    data.to_csv(FIGURE_PATH)