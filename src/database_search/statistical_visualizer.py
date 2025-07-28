"""
Statistical Visualization Utilities for Oceanographic Data
Provides plotting and visualization capabilities for statistical analysis results
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import os
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    HAS_PLOTTING_LIBS = True
except ImportError:
    HAS_PLOTTING_LIBS = False
    logging.warning("Matplotlib/Seaborn not available - plotting functionality disabled")

logger = logging.getLogger(__name__)


class StatisticalVisualizer:
    """
    Creates statistical visualizations for oceanographic data analysis
    """
    
    def __init__(self, output_dir: str = "statistical_plots"):
        """
        Initialize the statistical visualizer
        
        Args:
            output_dir: Directory to save generated plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.has_plotting = HAS_PLOTTING_LIBS
        
        if self.has_plotting:
            # Set up matplotlib style
            plt.style.use('default')
            sns.set_palette("husl")
            
        logger.info(f"Statistical visualizer initialized (plotting {'enabled' if self.has_plotting else 'disabled'})")
    
    def create_statistical_summary_plot(self, statistical_results: Dict[str, Any], 
                                      query: str = "") -> Optional[str]:
        """
        Create a summary plot showing key statistical measures
        
        Args:
            statistical_results: Results from statistical analysis
            query: Original user query for context
            
        Returns:
            Path to generated plot file, or None if plotting unavailable
        """
        if not self.has_plotting:
            logger.warning("Plotting libraries not available")
            return None
        
        try:
            statistics = statistical_results.get('statistics', {})
            if not statistics:
                logger.warning("No statistics data available for plotting")
                return None
            
            # Extract data for plotting
            plot_data = self._extract_plot_data(statistics)
            if not plot_data:
                return None
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"Statistical Analysis Summary", fontsize=16, fontweight='bold')
            
            # Plot 1: Basic statistics (min, max, avg) as bar chart
            if any(op in statistics for op in ['min', 'minimum', 'max', 'maximum', 'avg', 'average']):
                self._plot_basic_statistics(axes[0, 0], statistics, plot_data)
            
            # Plot 2: Distribution statistics (std, variance) if available
            if any(op in statistics for op in ['std', 'stdev', 'variance', 'var']):
                self._plot_distribution_statistics(axes[0, 1], statistics, plot_data)
            
            # Plot 3: Range and count information
            if any(op in statistics for op in ['range', 'count']):
                self._plot_range_count(axes[1, 0], statistics, plot_data)
            
            # Plot 4: Data quality metrics
            quality_metrics = statistical_results.get('quality_metrics', {})
            if quality_metrics:
                self._plot_quality_metrics(axes[1, 1], quality_metrics)
            
            # Add query information
            if query:
                fig.text(0.02, 0.02, f"Query: {query[:80]}{'...' if len(query) > 80 else ''}", 
                        fontsize=8, style='italic')
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"statistical_summary_{timestamp}.png"
            plot_path = self.output_dir / plot_filename
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Statistical summary plot saved: {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error creating statistical summary plot: {e}")
            return None
    
    def create_time_series_plot(self, time_series_data: Dict[str, Any], 
                               parameter: str = "temperature") -> Optional[str]:
        """
        Create a time series plot showing trends over time
        
        Args:
            time_series_data: Time series analysis results
            parameter: Parameter to plot
            
        Returns:
            Path to generated plot file, or None if plotting unavailable
        """
        if not self.has_plotting:
            return None
        
        try:
            aggregated_data = time_series_data.get('aggregated_data', {})
            if parameter not in aggregated_data:
                logger.warning(f"Parameter {parameter} not found in time series data")
                return None
            
            param_data = aggregated_data[parameter]
            
            # Convert timestamp strings to datetime objects
            timestamps = []
            means = []
            mins = []
            maxs = []
            
            for timestamp_str, mean_val in param_data.get('mean', {}).items():
                if pd.notna(mean_val):
                    try:
                        timestamp = pd.to_datetime(timestamp_str)
                        timestamps.append(timestamp)
                        means.append(mean_val)
                        
                        # Get corresponding min/max values
                        min_val = param_data.get('min', {}).get(timestamp_str, mean_val)
                        max_val = param_data.get('max', {}).get(timestamp_str, mean_val)
                        mins.append(min_val)
                        maxs.append(max_val)
                    except:
                        continue
            
            if not timestamps:
                logger.warning("No valid timestamps found for time series plot")
                return None
            
            # Create time series plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot mean line
            ax.plot(timestamps, means, label='Average', linewidth=2, marker='o', markersize=4)
            
            # Fill between min and max
            if mins != means or maxs != means:
                ax.fill_between(timestamps, mins, maxs, alpha=0.3, label='Min-Max Range')
            
            # Formatting
            ax.set_title(f'{parameter.title()} Over Time', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time')
            ax.set_ylabel(f'{parameter.title()} ({self._get_unit_for_parameter(parameter)})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(timestamps)//10)))
            plt.xticks(rotation=45)
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"time_series_{parameter}_{timestamp}.png"
            plot_path = self.output_dir / plot_filename
            
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Time series plot saved: {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error creating time series plot: {e}")
            return None
    
    def create_correlation_matrix_plot(self, correlation_data: Dict[str, Any]) -> Optional[str]:
        """
        Create a correlation matrix heatmap
        
        Args:
            correlation_data: Correlation analysis results
            
        Returns:
            Path to generated plot file, or None if plotting unavailable
        """
        if not self.has_plotting:
            return None
        
        try:
            correlation_matrix = correlation_data.get('correlation_matrix', {})
            if not correlation_matrix:
                logger.warning("No correlation matrix data available")
                return None
            
            # Convert to DataFrame for easier plotting
            df_corr = pd.DataFrame(correlation_matrix)
            
            # Create correlation heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create heatmap
            mask = np.triu(np.ones_like(df_corr, dtype=bool))  # Mask upper triangle
            sns.heatmap(df_corr, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, ax=ax, cbar_kws={"shrink": .8})
            
            ax.set_title('Parameter Correlation Matrix', fontsize=14, fontweight='bold')
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"correlation_matrix_{timestamp}.png"
            plot_path = self.output_dir / plot_filename
            
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Correlation matrix plot saved: {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error creating correlation matrix plot: {e}")
            return None
    
    def create_distribution_plot(self, data: pd.DataFrame, parameter: str) -> Optional[str]:
        """
        Create a distribution plot (histogram + box plot)
        
        Args:
            data: DataFrame with parameter data
            parameter: Parameter to analyze
            
        Returns:
            Path to generated plot file, or None if plotting unavailable
        """
        if not self.has_plotting:
            return None
        
        try:
            if parameter not in data.columns:
                logger.warning(f"Parameter {parameter} not found in data")
                return None
            
            param_data = data[parameter].dropna()
            if len(param_data) == 0:
                logger.warning(f"No valid data for parameter {parameter}")
                return None
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Histogram
            ax1.hist(param_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title(f'{parameter.title()} Distribution', fontsize=12, fontweight='bold')
            ax1.set_xlabel(f'{parameter.title()} ({self._get_unit_for_parameter(parameter)})')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = f"Mean: {param_data.mean():.2f}\nStd: {param_data.std():.2f}\nCount: {len(param_data)}"
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Box plot
            ax2.boxplot(param_data, vert=False, patch_artist=True, 
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
            ax2.set_xlabel(f'{parameter.title()} ({self._get_unit_for_parameter(parameter)})')
            ax2.set_title('Box Plot (showing outliers)', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"distribution_{parameter}_{timestamp}.png"
            plot_path = self.output_dir / plot_filename
            
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Distribution plot saved: {plot_path}")
            return str(plot_path)
            
        except Exception as e:
            logger.error(f"Error creating distribution plot: {e}")
            return None
    
    def _extract_plot_data(self, statistics: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data suitable for plotting from statistics"""
        plot_data = {
            'parameters': [],
            'values': {},
            'operations': []
        }
        
        for operation, results in statistics.items():
            if isinstance(results, dict):
                plot_data['operations'].append(operation)
                
                for parameter, values in results.items():
                    if parameter not in plot_data['parameters']:
                        plot_data['parameters'].append(parameter)
                    
                    if parameter not in plot_data['values']:
                        plot_data['values'][parameter] = {}
                    
                    if isinstance(values, dict) and 'value' in values:
                        plot_data['values'][parameter][operation] = values['value']
        
        return plot_data
    
    def _plot_basic_statistics(self, ax, statistics: Dict[str, Any], plot_data: Dict[str, Any]):
        """Plot basic statistics (min, max, avg) as bar chart"""
        parameters = plot_data['parameters'][:5]  # Limit to first 5 parameters
        basic_ops = ['min', 'minimum', 'max', 'maximum', 'avg', 'average', 'mean']
        
        x = np.arange(len(parameters))
        width = 0.25
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        op_labels = []
        
        offset = 0
        for i, op in enumerate(basic_ops):
            if op in statistics:
                values = []
                for param in parameters:
                    if param in plot_data['values'] and op in plot_data['values'][param]:
                        values.append(plot_data['values'][param][op])
                    else:
                        values.append(0)
                
                if any(values):  # Only plot if we have non-zero values
                    ax.bar(x + offset * width, values, width, label=op.title(), 
                          color=colors[offset % len(colors)], alpha=0.8)
                    op_labels.append(op.title())
                    offset += 1
        
        ax.set_xlabel('Parameters')
        ax.set_ylabel('Values')
        ax.set_title('Basic Statistics')
        ax.set_xticks(x + width)
        ax.set_xticklabels([p.title() for p in parameters], rotation=45)
        if op_labels:
            ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_distribution_statistics(self, ax, statistics: Dict[str, Any], plot_data: Dict[str, Any]):
        """Plot distribution statistics (std, variance)"""
        parameters = plot_data['parameters'][:5]
        dist_ops = ['std', 'stdev', 'variance', 'var']
        
        found_ops = [op for op in dist_ops if op in statistics]
        if not found_ops:
            ax.text(0.5, 0.5, 'No distribution\nstatistics available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Distribution Statistics')
            return
        
        x = np.arange(len(parameters))
        width = 0.4
        
        for i, op in enumerate(found_ops[:2]):  # Limit to 2 operations
            values = []
            for param in parameters:
                if param in plot_data['values'] and op in plot_data['values'][param]:
                    values.append(plot_data['values'][param][op])
                else:
                    values.append(0)
            
            ax.bar(x + i * width, values, width, label=op.title(), alpha=0.8)
        
        ax.set_xlabel('Parameters')
        ax.set_ylabel('Values')
        ax.set_title('Distribution Statistics')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels([p.title() for p in parameters], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_range_count(self, ax, statistics: Dict[str, Any], plot_data: Dict[str, Any]):
        """Plot range and count information"""
        parameters = plot_data['parameters'][:5]
        
        if 'range' in statistics:
            ranges = []
            for param in parameters:
                if param in plot_data['values'] and 'range' in plot_data['values'][param]:
                    ranges.append(plot_data['values'][param]['range'])
                else:
                    ranges.append(0)
            
            ax.bar(parameters, ranges, color='orange', alpha=0.7)
            ax.set_ylabel('Range')
            ax.set_title('Parameter Ranges')
        elif 'count' in statistics:
            counts = []
            for param in parameters:
                if param in plot_data['values'] and 'count' in plot_data['values'][param]:
                    counts.append(plot_data['values'][param]['count'])
                else:
                    counts.append(0)
            
            ax.bar(parameters, counts, color='green', alpha=0.7)
            ax.set_ylabel('Count')
            ax.set_title('Data Point Counts')
        else:
            ax.text(0.5, 0.5, 'No range or count\ndata available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Range/Count')
        
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _plot_quality_metrics(self, ax, quality_metrics: Dict[str, Any]):
        """Plot data quality metrics"""
        completeness = quality_metrics.get('completeness', {})
        
        if completeness:
            params = list(completeness.keys())[:5]
            completeness_values = [completeness[p].get('completeness_percentage', 0) for p in params]
            
            bars = ax.bar(params, completeness_values, color='lightgreen', alpha=0.7)
            ax.set_ylabel('Completeness %')
            ax.set_title('Data Quality (Completeness)')
            ax.set_ylim(0, 100)
            
            # Add value labels on bars
            for bar, value in zip(bars, completeness_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No quality metrics\navailable', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Data Quality')
        
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _get_unit_for_parameter(self, parameter: str) -> str:
        """Get appropriate unit for a parameter"""
        unit_mappings = {
            'temperature': '°C',
            'salinity': 'PSU',
            'pressure': 'dbar',
            'oxygen': 'mg/L',
            'ph': 'pH units',
            'conductivity': 'S/m',
            'depth': 'm',
            'chlorophyll': 'mg/m³',
            'turbidity': 'NTU'
        }
        
        parameter_lower = parameter.lower()
        for param, unit in unit_mappings.items():
            if param in parameter_lower:
                return unit
        
        return 'units'
    
    def get_available_plot_types(self) -> List[str]:
        """Get list of available plot types"""
        if not self.has_plotting:
            return []
        
        return [
            'statistical_summary',
            'time_series',
            'correlation_matrix',
            'distribution',
            'quality_metrics'
        ]
    
    def cleanup_old_plots(self, days_old: int = 7):
        """
        Clean up old plot files
        
        Args:
            days_old: Remove files older than this many days
        """
        try:
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            removed_count = 0
            for plot_file in self.output_dir.glob("*.png"):
                if plot_file.stat().st_mtime < cutoff_date.timestamp():
                    plot_file.unlink()
                    removed_count += 1
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old plot files")
                
        except Exception as e:
            logger.warning(f"Error cleaning up old plots: {e}")