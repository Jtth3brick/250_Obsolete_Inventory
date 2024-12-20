import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List
from .simulation import SimulationResults
from matplotlib.gridspec import GridSpec

def print_performance_statistics(results: Dict[str, List[SimulationResults]]) -> Dict[str, Dict]:
    """
    Print detailed performance statistics for each policy.
    
    Args:
        results: Dictionary mapping policy names to lists of simulation results
        
    Returns:
        Dictionary of computed statistics by policy
    """
    # Calculate statistics for each policy
    stats = {}
    for policy_name, policy_results in results.items():
        stats[policy_name] = {
            'cost': {
                'mean': np.mean([r.total_cost for r in policy_results]),
                'std': np.std([r.total_cost for r in policy_results])
            },
            'emissions': {
                'mean': np.mean([r.carbon_emissions for r in policy_results]),
                'std': np.std([r.carbon_emissions for r in policy_results])
            },
            'service': {
                'mean': np.mean([r.service_level for r in policy_results]),
                'std': np.std([r.service_level for r in policy_results])
            },
            'backorder': {
                'mean': np.mean([r.summary_stats["backorder_costs"] for r in policy_results]),
                'std': np.std([r.summary_stats["backorder_costs"] for r in policy_results])
            },
            'holding': {
                'mean': np.mean([r.summary_stats["holding_costs"] for r in policy_results]),
                'std': np.std([r.summary_stats["holding_costs"] for r in policy_results])
            },
            'cpu_time': {
                'mean': np.mean([r.info['cpu_time'] for r in policy_results]),
                'std': np.std([r.info['cpu_time'] for r in policy_results])
            }
        }
    
    # Print formatted statistics
    print("\nPerformance Statistics")
    print("=====================")
    for policy_name, policy_stats in stats.items():
        print(f"\n{policy_name}:")
        print(f"  Total Cost: ${policy_stats['cost']['mean']:,.2f} (±${policy_stats['cost']['std']:,.2f})")
        print(f"  Backorder Cost: ${policy_stats['backorder']['mean']:,.2f} (±${policy_stats['backorder']['std']:,.2f})")
        print(f"  Holding Cost: ${policy_stats['holding']['mean']:,.2f} (±${policy_stats['holding']['std']:,.2f})")
        print(f"  Carbon Emissions: {policy_stats['emissions']['mean']:,.2f} (±{policy_stats['emissions']['std']:,.2f})")
        print(f"  Service Level: {policy_stats['service']['mean']*100:.1f}% (±{policy_stats['service']['std']*100:.1f}%)")
        print(f"  CPU Time: {policy_stats['cpu_time']['mean']:.2f}s (±{policy_stats['cpu_time']['std']:.2f}s)")
    
    return stats

def plot_relative_performance(results: Dict[str, List[SimulationResults]]) -> plt.Figure:
    """
    Create subplots showing each metric across policies in absolute values.
    
    Args:
        results: Dictionary mapping policy names to lists of simulation results
        
    Returns:
        Matplotlib figure object
    """
    # Calculate means for each policy
    metrics = {}
    for policy_name, policy_results in results.items():
        metrics[policy_name] = {
            'cost': np.mean([r.total_cost for r in policy_results]),
            'emissions': np.mean([r.carbon_emissions for r in policy_results]),
            'service': np.mean([r.service_level for r in policy_results]),
            'backorder': np.mean([r.summary_stats["backorder_costs"] for r in policy_results]),
            'holding': np.mean([r.summary_stats["holding_costs"] for r in policy_results])
        }
    
    # Create plot with 5 subplots
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(25, 5))
    
    # Setup data
    policies = list(metrics.keys())
    x = np.arange(len(policies))
    width = 0.35
    
    # Plot each metric
    plot_configs = [
        (ax1, 'cost', 'Total Cost ($)', 'skyblue'),
        (ax2, 'emissions', 'Carbon Emissions', 'lightgreen'),
        (ax3, 'service', 'Service Level', 'salmon'),
        (ax4, 'backorder', 'Backorder Cost ($)', 'orange'),
        (ax5, 'holding', 'Holding Cost ($)', 'purple')
    ]
    
    for ax, metric, title, color in plot_configs:
        values = [metrics[p][metric] for p in policies]
        ax.bar(x, values, width, color=color)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(policies, rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis for dollar values where applicable
        if 'Cost' in title:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    return fig

def plot_cost_breakdown(results: Dict[str, List[SimulationResults]]) -> plt.Figure:
    """
    Create pie charts showing cost breakdown for each policy.
    """
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 5))
    if len(results) == 1:
        axes = [axes]
    
    # Track all unique cost types and colors for master legend
    all_labels = set()
    label_colors = {}
    
    for ax, (policy_name, policy_results) in zip(axes, results.items()):
        # Calculate average costs across episodes
        costs_by_type = defaultdict(float)
        num_episodes = len(policy_results)
        
        for result in policy_results:
            stats = result.summary_stats
            for cost_type in ['holding_costs', 'backorder_costs', 'setup_costs', 'unit_costs']:
                costs_by_type[cost_type] += stats[cost_type] / num_episodes
        
        # Prepare data for pie chart
        costs = list(costs_by_type.values())
        labels = list(costs_by_type.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(costs)))
        
        # Only include non-zero costs and handle NaN values
        non_zero = [(c, l, col) for c, l, col in zip(costs, labels, colors) 
                   if c > 0 and not np.isnan(c)]
        
        if non_zero:
            costs, labels, colors = zip(*non_zero)
            
            # Update master legend tracking
            for label, color in zip(labels, colors):
                all_labels.add(label)
                label_colors[label] = color
            
            # Create pie chart with safe autopct function
            def make_autopct(values):
                def autopct(pct):
                    # Safely handle percentage calculation
                    idx = int(np.round(pct * len(values) / 100.0))
                    if 0 <= idx < len(values):
                        return f'${values[idx]/1000:.1f}k'
                    return ''
                return autopct
            
            wedges, texts, autotexts = ax.pie(
                costs, 
                colors=colors, 
                labels=[''] * len(costs),
                autopct=make_autopct(costs),
                textprops={'fontsize': 8}
            )
        else:
            # If no valid data, create empty pie chart
            ax.text(0.5, 0.5, 'No cost data', 
                   horizontalalignment='center',
                   verticalalignment='center')
            
        ax.set_title(f'{policy_name} Cost Breakdown')
    
    # Create master legend if we have labels
    if all_labels:
        fig.legend(
            [plt.Rectangle((0,0),1,1, fc=label_colors[label]) for label in sorted(all_labels)],
            sorted(all_labels),
            loc='center', 
            bbox_to_anchor=(0.5, 0), 
            ncol=len(all_labels)
        )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    return fig

def plot_process_analysis(results: Dict[str, SimulationResults]) -> plt.Figure:
    """
    Plot detailed process-by-process analysis of orders and production over time.
    Each policy gets its own row of subplots, one for each process.
    
    Args:
        results: Dictionary mapping policy names to their simulation results
    
    Returns:
        matplotlib Figure object
    """
    num_policies = len(results)
    num_processes = 5  # From the configuration
    
    # Create figure with GridSpec for better control
    fig = plt.figure(figsize=(20, 4 * num_policies))
    gs = GridSpec(num_policies, num_processes, figure=fig)
    
    # Color scheme
    process_colors = sns.color_palette('deep', num_processes)
    
    for policy_idx, (policy_name, result) in enumerate(results.items()):
        states = result.states
        schedule = result.production_schedule
        times = range(len(states))
        
        # Calculate process-specific production
        process_production = {p: [0] * len(states) for p in range(num_processes)}
        cumulative_orders = {p: [0] * len(states) for p in range(num_processes)}
        
        # Calculate cumulative orders and production for each process
        for p in range(num_processes):
            running_orders = 0
            for t in range(len(states)):
                # Update orders
                if t in schedule:
                    running_orders += schedule[t].get(p, 0)
                cumulative_orders[p][t] = running_orders
                
                # Calculate production specific to this process
                if t > 0:
                    if states[t].get_current_process() == p:
                        process_production[p][t] = states[t].total_produced - sum(
                            process_production[i][t-1] for i in range(num_processes)
                        )
                    process_production[p][t] += process_production[p][t-1]
        
        # Find process revision points
        revision_periods = []
        current_process = -1
        for t, state in enumerate(states):
            process = state.get_current_process()
            if process != current_process:
                revision_periods.append((t, process))
                current_process = process
                
        # Create subplot for each process
        for process in range(num_processes):
            ax = fig.add_subplot(gs[policy_idx, process])
            
            # Plot process-specific production (gray dashed line)
            ax.step(times, process_production[process], color='gray', linestyle='--', 
                   alpha=0.5, where='post', label='Process Production')
            
            # Plot process-specific orders (solid line)
            ax.step(times, cumulative_orders[process], color=process_colors[process],
                   where='post', label=f'P{process+1} Orders', linewidth=2)
            
            # Add shaded regions for when this process was active
            for i in range(len(revision_periods)):
                start_t = revision_periods[i][0]
                end_t = revision_periods[i+1][0] if i < len(revision_periods)-1 else len(states)
                active_process = revision_periods[i][1]
                
                if active_process == process:
                    # Draw shaded region up to process production line
                    ax.fill_between(
                        range(start_t, end_t),
                        [0] * (end_t - start_t),
                        process_production[process][start_t:end_t],
                        color=process_colors[process],
                        alpha=0.15,
                        label=f'P{process+1} Active'
                    )
            
            # Add target line if this process was ever active
            if any(period[1] == process for period in revision_periods):
                ax.axhline(y=result.config.total_target, color='red', linestyle=':',
                          alpha=0.5, label='Target')
            
            # Customize appearance
            ax.set_xlim(0, len(states))
            
            # Set y-limits based on process data
            max_y = max(
                max(cumulative_orders[process]),
                max(process_production[process]),
                result.config.total_target if any(period[1] == process for period in revision_periods) else 0
            )
            ax.set_ylim(0, max_y * 1.2 + 1)
            
            # Labels and titles
            if policy_idx == 0:
                ax.set_title(f'Process {process+1}', fontsize=12, pad=10)
            if process == 0:
                ax.set_ylabel(f'{policy_name}\nCumulative Units', fontsize=10)
            if policy_idx == num_policies - 1:
                ax.set_xlabel('Time (weeks)', fontsize=10)
                
            # Add legend only for the rightmost plot
            if process == num_processes - 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                         borderaxespad=0., fontsize=9)
            
            # Grid
            ax.grid(True, alpha=0.3)
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    return fig

def plot_combined_analysis(results: Dict[str, List[SimulationResults]], single_results: Dict[str, SimulationResults]) -> plt.Figure:
    """
    Create a combined plot with bar charts and process analysis.
    
    Args:
        results: Dictionary mapping policy names to lists of simulation results
        single_results: Dictionary mapping policy names to single simulation results
    
    Returns:
        matplotlib Figure object
    """
    num_policies = len(results)
    num_processes = 5
    
    # Create figure with custom GridSpec
    fig = plt.figure(figsize=(25, 20))
    gs = GridSpec(num_policies, 6)  # 6 columns: 1 for bars, 5 for processes
    
    # Color schemes
    process_colors = sns.color_palette('deep', num_processes)
    metrics_colors = {'cost': 'skyblue', 'emissions': 'lightgreen', 
                     'service': 'salmon', 'backorder': 'orange', 
                     'holding': 'purple'}
    
    # Calculate metrics for bar charts
    metrics = {}
    for policy_name, policy_results in results.items():
        metrics[policy_name] = {
            'cost': np.mean([r.total_cost for r in policy_results]),
            'emissions': np.mean([r.carbon_emissions for r in policy_results]),
            'service': np.mean([r.service_level for r in policy_results]),
            'backorder': np.mean([r.summary_stats["backorder_costs"] for r in policy_results]),
            'holding': np.mean([r.summary_stats["holding_costs"] for r in policy_results])
        }
    
    # Plot for each policy row
    for policy_idx, (policy_name, policy_results) in enumerate(results.items()):
        # 1. Bar Charts (First Column)
        ax_bar = fig.add_subplot(gs[policy_idx, 0])
        
        # Plot bars vertically for this policy
        metric_values = [metrics[policy_name][m] for m in ['cost', 'emissions', 'service', 'backorder', 'holding']]
        colors = [metrics_colors[m] for m in ['cost', 'emissions', 'service', 'backorder', 'holding']]
        
        y_pos = np.arange(len(metric_values))
        ax_bar.barh(y_pos, metric_values, color=colors)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(['Total Cost', 'Emissions', 'Service', 'Backorder', 'Holding'])
        
        # Format x-axis for dollar values
        def format_value(x, p):
            if x >= 1e6:
                return f'${x/1e6:.1f}M' if x > 1000 else f'${x:,.0f}'
            elif x >= 1e3:
                return f'${x/1e3:.1f}K' if x > 1000 else f'${x:,.0f}'
            else:
                return f'${x:.0f}' if x > 1 else f'{x:.2f}'
            
        ax_bar.xaxis.set_major_formatter(plt.FuncFormatter(format_value))
        
        ax_bar.set_title(f'{policy_name}\nMetrics')
        ax_bar.grid(True, alpha=0.3)
        
        # 2. Process Analysis (Remaining 5 Columns)
        result = single_results[policy_name]
        states = result.states
        schedule = result.production_schedule
        times = range(len(states))
        
        # Calculate process-specific production and orders
        process_production = {p: [0] * len(states) for p in range(num_processes)}
        cumulative_orders = {p: [0] * len(states) for p in range(num_processes)}
        
        for p in range(num_processes):
            running_orders = 0
            for t in range(len(states)):
                if t in schedule:
                    running_orders += schedule[t].get(p, 0)
                cumulative_orders[p][t] = running_orders
                
                if t > 0:
                    if states[t].get_current_process() == p:
                        process_production[p][t] = states[t].total_produced - sum(
                            process_production[i][t-1] for i in range(num_processes)
                        )
                    process_production[p][t] += process_production[p][t-1]
        
        # Find process revision points
        revision_periods = []
        current_process = -1
        for t, state in enumerate(states):
            process = state.get_current_process()
            if process != current_process:
                revision_periods.append((t, process))
                current_process = process
        
        # Create subplot for each process
        for process in range(num_processes):
            ax = fig.add_subplot(gs[policy_idx, process + 1])
            
            # Plot process-specific production and orders
            ax.step(times, process_production[process], color='gray', 
                   linestyle='--', alpha=0.5, where='post', 
                   label='Process Production')
            ax.step(times, cumulative_orders[process], 
                   color=process_colors[process], where='post',
                   label=f'P{process+1} Orders', linewidth=2)
            
            # Add active process shading
            for i in range(len(revision_periods)):
                start_t = revision_periods[i][0]
                end_t = revision_periods[i+1][0] if i < len(revision_periods)-1 else len(states)
                active_process = revision_periods[i][1]
                
                if active_process == process:
                    ax.fill_between(
                        range(start_t, end_t),
                        [0] * (end_t - start_t),
                        process_production[process][start_t:end_t],
                        color=process_colors[process],
                        alpha=0.15,
                        label=f'P{process+1} Active'
                    )
            
            # Add target line if process was active
            if any(period[1] == process for period in revision_periods):
                ax.axhline(y=result.config.total_target, color='red', 
                          linestyle=':', alpha=0.5, label='Target')
            
            # Customize appearance
            ax.set_xlim(0, len(states))
            max_y = max(
                max(cumulative_orders[process]),
                max(process_production[process]),
                result.config.total_target if any(period[1] == process for period in revision_periods) else 0
            )
            ax.set_ylim(0, max_y * 1.2 + 1)
            
            # Labels and titles
            if policy_idx == 0:
                ax.set_title(f'Process {process+1}', fontsize=12)
            if process == 0:
                ax.set_ylabel('Cumulative Units', fontsize=10)
            if policy_idx == num_policies - 1:
                ax.set_xlabel('Time (weeks)', fontsize=10)
            
            # Legend for rightmost plot
            if process == num_processes - 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                         borderaxespad=0., fontsize=9)
            
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig