from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import os 
import json
import matplotlib.pyplot as plt
import numpy as np

# Create plots directory if it doesn't exist
plots_dir = "./plots"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

ablation_dir = "/local/vondrick/alper/torchtitan-dit-0/outputs/ablations_20241215_205446"
experiment_dirs = [os.path.join(ablation_dir, dir) for dir in os.listdir(ablation_dir) if os.path.isdir(os.path.join(ablation_dir, dir))]


def get_experiment_config_from_experiment_dir(experiment_dir):
    #find config.json in experiment_dir
    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    return {
        'mixed_precision_param': config_dict['training']['mixed_precision_param'],
        'replicate_degree': config_dict['training']['data_parallel_replicate_degree'],
        'shard_degree': config_dict['training']['data_parallel_shard_degree'],
        'batch_size': config_dict['training']['batch_size'],
        'compile': config_dict['training']['compile'],
        'model_flavor': config_dict['model']['flavor'],
        'patch_size': config_dict['model']['patch_size'],
        'activation_checkpoint': config_dict['activation_checkpoint']['mode'],
        'float8_enabled': config_dict.get('float8', {}).get('enable_float8_linear', False),
        'experiment_name': config_dict['metrics']['experiment_name'],
        'hsdp': config_dict['training']['data_parallel_replicate_degree'] > 1 and config_dict['training']['data_parallel_shard_degree'] > 1,
        'fsdp': config_dict['training']['data_parallel_replicate_degree'] > 1 and config_dict['training']['data_parallel_shard_degree'] == 1,
        'ddp': config_dict['training']['data_parallel_replicate_degree'] == 1 and config_dict['training']['data_parallel_shard_degree'] > 1,
    }

measured_metrics = [
    'it/s',
    'im/s',
    'mfu(%)',
    'memory/max_reserved(%)'
]

def read_tensorboard_logs(log_dir):

   
    # Load the TensorBoard data
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Get list of tags (different metrics)
    tags = event_acc.Tags()['scalars']
    
    # Create a dictionary to store all metrics
    data = {}
    
    # Read each metric
    for tag in tags:
        events = event_acc.Scalars(tag)
        data[tag] = pd.DataFrame([
            {'step': event.step, 'value': event.value, 'wall_time': event.wall_time}
            for event in events
        ])
    
    return data

def parse_metric_name(metric_name):
    return metric_name.replace("(%)", "").replace("/", "_")

def combine_config_and_metrics(experiment_dirs):
    all_experiments = []
    
    for experiment_dir in experiment_dirs:
        # Get experiment config
        config = get_experiment_config_from_experiment_dir(experiment_dir)
        
        # Get metrics from tensorboard
        metrics = read_tensorboard_logs(experiment_dir)
        
        # Get value at latest step for each metric
        metric_latest = {}
        for metric_name in ['it/s', 'im/s', 'mfu(%)', 'memory/max_reserved(%)']:
            if metric_name in metrics:
                # Get the row with the maximum step
                latest_row = metrics[metric_name].loc[metrics[metric_name]['step'].idxmax()]
                metric_latest[f'latest_{parse_metric_name(metric_name)}'] = \
                    latest_row['value']
                # Also store the step number for verification
                metric_latest[f'step_{parse_metric_name(metric_name)}'] = \
                    latest_row['step']
        
        # Combine config and metric values
        experiment_data = {**config, **metric_latest}
        all_experiments.append(experiment_data)
    
    # Create DataFrame from all experiments
    return pd.DataFrame(all_experiments)

def filter_incomplete_experiments(df):
    """
    Filter out experiments that have NaN values or step counts lower than the mode.
    
    Args:
        df (pd.DataFrame): DataFrame containing experiment results
        
    Returns:
        pd.DataFrame: Filtered DataFrame containing only complete experiments
    """
    # Get all step columns
    step_columns = [col for col in df.columns if col.startswith('step_')]
    
    # Calculate mode for each step column
    step_modes = {col: df[col].mode()[0] for col in step_columns}
    
    # Create a mask for rows that meet both conditions:
    # 1. No NaN values in any column
    # 2. Step values are greater than or equal to their respective modes
    mask = ~df.isna().any(axis=1)  # Start with no NaN condition
    for col in step_columns:
        mask = mask & (df[col] >= step_modes[col])
    
    # Apply the filter
    filtered_df = df[mask]
    
    return filtered_df

def filter_top_two_batch_sizes(df):
    """
    For each configuration, keeps only the two highest batch sizes.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: Filtered DataFrame with only top two batch sizes per config
    """
    # Get configuration columns excluding batch_size
    config_columns = ['mixed_precision_param', 'replicate_degree', 'shard_degree', 
                     'compile', 'model_flavor', 'patch_size', 
                     'float8_enabled', 'activation_checkpoint']
    
    # Initialize empty DataFrame for results
    filtered_df = pd.DataFrame()
    
    # Group by all configuration parameters except batch size
    for _, group in df.groupby(config_columns):
        # Get unique batch sizes and sort them in descending order
        batch_sizes = sorted(group['batch_size'].unique(), reverse=True)
        
        # Keep only rows with the top two batch sizes
        if len(batch_sizes) >= 2:
            top_two_batches = batch_sizes[:2]
            group_filtered = group[group['batch_size'].isin(top_two_batches)]
            filtered_df = pd.concat([filtered_df, group_filtered])
        else:
            # If there's only one batch size, keep it
            filtered_df = pd.concat([filtered_df, group])
    
    return filtered_df

# Create the combined DataFrame
filtered_incomplete_expimernts_df = filter_incomplete_experiments(combine_config_and_metrics(experiment_dirs))
print("filtering top two batch sizes, current size", len(filtered_incomplete_expimernts_df))
combined_df = filter_top_two_batch_sizes(filtered_incomplete_expimernts_df)
print("after filtering top two batch sizes, current size", len(combined_df))

#if activation_checkpoint is 'selective', set to true, otherwise false
combined_df['activation_checkpoint'] = combined_df['activation_checkpoint'] == 'selective'

EFFECT_OF = 'compile'
MEASURE = parse_metric_name('im/s')

def plot_paired_bar_comparison(combined_df, EFFECT_OF, MEASURE):
    """
    Creates a bar plot comparing configurations with and without a specific effect.
    Only keeps the highest batch size version when configurations are otherwise identical.
    """
    # Get configuration columns excluding the effect being studied and batch_size
    config_columns = ['mixed_precision_param', 'replicate_degree', 'shard_degree', 
                     'batch_size', 'compile', 'model_flavor', 'patch_size', 
                     'float8_enabled', 'activation_checkpoint']
    config_columns.remove(EFFECT_OF)
    config_columns_without_batch = [col for col in config_columns if col != 'batch_size']

    # First, group by everything except batch size and EFFECT_OF
    # For each group, keep only the highest batch size
    filtered_df = pd.DataFrame()
    for _, group in combined_df.groupby(config_columns_without_batch):
        print("new group")
        #sort unique batch sizes and start descending until finding a pair
        batch_sizes = sorted(group['batch_size'].unique(), reverse=True)
        for batch_size in batch_sizes:
            if len(group[group['batch_size'] == batch_size]) == 2:
                filtered_df = pd.concat([filtered_df, group[group['batch_size'] == batch_size]])
                break
                

    # Check for duplicate configurations
    config_counts = filtered_df.groupby(config_columns).size()
    if (config_counts > 2).any():
        raise ValueError("Found duplicate configurations! Each configuration should appear at most twice (with/without effect)")

    # Create a figure with larger size
    plt.figure(figsize=(12, 6))

    # Group by configuration and keep only paired runs
    grouped = filtered_df.groupby(config_columns)
    paired_configs = []
    for name, group in grouped:
        if len(group) == 2:
            paired_configs.append(name)
        else:
            print(f"Skipping {name} because it has {len(group)} runs")

    if not paired_configs:
        raise ValueError("No paired configurations found!")

    heights_with = []
    heights_without = []
    labels = []

    for config in paired_configs:
        group = grouped.get_group(config)
        
        # Get the measurements
        with_effect = group[group[EFFECT_OF] == True][f'latest_{parse_metric_name(MEASURE)}'].values[0]
        without_effect = group[group[EFFECT_OF] == False][f'latest_{parse_metric_name(MEASURE)}'].values[0]
        
        heights_with.append(with_effect)
        heights_without.append(without_effect)
        
        labels.append(group['experiment_name'].iloc[0])

    # Plot bars
    bar_width = 0.35
    x = np.arange(len(paired_configs))

    plt.bar(x - bar_width/2, heights_without, bar_width, label=f'Without {EFFECT_OF}', color='lightcoral')
    plt.bar(x + bar_width/2, heights_with, bar_width, label=f'With {EFFECT_OF}', color='lightblue')

    # Customize the plot
    plt.xlabel('Configuration')
    plt.ylabel(MEASURE)
    plt.title(f'Effect of {EFFECT_OF} on {MEASURE}')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend()

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save and show the plot
    name = f"effect_of_{EFFECT_OF}_on_{MEASURE}"
    plt.savefig(os.path.join(plots_dir, f"{name.replace('/', '_')}.png"))
    plt.show()

def plot_percentage_increase(combined_df, EFFECT_OF, MEASURE):
    """
    Creates a bar plot showing the percentage increase when applying a specific effect.
    Only keeps the highest batch size version when configurations are otherwise identical.
    """
    # Get configuration columns excluding the effect being studied and batch_size
    config_columns = ['mixed_precision_param', 'replicate_degree', 'shard_degree', 
                     'batch_size', 'compile', 'model_flavor', 'patch_size', 
                     'float8_enabled', 'activation_checkpoint']
    config_columns.remove(EFFECT_OF)
    config_columns_without_batch = [col for col in config_columns if col != 'batch_size']

    # First, group by everything except batch size and EFFECT_OF
    # For each group, keep only the highest batch size
    filtered_df = pd.DataFrame()
    for _, group in combined_df.groupby(config_columns_without_batch):
        batch_sizes = sorted(group['batch_size'].unique(), reverse=True)
        for batch_size in batch_sizes:
            if len(group[group['batch_size'] == batch_size]) == 2:
                filtered_df = pd.concat([filtered_df, group[group['batch_size'] == batch_size]])
                break

    # Create a figure with larger size
    plt.figure(figsize=(12, 6))

    # Group by configuration and keep only paired runs
    grouped = filtered_df.groupby(config_columns)
    paired_configs = []
    for name, group in grouped:
        if len(group) == 2:
            paired_configs.append(name)
        else:
            print(f"Skipping {name} because it has {len(group)} runs")

    if not paired_configs:
        raise ValueError("No paired configurations found!")

    percent_increases = []
    labels = []

    for config in paired_configs:
        group = grouped.get_group(config)
        
        # Get the measurements
        with_effect = group[group[EFFECT_OF] == True][f'latest_{parse_metric_name(MEASURE)}'].values[0]
        without_effect = group[group[EFFECT_OF] == False][f'latest_{parse_metric_name(MEASURE)}'].values[0]
        
        # Calculate percentage increase
        percent_increase = ((with_effect - without_effect) / without_effect) * 100
        percent_increases.append(percent_increase)
        
        labels.append(group['experiment_name'].iloc[0])

    # Plot bars
    x = np.arange(len(paired_configs))
    plt.bar(x, percent_increases, color='lightblue')

    # Customize the plot
    plt.xlabel('Configuration')
    plt.ylabel(f'Percentage Increase in {MEASURE} (%)')
    plt.title(f'Percentage Increase in {MEASURE} with {EFFECT_OF}')
    plt.xticks(x, labels, rotation=45, ha='right')

    # Add percentage values on top of bars
    for i, v in enumerate(percent_increases):
        plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save and show the plot
    name = f"percent_increase_{EFFECT_OF}_on_{MEASURE}"
    plt.savefig(os.path.join(plots_dir, f"{name.replace('/', '_')}.png"))
    plt.show()

def plot_batch_size_scaling(combined_df):
    """
    Creates line plots showing how metrics scale with batch size for different configurations.
    
    Args:
        combined_df (pd.DataFrame): The processed experiment data
    """
    # Create subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()

    # Group everything except batch size
    group_columns = ['mixed_precision_param', 'replicate_degree', 'shard_degree',
                    'compile', 'model_flavor', 'patch_size', 'float8_enabled', 
                    'activation_checkpoint']

    for idx, metric in enumerate(measured_metrics):
        metric_name = f'latest_{parse_metric_name(metric)}'
        
        # Group by all parameters except batch size
        grouped = combined_df.groupby(group_columns)
        
        # Plot each configuration as a separate line
        for name, group in grouped:
            # Sort by batch size
            group_sorted = group.sort_values('batch_size')
            
            # Create more compact label
            config_dict = dict(zip(group_columns, name))
            label_parts = []
            if config_dict['mixed_precision_param'] != 'fp32':
                label_parts.append(f"mp={config_dict['mixed_precision_param']}")
            if config_dict['replicate_degree'] > 1:
                label_parts.append(f"rep={config_dict['replicate_degree']}")
            if config_dict['shard_degree'] > 1:
                label_parts.append(f"sh={config_dict['shard_degree']}")
            if config_dict['compile']:
                label_parts.append("comp")
            if config_dict['float8_enabled']:
                label_parts.append("f8")
            if config_dict['activation_checkpoint']:
                label_parts.append("ckpt")
            
            label = ','.join(label_parts)
            
            # Plot
            axes[idx].plot(group_sorted['batch_size'], 
                         group_sorted[metric_name], 
                         marker='o', 
                         label=label)
        
        axes[idx].set_xlabel('Batch Size')
        axes[idx].set_ylabel(metric)
        axes[idx].set_title(f'{metric} vs Batch Size')
        axes[idx].grid(True)
        # Put legend inside the plot with smaller font
        axes[idx].legend(loc='best', fontsize='small')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'batch_size_scaling.png'), bbox_inches='tight')
    plt.show()

def plot_interaction_effects(combined_df, EFFECT_OF, MEASURE):
    """
    Creates a bar plot showing how different configuration changes interact with the EFFECT_OF.
    
    Args:
        combined_df (pd.DataFrame): The processed experiment data
        EFFECT_OF (str): The effect being studied (e.g., 'compile')
        MEASURE (str): The metric to measure (e.g., 'im_s')
    """
    # Get configuration columns excluding the effect being studied
    config_columns = ['mixed_precision_param', 'replicate_degree', 'shard_degree', 
                     'batch_size', 'compile', 'model_flavor', 'patch_size', 
                     'float8_enabled', 'activation_checkpoint']
    config_columns.remove(EFFECT_OF)
    config_columns_without_batch = [col for col in config_columns if col != 'batch_size']

    # First, filter to keep only the highest batch size for each configuration
    filtered_df = pd.DataFrame()
    for _, group in combined_df.groupby(config_columns_without_batch):
        max_batch = group['batch_size'].max()
        filtered_df = pd.concat([filtered_df, group[group['batch_size'] == max_batch]])
    
    # Dictionary to store effect differences
    effect_differences = {}
    
    # Process each configuration column (except batch_size)
    for col in config_columns_without_batch:
        unique_values = sorted(filtered_df[col].unique())
        
        # For each pair of values in this column
        for i in range(len(unique_values)):
            for j in range(i + 1, len(unique_values)):
                val1, val2 = unique_values[i], unique_values[j]
                effect_name = f"{col}: {val1}->{val2}"

                print("effect_name", effect_name)
                
                # Calculate average improvement for val1
                mask_val1 = filtered_df[col] == val1
                val1_with = filtered_df[mask_val1 & (filtered_df[EFFECT_OF] == True)][f'latest_{MEASURE}'].mean()
                val1_without = filtered_df[mask_val1 & (filtered_df[EFFECT_OF] == False)][f'latest_{MEASURE}'].mean()
                improvement_val1 = ((val1_with - val1_without) / val1_without) * 100 if not pd.isna(val1_with) and not pd.isna(val1_without) else None
                
                # Calculate average improvement for val2
                mask_val2 = filtered_df[col] == val2
                val2_with = filtered_df[mask_val2 & (filtered_df[EFFECT_OF] == True)][f'latest_{MEASURE}'].mean()
                val2_without = filtered_df[mask_val2 & (filtered_df[EFFECT_OF] == False)][f'latest_{MEASURE}'].mean()
                improvement_val2 = ((val2_with - val2_without) / val2_without) * 100 if not pd.isna(val2_with) and not pd.isna(val2_without) else None
                
                print("\t improvement_val1", improvement_val1)
                print("\t improvement_val2", improvement_val2)

                if improvement_val1 is not None and improvement_val2 is not None:
                    effect_differences[effect_name] = improvement_val2 - improvement_val1
                    effect_differences[f"{col}: {val2}->{val1}"] = improvement_val1 - improvement_val2

    # Sort effects by their impact
    sorted_effects = sorted(effect_differences.items(), key=lambda x: x[1])
    effects, differences = zip(*sorted_effects)

    # Create the plot
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(differences)), differences)
    
    # Color positive and negative bars differently
    for i, diff in enumerate(differences):
        bars[i].set_color('lightblue' if diff >= 0 else 'lightcoral')

    plt.xticks(range(len(effects)), effects, rotation=45, ha='right')
    plt.xlabel('Configuration Change')
    plt.ylabel(f'Difference in {EFFECT_OF} Improvement (%)')
    plt.title(f'Impact of Configuration Changes on {EFFECT_OF} Effectiveness')
    
    # Add value labels on top of bars
    for i, v in enumerate(differences):
        plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom' if v >= 0 else 'top')

    plt.grid(True, axis='y')
    plt.tight_layout()
    
    # Save and show the plot
    name = f"interaction_effects_{EFFECT_OF}_on_{MEASURE}"
    plt.savefig(os.path.join(plots_dir, f"{name.replace('/', '_')}.png"))
    plt.show()

# Example usage:
# plot_paired_bar_comparison(combined_df, 'compile', parse_metric_name('im/s'))

for effect in ['compile', 'activation_checkpoint', 'mixed_precision_param']:
    plot_percentage_increase(combined_df, effect, parse_metric_name('im/s'))
    plot_interaction_effects(combined_df, effect, parse_metric_name('im/s'))
    plot_batch_size_scaling(combined_df)