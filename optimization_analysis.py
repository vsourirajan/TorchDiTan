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

#remove everything inside the plots dir
os.system(f"rm -rf {plots_dir}/*")
ablation_dir_1 = "/local/vondrick/alper/torchtitan-dit-0/outputs/dit-7b-ablations" #DiT-S
ablation_dirs = [ablation_dir_1]
experiment_dirs = [os.path.join(ablation_dir, dir) for ablation_dir in ablation_dirs for dir in os.listdir(ablation_dir) if os.path.isdir(os.path.join(ablation_dir, dir))]


def get_experiment_config_from_experiment_dir(experiment_dir):
    #find config.json in experiment_dir
    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    out = {
        'mixed_precision_param': config_dict['training']['mixed_precision_param'],
        'batch_size': config_dict['training']['batch_size'],
        'compile': config_dict['training']['compile'],
        'model_flavor': config_dict['model']['flavor'],
        'patch_size': config_dict['model']['patch_size'],
        'activation_checkpoint': config_dict['activation_checkpoint']['mode'],
        'float8_enabled': config_dict.get('float8', {}).get('enable_float8_linear', False),
        'experiment_name': config_dict['metrics']['experiment_name'],
    }

    if config_dict['training']['data_parallel_replicate_degree'] > 1 and config_dict['training']['data_parallel_shard_degree'] > 1:
        out['data_parallel_mode'] = 'hsdp'
    elif config_dict['training']['data_parallel_replicate_degree'] > 1 and config_dict['training']['data_parallel_shard_degree'] == 1:
        out['data_parallel_mode'] = 'ddp'
    elif config_dict['training']['data_parallel_replicate_degree'] == 1 and config_dict['training']['data_parallel_shard_degree'] > 1:
        out['data_parallel_mode'] = 'fsdp'
    else:
        out['data_parallel_mode'] = 'none'

    return out

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
    config_columns = ['mixed_precision_param',
                     'compile', 'model_flavor', 'patch_size', 
                     'float8_enabled', 'activation_checkpoint', 'data_parallel_mode']
    
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
combined_df = filter_incomplete_experiments(combine_config_and_metrics(experiment_dirs))

# breakpoint()
# print("filtering top two batch sizes, current size", len(filtered_incomplete_expimernts_df))

#if activation_checkpoint is 'selective', set to true, otherwise false
# combined_df['activation_checkpoint'] = combined_df['activation_checkpoint'] == 'selective'

# EFFECT_OF = 'compile'
# MEASURE = parse_metric_name('im/s')

def plot_percentage_increase_highest_batch(combined_df, EFFECT_OF, MEASURE, reference_value=None):
    """
    Creates a bar plot showing the percentage increase when applying a specific effect.
    Only keeps the highest batch size version for each configuration.
    
    Args:
        combined_df (pd.DataFrame): Input DataFrame
        EFFECT_OF (str): The column name to study the effect of
        MEASURE (str): The metric to measure
        reference_value: The reference value to compare against for categorical variables
    """
    # Create a copy of the DataFrame
    df = combined_df.copy()
    
    # If reference_value is provided, convert categorical column to boolean
    if reference_value is not None:
        bool_column_name = f"{EFFECT_OF}={reference_value}"
        df[bool_column_name] = (df[EFFECT_OF] == reference_value)
        original_EFFECT_OF = EFFECT_OF
        EFFECT_OF = bool_column_name
        df.drop(columns=[original_EFFECT_OF], inplace=True)

    # Get configuration columns
    config_columns = ['mixed_precision_param', 
                     'batch_size', 'compile', 'model_flavor', 'patch_size', 
                     'float8_enabled', 'activation_checkpoint', 'data_parallel_mode']
    if reference_value is not None:
        config_columns.remove(original_EFFECT_OF)
        config_columns.append(EFFECT_OF)
    
    config_columns.remove(EFFECT_OF)
    config_columns_without_batch = [col for col in config_columns if col != 'batch_size']

    print("table size before filtering", len(df))
    # Filter to keep only the highest batch size for each configuration
    filtered_df = pd.DataFrame()
    for _, group in df.groupby(config_columns_without_batch + [EFFECT_OF]):
        max_batch = group['batch_size'].max()
        filtered_df = pd.concat([filtered_df, group[group['batch_size'] == max_batch]])

    print("table size after filtering", len(filtered_df))
# 
    # breakpoint()

    plt.figure(figsize=(12, 6))

    # Group by all configuration columns except EFFECT_OF and batch_size
    grouped = filtered_df.groupby(config_columns_without_batch)
    paired_configs = []
    
    for name, group in grouped:
        # breakpoint()
        if len(group) >= 2:  # Must have both with and without effect
            if group[EFFECT_OF].nunique() == 2:
                paired_configs.append(name)
        else:
            print(f"Skipping {name} because it has {len(group)} runs")

    if not paired_configs:
        raise ValueError("No paired configurations found!")

    percent_increases = []
    labels = []

    for config in paired_configs:
        group = grouped.get_group(config)
        # print("length of group", len(group))
        # breakpoint()
        # breakpoint()
        
        with_effect = group[group[EFFECT_OF] == True][f'latest_{MEASURE}'].values[0]
        without_effect = group[group[EFFECT_OF] == False][f'latest_{MEASURE}'].values[0]
        
        percent_increase = ((with_effect - without_effect) / without_effect) * 100
        percent_increases.append(percent_increase)
        
        labels.append(group['experiment_name'].iloc[0])

    x = np.arange(len(paired_configs))
    plt.bar(x, percent_increases, color='lightblue')

    plt.xlabel('Configuration')
    plt.ylabel(f'Percentage Increase in {MEASURE} (%)')
    plt.title(f'Percentage Increase in {MEASURE} with {EFFECT_OF} (Highest Batch Size)')
    plt.xticks(x, labels, rotation=45, ha='right')

    for i, v in enumerate(percent_increases):
        plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom')

    plt.tight_layout()

    name = f"percent_increase_highest_batch_{EFFECT_OF}_on_{MEASURE}"
    plt.savefig(os.path.join(plots_dir, f"{name.replace('/', '_')}.png"))
    plt.show()

def plot_percentage_increase(combined_df, EFFECT_OF, MEASURE, reference_value=None):
    """
    Creates a bar plot showing the percentage increase when applying a specific effect.
    Only keeps the highest batch size version when configurations are otherwise identical.
    
    Args:
        combined_df (pd.DataFrame): Input DataFrame
        EFFECT_OF (str): The column name to study the effect of
        MEASURE (str): The metric to measure
        reference_value: The reference value to compare against for categorical variables
    """
    # Create a copy of the DataFrame
    df = combined_df.copy()
    
    # If reference_value is provided, convert categorical column to boolean
    if reference_value is not None:
        # Create new boolean column name
        bool_column_name = f"{EFFECT_OF}={reference_value}"
        
        # Create boolean column based on reference value
        df[bool_column_name] = (df[EFFECT_OF] == reference_value)
        
        # Store original EFFECT_OF value
        original_EFFECT_OF = EFFECT_OF
        
        # Update EFFECT_OF to use new boolean column
        EFFECT_OF = bool_column_name

        #drop the original EFFECT_OF column
        df.drop(columns=[original_EFFECT_OF], inplace=True)

    # Get configuration columns excluding the effect being studied and batch_size
    config_columns = ['mixed_precision_param', 
                     'batch_size', 'compile', 'model_flavor', 'patch_size', 
                     'float8_enabled', 'activation_checkpoint', 'data_parallel_mode']
    if reference_value is not None:
        config_columns.remove(original_EFFECT_OF)  # Remove original categorical column
        config_columns.append(EFFECT_OF)  # Add new boolean column
        
    config_columns.remove(EFFECT_OF)
    print("config_columns", config_columns)
    config_columns_without_batch = [col for col in config_columns if col != 'batch_size']

    # Rest of your existing function remains the same...
    filtered_df = pd.DataFrame()
    for _, group in df.groupby(config_columns_without_batch):
        batch_sizes = sorted(group['batch_size'].unique(), reverse=True)
        for batch_size in batch_sizes:
            if reference_value is None:
                if len(group[group['batch_size'] == batch_size]) == 2:
                    assert group[group['batch_size'] == batch_size][EFFECT_OF].nunique() == 2
                    filtered_df = pd.concat([filtered_df, group[group['batch_size'] == batch_size]])
                    break
            else:
                if len(group[group['batch_size'] == batch_size]) >= 2 and len(group[(group[EFFECT_OF] == True) & (group['batch_size'] == batch_size)]) == 1:
                    # print("adding batch size", batch_size, "len group", len(group[group['batch_size'] == batch_size]))
                    # # breakpoint()
                    filtered_df = pd.concat([filtered_df, group[group['batch_size'] == batch_size]])
                    break
                    
    plt.figure(figsize=(12, 6))

    grouped = filtered_df.groupby(config_columns)
    paired_configs = []
    for name, group in grouped:
        if len(group) >= 2:
            if len(group) > 2:
                assert reference_value is not None
            paired_configs.append(name)
        else:
            print(f"Skipping {name} because it has {len(group)} runs")

    if not paired_configs:
        raise ValueError("No paired configurations found!")

    percent_increases = []
    labels = []

    for config in paired_configs:
        group = grouped.get_group(config)

        assert len(group[group[EFFECT_OF] == True][f'latest_{parse_metric_name(MEASURE)}'].values) == 1
        with_effect = group[group[EFFECT_OF] == True][f'latest_{parse_metric_name(MEASURE)}'].values[0]
        without_effect = group[group[EFFECT_OF] == False][f'latest_{parse_metric_name(MEASURE)}'].values.max()
        
        percent_increase = ((with_effect - without_effect) / without_effect) * 100
        percent_increases.append(percent_increase)
        
        labels.append(group['experiment_name'].iloc[0])

    x = np.arange(len(paired_configs))
    plt.bar(x, percent_increases, color='lightblue')

    plt.xlabel('Configuration')
    plt.ylabel(f'Percentage Increase in {MEASURE} (%)')
    plt.title(f'Percentage Increase in {MEASURE} with {EFFECT_OF}')
    plt.xticks(x, labels, rotation=45, ha='right')

    for i, v in enumerate(percent_increases):
        plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom')

    plt.tight_layout()

    name = f"percent_increase_{EFFECT_OF}_on_{MEASURE}"
    plt.savefig(os.path.join(plots_dir, f"{name.replace('/', '_')}.png"))
    plt.show()

def plot_interaction_effects(combined_df, EFFECT_OF, MEASURE, reference_value=None):
    """
    Creates a bar plot showing how different configuration changes interact with the EFFECT_OF.
    
    Args:
        combined_df (pd.DataFrame): The processed experiment data
        EFFECT_OF (str): The effect being studied (e.g., 'compile')
        MEASURE (str): The metric to measure (e.g., 'im_s')
    """
    # Get configuration columns excluding the effect being studied
    config_columns = ['mixed_precision_param', 
                     'batch_size', 'compile', 'model_flavor', 'patch_size', 
                     'float8_enabled', 'activation_checkpoint', 'data_parallel_mode']
    config_columns_without_batch = [col for col in config_columns if col != 'batch_size']

    df = combined_df.copy()

    if reference_value is not None:
        bool_column_name = f"{EFFECT_OF}={reference_value}"

        config_columns_without_batch.remove(EFFECT_OF)

        config_columns_without_batch.append(bool_column_name)

        df[bool_column_name] = (df[EFFECT_OF] == reference_value)
        
        df.drop(columns=[EFFECT_OF], inplace=True)
        EFFECT_OF = bool_column_name

    # First, filter to keep only the highest batch size for each configuration
    filtered_df = pd.DataFrame()
    for _, group in df.groupby(config_columns_without_batch):
        max_batch = group['batch_size'].max()
        filtered_df = pd.concat([filtered_df, group[group['batch_size'] == max_batch]])

    # Dictionary to store effect differences
    effect_differences = {}

    config_columns_without_batch.remove(EFFECT_OF)
    
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

                if improvement_val1 is not None and improvement_val2 is not None:
                    effect_differences[effect_name] = improvement_val2 - improvement_val1
                    effect_differences[f"{col}: {val2}->{val1}"] = improvement_val1 - improvement_val2

    # Sort effects by their impact and filter for positive values only
    sorted_effects = sorted([(k, v) for k, v in effect_differences.items() if v > 0], 
                          key=lambda x: x[1])
    
    if not sorted_effects:  # Check if we have any positive effects
        print(f"No positive interaction effects found for {EFFECT_OF}")
        return
        
    effects, differences = zip(*sorted_effects)

    # Create the plot
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(differences)), differences)
    
    # Color all bars lightblue since they're all positive
    for bar in bars:
        bar.set_color('lightblue')

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

for metric in ['im/s', 'mfu', 'memory_max_reserved']:
    for effect, reference_value in [('float8_enabled', None), ('data_parallel_mode', 'fsdp'), ('data_parallel_mode', 'hsdp'), ('data_parallel_mode', 'ddp'), ('compile', None), ('activation_checkpoint', 'selective')]:
        print("plotting for", effect)
        if reference_value is not None and combined_df[combined_df[effect] == reference_value].empty:
            print("skipping", effect, reference_value)
            continue
        print("columns", combined_df.columns)
        # plot_percentage_increase(combined_df, effect, parse_metric_name(metric), reference_value=reference_value)
        plot_interaction_effects(combined_df, effect, parse_metric_name(metric), reference_value=reference_value)
        plot_percentage_increase_highest_batch(combined_df, effect, parse_metric_name(metric), reference_value=reference_value)
