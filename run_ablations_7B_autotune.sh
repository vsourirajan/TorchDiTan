#!/bin/bash

# Get start datetime
start_datetime=$(date +"%Y%m%d_%H%M%S")

#Put parameters in decreasing memory footprint
# Arrays for each parameter to ablate
dtypes=("bfloat16_f8" "bfloat16")  # Plus one more with bfloat16_f8
parallel_modes=("1,8" "2,4" "8,1")  # Format: "replicate,shard"
model_sizes=("DiT-7B")
patch_sizes=(2 4 8)
activation_checkpointing=("none" "selective")
compile_modes=("false" "true")

# Calculate total experiments
total_experiments=$((${#dtypes[@]} * ${#parallel_modes[@]} * ${#model_sizes[@]} * ${#patch_sizes[@]} * ${#activation_checkpointing[@]} * ${#compile_modes[@]}))
# Initialize progress tracking
current_experiment=0
progress_file="outputs/ablations_autotune_${start_datetime}/progress.log"
mkdir -p "outputs/ablations_autotune_${start_datetime}"

# Add header to progress file
echo -e "Experiment#\tTotal\tProgress\tExperiment_Name\tDuration\tAvg_Time_Per_Exp\tEstimated_Completion\tOptimal_Batch_Size" > "$progress_file"

echo "Total number of experiments to run: $total_experiments"
echo "Start time: $start_datetime"
echo "Progress will be logged to: $progress_file"

# Base commands
BASE_CMD="NGPU=8 ./train_latent_diffusion_ablate.sh"
AUTOTUNE_CMD="NGPU=8 bash autotune_bs_train_latent_diffusion.sh"

state_file=$(mktemp)
echo "$last_attempted_bs" > "$state_file"

# Function to run experiment
run_experiment() {

    # Declare global variable
    local last_attempted_bs=$(cat "$state_file")
    local optimal_batch_size=""

    echo "🔄  Starting autotuning for configuration... with optimal batch size: ($optimal_batch_size) and last attempted batch size: ($last_attempted_bs)"

    local dtype=$1
    local parallel=$2
    local model=$3
    local patch=$4
    local activ=$5
    local do_compile=$6
    
    # Start timing
    local start_time=$(date +%s)
    
    # Increment experiment counter
    ((current_experiment++))
    
    # Calculate progress percentage
    local progress=$((current_experiment * 100 / total_experiments))
    
    # Parse parallel configuration
    IFS=',' read -r replicate_degree shard_degree <<< "$parallel"
    
    # Construct command with modified dtype handling
    local cmd_dtype="$dtype"
    if [ "$dtype" = "bfloat16_f8" ]; then
        cmd_dtype="bfloat16"
    fi
    
    # First run autotuning to get optimal batch size
    local autotune_cmd="$AUTOTUNE_CMD \
        --training.mixed_precision_param=$cmd_dtype \
        --training.data_parallel_replicate_degree=$replicate_degree \
        --training.data_parallel_shard_degree=$shard_degree \
        --model.flavor=$model \
        --model.patch_size=$patch \
        --metrics.experiment_name=autotune_bs \
        --activation_checkpoint.mode=$activ \
        --metrics.log_freq=5"

    # Add float8 flag if needed
    if [ "$dtype" = "bfloat16_f8" ]; then
        autotune_cmd="$autotune_cmd --float8.enable_float8_linear --autotune.batch_size_multiple=16"
    fi

    # Add compile flag if true
    if [ "$do_compile" = "true" ]; then
        autotune_cmd="$autotune_cmd --training.compile"
    fi

    if [ ! -z "$last_attempted_bs" ]; then
        autotune_cmd="$autotune_cmd --training.batch_size=$last_attempted_bs"
        echo "Starting from last optimal batch size: $last_attempted_bs"
    fi

    local optimal_bs_file=$(mktemp)

    echo "Running autotuning for configuration..."
    eval "$autotune_cmd" 2>&1 | while IFS= read -r line; do
        echo "$line"
        if [[ $line == *"Binary search complete. Optimal batch size found"* ]]; then
            optimal_batch_size=$(echo $line | grep -o '[0-9]\+$')
            echo "🎯 Optimal batch size found: $optimal_batch_size"
            echo "$optimal_batch_size" > "$optimal_bs_file"
        fi
        # Capture the last attempted batch size
        if [[ $line == *"Attempting training with batch size:"* ]]; then
            last_attempted_bs=$(echo $line | grep -o '[0-9]\+$')
            echo "$last_attempted_bs" > "$state_file"
        fi
    done

    # Read the optimal batch size from the temp file
    optimal_batch_size=$(cat "$optimal_bs_file")
    rm -f "$optimal_bs_file"  # Clean up

    if [ -z "$optimal_batch_size" ]; then
        echo "🔄 ERROR: Autotuning failed to find valid batch size"
        return 1
    else
        echo "✅ Valid batch size found: $optimal_batch_size"
    fi


    # Update last_optimal_bs with the last attempted batch size instead of the optimal one
    last_attempted_bs=$last_attempted_bs
    
    # Create simplified experiment name (lowercase model name only)
    exp_name=$(echo "$model" | tr '[:upper:]' '[:lower:]' | sed 's/DiT-//')"_bs${last_attempted_bs}_ps${patch}_act${activ}_c${do_compile}_d${dtype}_p${parallel}"    

    # Create TB folder with datetime
    tb_folder="ablations_autotune_${start_datetime}"


    cmd="$BASE_CMD \
        --training.mixed_precision_param=$cmd_dtype \
        --training.data_parallel_replicate_degree=$replicate_degree \
        --training.data_parallel_shard_degree=$shard_degree \
        --model.flavor=$model \
        --model.patch_size=$patch \
        --training.batch_size=$optimal_batch_size \
        --activation_checkpoint.mode=$activ \
        --metrics.experiment_name=$exp_name \
        --metrics.enable_tensorboard \
        --metrics.save_tb_folder=$tb_folder \
        --metrics.log_freq=5"

    # Add float8 flag if needed
    if [ "$dtype" = "bfloat16_f8" ]; then
        cmd="$cmd --float8.enable_float8_linear"
    fi

    # Add compile flag if true
    if [ "$do_compile" = "true" ]; then
        cmd="$cmd --training.compile"
    fi
    
    echo "Running experiment: $exp_name (saving to $tb_folder)"
    echo "$cmd"
    
    # Run the command and capture output
    eval "$cmd" 2>&1
    exit_code=$?
    
    # End timing
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Calculate average time per experiment
    local start_time_seconds=$(date -d "${start_datetime:0:8} ${start_datetime:9:2}:${start_datetime:11:2}:${start_datetime:13:2}" +%s)
    local avg_time_per_exp=$(( (end_time - start_time_seconds) / current_experiment )) 

    # Estimate time remaining
    local remaining_experiments=$((total_experiments - current_experiment))
    local estimated_remaining_seconds=$((remaining_experiments * avg_time_per_exp))
    local estimated_completion_time=$(date -d "@$(($(date +%s) + estimated_remaining_seconds))" '+%Y-%m-%d %H:%M:%S')
    
    # Write single line to progress file with tab separation
    echo -e "${current_experiment}\t${total_experiments}\t${progress}%\t${exp_name}\t${duration}s\t${avg_time_per_exp}s\t${estimated_completion_time}\t${optimal_batch_size}" >> "$progress_file"
    
    # Wait for the process to finish
    wait
    
    echo "🔄 Returning with last attempted batch size: $last_attempted_bs"
    return $last_attempted_bs
}

# Run all combinations
for dtype in "${dtypes[@]}"; do
    for parallel in "${parallel_modes[@]}"; do
        for model in "${model_sizes[@]}"; do
            for patch in "${patch_sizes[@]}"; do
                for activ in "${activation_checkpointing[@]}"; do
                    for do_compile in "${compile_modes[@]}"; do
                        run_experiment "$dtype" "$parallel" "$model" "$patch" "$activ" "$do_compile"
                    done
                done
            done
        done
    done
done

echo "🔄 All experiments completed"
#remove state file
rm -f "$state_file"
