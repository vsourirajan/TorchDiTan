#!/bin/bash

# Get start datetime
start_datetime=$(date +"%Y%m%d_%H%M%S")

# Arrays for each parameter to ablate
dtypes=("bfloat16")  # Plus one more with bfloat16_f8
parallel_modes=("1,8" "8,1" "2,4")  # Format: "replicate,shard"
model_sizes=("DiT-L")
patch_sizes=(8 4 2)
batch_sizes=(1 2 4 8 16 32 64 128 256 512 1024)
activation_checkpointing=("none" "selective")
compile_modes=("true" "false")

# Calculate total experiments
total_experiments=$((${#dtypes[@]} * ${#parallel_modes[@]} * ${#model_sizes[@]} * ${#patch_sizes[@]} * ${#batch_sizes[@]} * ${#activation_checkpointing[@]} * ${#compile_modes[@]} + ${#parallel_modes[@]} * ${#model_sizes[@]} * ${#patch_sizes[@]} * ${#batch_sizes[@]} * ${#activation_checkpointing[@]} * ${#compile_modes[@]}))

# Initialize progress tracking
current_experiment=0
progress_file="outputs/ablations_${start_datetime}/progress.log"
mkdir -p "outputs/ablations_${start_datetime}"

# Add header to progress file
echo -e "Experiment#\tTotal\tProgress\tExperiment_Name\tDuration\tAvg_Time_Per_Exp\tEstimated_Completion" > "$progress_file"

echo "Total number of experiments to run: $total_experiments"
echo "Start time: $start_datetime"
echo "Progress will be logged to: $progress_file"

# Base command
BASE_CMD="NGPU=8 ./train_latent_diffusion_ablate.sh"

# Function to run experiment
run_experiment() {
    local dtype=$1
    local parallel=$2
    local model=$3
    local patch=$4
    local batch=$5
    local activ=$6
    local do_compile=$7
    
    # Start timing
    local start_time=$(date +%s)
    
    # Increment experiment counter
    ((current_experiment++))
    
    # Calculate progress percentage
    local progress=$((current_experiment * 100 / total_experiments))
    
    # Parse parallel configuration
    IFS=',' read -r replicate_degree shard_degree <<< "$parallel"
    
    # Create simplified experiment name (lowercase model name only)
    exp_name=$(echo "$model" | tr '[:upper:]' '[:lower:]' | sed 's/DiT-//')"_bs${batch}_ps${patch}_act${activ}_c${do_compile}_d${dtype}_p${parallel}"    

    # Create TB folder with datetime
    tb_folder="ablations_${start_datetime}"
    
    # Construct command with modified dtype handling
    local cmd_dtype="$dtype"
    if [ "$dtype" = "bfloat16_f8" ]; then
        cmd_dtype="bfloat16"
    fi

    cmd="$BASE_CMD \
        --training.mixed_precision_param=$cmd_dtype \
        --training.data_parallel_replicate_degree=$replicate_degree \
        --training.data_parallel_shard_degree=$shard_degree \
        --model.flavor=$model \
        --model.patch_size=$patch \
        --training.batch_size=$batch \
        --activation_checkpoint.mode=$activ \
        --metrics.experiment_name=$exp_name \
        --metrics.enable_tensorboard \
        --metrics.save_tb_folder=$tb_folder \
        --metrics.log_freq=10"

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
    echo -e "${current_experiment}\t${total_experiments}\t${progress}%\t${exp_name}\t${duration}s\t${avg_time_per_exp}s\t${estimated_completion_time}" >> "$progress_file"
    
    # Wait for the process to finish
    wait
    
    # Check for OOM error in exit code
    if [ $exit_code -eq 1 ]; then
        return 1
    fi
    return 0
}

# Run all combinations
for dtype in "${dtypes[@]}"; do
    for parallel in "${parallel_modes[@]}"; do
        for model in "${model_sizes[@]}"; do
            for patch in "${patch_sizes[@]}"; do
                for activ in "${activation_checkpointing[@]}"; do
                    for do_compile in "${compile_modes[@]}"; do
                        hit_oom=false
                        for batch in "${batch_sizes[@]}"; do
                            if [ "$hit_oom" = true ]; then
                                # Skip running but still increment counters
                                ((current_experiment++))
                                continue
                            fi
                            run_experiment "$dtype" "$parallel" "$model" "$patch" "$batch" "$activ" "$do_compile"
                            if [ $? -eq 1 ]; then
                                hit_oom=true
                            fi
                        done
                    done
                done
            done
        done
    done
done