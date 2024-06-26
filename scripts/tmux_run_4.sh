#!/bin/bash
CUDA_DEVICES=(1 2 4 5 6)

# Define parameter values
# trainset_portion=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
# trainset_portion=(0.2 0.3 0.4 0.6 0.7 0.8 0.9)
trainset_portion=(0.1 0.5 1.0)
# poisoning_rate=(0.001 0.002 0.005 0.01)
# poisoning_rate=(0.002 0.005 0.01)
poisoning_rate=(0.01)
model_name="resnet18"
epochs=200
lr=0.1
optimizer_name=sgd

# Create an array to store job commands
jobs=()


# Generate combinations of parameters and create job commands
for prate in ${poisoning_rate[@]}; do
for portion in ${trainset_portion[@]}; do
if [[ $(echo "$portion > $prate" |bc -l) ]]
then
    command="python badnet_attack.py --trainset_portion $portion --epochs $epochs --poisoning_rate $prate --model_name $model_name --lr $lr --optimizer_name $optimizer_name"
    # command="echo $portion $prate"
    echo -e "$command\n"
    jobs+=("$command")
fi
done
done

# jobs=(
#     "CUDA_VISIBLE_DEVICES=3 python badnet_v2.py --trainset_portion 0.01 --epochs 1 --poisoning_rate 0.001"
#     "CUDA_VISIBLE_DEVICES=4 python badnet_v2.py --trainset_portion 0.01 --epochs 1 --poisoning_rate 0.001"
#     "CUDA_VISIBLE_DEVICES=5 python badnet_v2.py --trainset_portion 0.01 --epochs 1 --poisoning_rate 0.001"
#     "CUDA_VISIBLE_DEVICES=4 python badnet_v2.py --trainset_portion 0.01 --epochs 1 --poisoning_rate 0.001"
# )

# Maximum number of parallel jobs
max_parallel_jobs=10

# Function to get the number of running tmux jobs
get_running_jobs_count() {
    tmux list-sessions 2>/dev/null | grep -c "^job_"
}

# Function to start a job in tmux
start_job() {
    job_command=$1
    job_name=$2
    job_index=$3
    CUDA_IDX=$(($job_index % ${#CUDA_DEVICES[@]}))
    # echo -e "$job_index $CUDA_IDX"
    echo -e "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES[CUDA_IDX]} $job_command\n"
    tmux new-session -d -s $job_name "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES[i]} $job_command; tmux kill-session -t $job_name"
}

# Start initial batch of jobs
for i in "${!jobs[@]}"; do
    while [ "$(get_running_jobs_count)" -ge "$max_parallel_jobs" ]; do
        sleep 1
    done
    job_name="job_$((i + 1))"
    start_job "${jobs[$i]}" "$job_name" "$i"
done

# Function to wait for all tmux sessions to finish
wait_for_jobs() {
    while [ "$(get_running_jobs_count)" -gt 0 ]; do
        sleep 1
    done
}

# Wait for all jobs to complete
wait_for_jobs

echo "All jobs completed."