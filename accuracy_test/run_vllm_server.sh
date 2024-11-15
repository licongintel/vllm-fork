#!/bin/bash -ex
###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################


# Kill vLLM if present
pkill -f 'python -m vllm.entrypoints.openai.api_server' && sleep 15
pkill -9 python
pkill -9 pt_main_thread
pkill -9 RAY
pkill -9 sleep
pkill -9 ray::IDLE
pkill -9 ray::RayWorkerW
pkill -9 tpc-clang


script_dir=$(dirname "$(realpath "${BASH_SOURCE[0]}")")

model="mixtral-8x7b"
tensor_parallel=1
eager_mode="Off"
load_balancer="Off"
delay_sampling="Off"
fp8="Off"
batch_size=68
input_len=1024
output_len=1024
steps=4
num_scheduler_steps=80
dtype=auto
kv_cache_dtype=auto
gpu_memory_utilization=0.95
output_dir="$script_dir/results"
port=8080


usage() {
    echo "Options:"
    echo "  --model, -m                     Specify the model, possible choices: [llama2-70b, llama2-7b, llama3-8b-instruct], default: $model"
    echo "  --batch_size,-b                 Specify the batch size, default: $batch_size"
    echo "  --tensor_parallel, -t           Specify the number of HPUs, default: $tensor_parallel"
    echo "  --eager_mode                    Turn On or Off eager mode, choices: [On, Off], default: $eager_mode"
    echo "  --load_balancer                 Turn On or Off load balancer, choices: [On, Off], default: $load_balancer"
    echo "  --fp8                           Enable or Disable fp8/quantization, choices: [On, Off], default: $fp8"
    echo "  --input_len, -i                 Specify the size of prompt sequence bucket, default: $input_len"
    echo "  --output_len, -o                Specify the size of output sequence, default: $output_len"
    echo "  --steps, -s                     Specify number of steps, default: $steps"
    echo "  --num_scheduler_steps, -st       Specify number of scheduler steps, default: $num_scheduler_steps"
    echo "  --dtype, -d                     Specify dtype, default: $dtype"
    echo "  --kv_cache_dtype, -k            Specify kv_cache_dtype, default: $kv_cache_dtype"
    echo "  --gpu_memory_utilization, -u    Specify gpu_memory_utilization, default: $gpu_memory_utilization"
    echo "  --enable_delayed_sampling, -eds Enable delayed sampling, default: $delay_sampling"
    echo "  --output_dir                    Specify the output dir for logs, default: $output_dir"
    echo "  --port, -p                      Port to use, default: $port"
    echo "  --help, -h                      Display this help message"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|-m)
            model=$2
            shift 2
            ;;
        --bs|-b)
            batch_size=$2
            shift 2
            ;;
        --output_dir)
            output_dir=$2
            shift 2
            ;;
        --tensor_parallel|-t)
            tensor_parallel=$2
            shift 2
            ;;
        --eager_mode)
            eager_mode=$2
            shift 2
            ;;
        --load_balancer)
            load_balancer=$2
            shift 2
            ;;
        --enable_delayed_sampling|-eds)
           delay_sampling="On"
            shift 1
            ;;
        --fp8)
            fp8=$2
            shift 2
            ;;
        --input_len|-i)
            input_len=$2
            shift 2
            ;;
        --output_len|-o)
            output_len=$2
            shift 2
            ;;
        --dtype|-d)
            dtype=$2
            shift 2
            ;;
        --kv_cache_dtype|-k)
            kv_cache_dtype=$2
            shift 2
            ;;
        --gpu_memory_utilization|-u)
            gpu_memory_utilization=$2
            shift 2
            ;;
        --num_scheduler_steps|-st)
            num_scheduler_steps=$2
            shift 2
            ;;
        --steps|-s)
            steps=$2
            shift 2
            ;;
        --port|-p)
            port=$2
            shift 2
            ;;
        --profile)
            export VLLM_PROFILER_ENABLED=true
            shift 1
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "Invalid option: $1"
            exit 1
            ;;
    esac
done

if [[ -n $HELP || -z $model || -z $batch_size ]]; then
    usage
fi

selected_model=$model

case $model in
    "llama3-8b-instruct")
    model="/mnt/weka/data/pytorch/llama3/Meta-Llama-3-8B-Instruct/"
    ;;
    "llama3-70b-instruct")
    model="/mnt/weka/data/pytorch/llama3/Meta-Llama-3-70B-Instruct/"
    ;;
    "llama3.1-8b-instruct")
    model="/mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-8B-Instruct/"
    ;;
    "llama3.1-70b-instruct")
    model="/mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-70B-Instruct/"
    ;;
    "llama3.1-405b-instruct")
    model="/mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-405B-Instruct/"
    ;;
    "mixtral-8x7b")
    model="/root/ckpt/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/41bd4c9e7e4fb318ca40e721131d4933966c2cc1/"
    ;;
esac


if [[ $eager_mode == "On" ]]; then
    EAGER_FLAG="--enforce-eager"
else
    export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
fi

if [[ $delay_sampling == "On" ]]; then
    DELAY_SAMPLING_FLAG="--enable-delayed-sampling  --num-lookahead-slots 1  --use-v2-block-manager"
else
    DELAY_SAMPLING_FLAG=""
fi

if [[ $fp8 == "On" ]]; then
    QUANT_FLAGS="--quantization hqt --kv-cache-dtype hf8 --weights-load-device cpu"
    case $selected_model in
        "llama3-8b-instruct")
        export QUANT_CONFIG=hqt/llama3-8b/quantization_config/maxabs_quant.json
        ;;
        "llama3-70b-instruct")
        QUANT_FLAGS="--quantization inc --kv-cache-dtype fp8_inc --weights-load-device cpu"
        export QUANT_CONFIG=hqt/llama3-70b-8x/quantization_config/maxabs_quant.json
        ;;
        "llama3.1-8b-instruct")
        QUANT_FLAGS="--quantization inc --kv-cache-dtype fp8_inc --weights-load-device cpu"
        export QUANT_CONFIG=hqt/llama3.1-8b-1x/quantization_config/maxabs_quant.json
        ;;
        "llama3.1-70b-instruct")
        QUANT_FLAGS="--quantization inc --kv-cache-dtype fp8_inc --weights-load-device cpu"
        export QUANT_CONFIG=hqt/llama3.1-70b-8x/quantization_config/maxabs_quant.json
        ;;
        "mixtral-8x7b")
        export QUANT_CONFIG=hqt/mixtral-8x7b/quantization_config/maxabs_quant.json
        ;;
    esac
fi

if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

export OPENAI_API_KEY=secret_abcdefg
export OPENAI_API_BASE="http://localhost:$port/v1"

export VLLM_ENGINE_ITERATION_TIMEOUT_S=600

export VLLM_PROMPT_BS_BUCKET_MIN=1
export VLLM_PROMPT_BS_BUCKET_STEP=1
export VLLM_PROMPT_BS_BUCKET_MAX=2

export VLLM_PROMPT_SEQ_BUCKET_MIN=$input_len
export VLLM_PROMPT_SEQ_BUCKET_STEP=$input_len
export VLLM_PROMPT_SEQ_BUCKET_MAX=$input_len

export VLLM_DECODE_BS_BUCKET_MIN=$batch_size
export VLLM_DECODE_BS_BUCKET_STEP=$batch_size
export VLLM_DECODE_BS_BUCKET_MAX=$batch_size

block_bucket_min=$(($batch_size * $input_len/128))
block_bucket_max=$((($batch_size * ($input_len+$output_len)/128) + (3 * $steps)))
export VLLM_DECODE_BLOCK_BUCKET_MIN=$block_bucket_min
export VLLM_DECODE_BLOCK_BUCKET_STEP=$steps
export VLLM_DECODE_BLOCK_BUCKET_MAX=$block_bucket_max

export VLLM_PROMPT_USE_FUSEDSDPA=true
export VLLM_PA_SOFTMAX_IMPL=const
export VLLM_CONTIGUOUS_PA=true

max_model_len=$(($input_len + $output_len))
max_batched_token=$(($batch_size * $input_len))


python -m vllm.entrypoints.openai.api_server --port $port \
        --model $model \
        --tensor-parallel-size $tensor_parallel \
        --max-num-seqs $batch_size \
        --disable-log-requests \
        --dtype $dtype \
        --max-model-len $max_model_len \
        --gpu-memory-utilization $gpu_memory_utilization \
        --num-scheduler-steps $num_scheduler_steps \
        --trust-remote-code \
        --kv-cache-dtype $kv_cache_dtype \
        --num-lookahead-slots 1  \
        --use-v2-block-manager \
        --max-num-batched-tokens $max_batched_token \
        $EAGER_FLAG \
        $DELAY_SAMPLING_FLAG \
        $QUANT_FLAGS >> ${output_dir}/vllm_server.log 2>&1 &

VLLM_PID=$!

set

wait_for_server() {
    local port="$1"
    local model="$2"

    timeout=10800
    step=10
    current_time=0

    while [ "$current_time" -lt "$timeout" ]; do
        output=$(curl -s http://localhost:$port/v1/models | grep $model | wc -l)
        if (( $output > 0 )); then
            echo "vLLM server on port $port started"
            return 0
        fi
        if [ ! -d "/proc/${VLLM_PID}" ]; then
            echo "vLLM server on port $port failed to start"
            return -1
        fi

        current_date_time="`date +%Y-%m-%d_%H:%M:%S`";
        echo "$current_date_time: Waiting for vLLM server to start on port $port, wait_time=$current_time"
        sleep $step
        current_time=$((current_time + step))
    done

    echo "vLLM server on port $port didn't start"
    return -1
}

wait_for_server $port $model
if [[ $? -ne 0 ]]; then
    echo "Error: Server on port $port failed to start."
    exit 1
fi

