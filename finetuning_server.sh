#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

BASE_PATH="/home/niteshkumar/research/conceptual-spaces"

cd $BASE_PATH

conda activate llm

export PYTHONPATH="${PYTHONPATH}:${BASE_PATH}"

if [ "$1" == "queue" ]; then

    {
        # python $BASE_PATH/models/classifier_using_causalllm.py Meta-Llama-3-8B
        python $BASE_PATH/models/classifier_using_causalllm.py Qwen3-8B
        python $BASE_PATH/models/classifier_using_causalllm.py Qwen3-14B
        python $BASE_PATH/models/classifier_using_causalllm.py Mistral-Nemo-Base-2407
        python $BASE_PATH/models/classifier_using_causalllm.py Mistral-Small-24B-Base-2501
        python $BASE_PATH/models/classifier_using_causalllm.py Phi-4
        python $BASE_PATH/models/classifier_using_causalllm.py OLMo-2-1124-7B
        python $BASE_PATH/models/classifier_using_causalllm.py OLMo-2-1124-13B
    } >> $BASE_PATH/logs/classifier_queue_std.log 2>&1

else
    echo "Invalid argument. Please use 'queue'."
fi
