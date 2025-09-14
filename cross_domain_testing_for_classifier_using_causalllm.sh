#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

BASE_PATH="/home/niteshkumar/research/conceptual-spaces"

cd $BASE_PATH

conda activate llm

export PYTHONPATH="${PYTHONPATH}:${BASE_PATH}"

if [ "$1" == "queue" ]; then

    {
        # python $BASE_PATH/experiments/testing_classifier_using_causalllm_cross_domain.py Meta-Llama-3-8B rocks
        # python $BASE_PATH/experiments/testing_classifier_using_causalllm_cross_domain.py Mistral-Small-24B-Base-2501 rocks
        # python $BASE_PATH/experiments/testing_classifier_using_causalllm_cross_domain.py Meta-Llama-3-8B odours
        # python $BASE_PATH/experiments/testing_classifier_using_causalllm_cross_domain.py Mistral-Small-24B-Base-2501 odours
        # python $BASE_PATH/experiments/testing_classifier_using_causalllm_cross_domain.py Meta-Llama-3-8B music
        # python $BASE_PATH/experiments/testing_classifier_using_causalllm_cross_domain.py Mistral-Small-24B-Base-2501 music
        # python $BASE_PATH/experiments/testing_classifier_using_causalllm_cross_domain.py Meta-Llama-3-8B movies
        python $BASE_PATH/experiments/testing_classifier_using_causalllm_cross_domain.py Mistral-Small-24B-Base-2501 movies
        # python $BASE_PATH/experiments/testing_classifier_using_causalllm_cross_domain.py Meta-Llama-3-8B books
        python $BASE_PATH/experiments/testing_classifier_using_causalllm_cross_domain.py Mistral-Small-24B-Base-2501 books
        # python $BASE_PATH/experiments/testing_classifier_using_causalllm_cross_domain.py Meta-Llama-3-8B wikidata1
        python $BASE_PATH/experiments/testing_classifier_using_causalllm_cross_domain.py Mistral-Small-24B-Base-2501 wikidata1
        # python $BASE_PATH/experiments/testing_classifier_using_causalllm_cross_domain.py Meta-Llama-3-8B wikidata2
        python $BASE_PATH/experiments/testing_classifier_using_causalllm_cross_domain.py Mistral-Small-24B-Base-2501 wikidata2
        # python $BASE_PATH/experiments/testing_classifier_using_causalllm_cross_domain.py Meta-Llama-3-8B physical
        python $BASE_PATH/experiments/testing_classifier_using_causalllm_cross_domain.py Mistral-Small-24B-Base-2501 physical
    } >> $BASE_PATH/logs/cross_domain_testing_for_classifier_using_causalllm.log 2>&1

else
    echo "Invalid argument. Please use 'queue'."
fi
