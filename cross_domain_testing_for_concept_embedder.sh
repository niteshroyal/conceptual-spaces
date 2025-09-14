#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

BASE_PATH="/home/niteshkumar/research/conceptual-spaces"

cd $BASE_PATH

conda activate llm

export PYTHONPATH="${PYTHONPATH}:${BASE_PATH}"

if [ "$1" == "queue" ]; then

    {
        # python $BASE_PATH/experiments/testing_concept_embedder_cross_domain.py Meta-Llama-3-8B tastes 75
        # python $BASE_PATH/experiments/testing_concept_embedder_cross_domain.py Meta-Llama-3-8B tastes 100
        python $BASE_PATH/experiments/testing_concept_embedder_cross_domain.py Meta-Llama-3-8B tastes 123
        # python $BASE_PATH/experiments/testing_concept_embedder_cross_domain.py Meta-Llama-3-8B tastes 150
        # python $BASE_PATH/experiments/testing_concept_embedder_cross_domain.py Meta-Llama-3-8B tastes 175
        # python $BASE_PATH/experiments/testing_concept_embedder_cross_domain.py Meta-Llama-3-8B tastes 200
        # python $BASE_PATH/experiments/testing_concept_embedder_cross_domain.py Meta-Llama-3-8B tastes 250
        # python $BASE_PATH/experiments/testing_concept_embedder_cross_domain.py Meta-Llama-3-8B tastes 300
        # python $BASE_PATH/experiments/testing_concept_embedder_cross_domain.py Meta-Llama-3-8B tastes 350
        # python $BASE_PATH/experiments/testing_concept_embedder_cross_domain.py Meta-Llama-3-8B tastes 400
        # python $BASE_PATH/experiments/testing_concept_embedder_cross_domain.py Meta-Llama-3-8B tastes 450
        # python $BASE_PATH/experiments/testing_concept_embedder_cross_domain.py Meta-Llama-3-8B wikidata1 123
        # python $BASE_PATH/experiments/testing_concept_embedder_cross_domain.py Mistral-Small-24B-Base-2501 wikidata1 123
    } >> $BASE_PATH/logs/cross_domain_testing_for_concept_embedder.log 2>&1

else
    echo "Invalid argument. Please use 'queue'."
fi
